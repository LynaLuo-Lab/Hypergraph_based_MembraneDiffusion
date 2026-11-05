"""Generalized multi-lag model.

Pipeline:
  GAT geometry embed -> Multi-Lag LSTM encoder -> transition head
    -> stochastic head (brownian | ou | ou_gmm | ou_nsf)
    -> decoder (mlp | xformer | kan) -> per-step Delta_x -> wrap

Decoder and transition head are selected via config:
  decoder_type:          'mlp' | 'xformer' | 'kan'
  transition_head_type:  'mlp_gauss' | 'nsf' | 'kan'
  latent_type:           'brownian' | 'ou' | 'ou_gmm' | 'ou_nsf'

Multi-lag encoder config:
  n_lag_lstms:      up to len(ALL_LAGS); uses the first n lags. 0 = no bank: the transition
                    head reads the current frame's geometry embedding directly.
  lag_lstm_width:   hidden dim for every lag-LSTM (32 | 64 | 128 typical).

Lag selection rule: each LSTM must fire enough times per training sample to
learn its timescale's dynamics (rule of thumb: >= ~17 firings for stable
BPTT). At seq=5000 that caps useful lag at ~300. Lags of O(1000+) degenerate
to a sparse 2-point memory -- bad use of an LSTM. Use long seq_len +
long-lag losses for long-horizon learning instead.
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt_util

from membrane.base_model import (BaseBrownianModel, GAT_GeometryEmbedding, GNN_GeometryEmbedding,
                                          RotationHead, arm_pair_feature_3d, dlogpair_kl,
                                          ic_once_node_features, logpair_kl)
from membrane.geometry_features import cov_from_blades, rodrigues
from membrane.splat_encoder import SplatEncoder
from membrane.squash_head import (RadialSquashHead,
                                            minimal_contraction_target)
from membrane.membrane_encoder import MembranePatchEncoder
from membrane.membrane_hypergraph_module import DEPTH_MODES, MembraneHypergraphModule
from data.paths import resolve, resolve_all
from nbcore.transition_heads import build_transition_head
from nbcore.decoders import build_decoder


ALL_LAGS = [1, 3, 10, 30, 100, 200, 300]

# Frames handed to MembraneHypergraphModule._calibrate_inputs. The whole 3-part
# 7piezo pool is 18.7k frames and _geom over it takes well under a second, so the cap
# only bounds the cost on a bigger pool; the stride keeps every part represented in
# proportion to its length instead of truncating the pool at the first N frames.
_CALIB_FRAMES = 20000


def _calib_frames(_np, parts):
    """Strided sample of (coords, node_depth, hyper_depth) over the pool, with the
    depths divided by the POOLED std -- the exact unit data/membrane_dataset.py feeds
    the model. Taking the std per file instead would be wrong by ~10% (the per-file
    node_std_norm is 30.48 against the pooled 27.84)."""
    cs = [_np.load(p)['coords'] for p in parts]
    ns = [_np.load(p)['node_depth'] for p in parts]
    hs = [_np.load(p)['hyper_depth'] for p in parts]
    total = sum(c.shape[0] for c in cs)
    stride = max(1, -(-total // _CALIB_FRAMES))
    nstd = float(_np.concatenate([n.ravel() for n in ns]).std()) or 1.0
    hstd = float(_np.concatenate([h.ravel() for h in hs]).std()) or 1.0
    return (_np.concatenate([c[::stride] for c in cs], 0),
            _np.concatenate([n[::stride] for n in ns], 0) / nstd,
            _np.concatenate([h[::stride] for h in hs], 0) / hstd)


def _lag1_autocorr(_np, arrs):
    """Pooled lag-1 autocorrelation of the depth VALUE. The POOLED mean is removed,
    not the per-channel one, so the static between-protein offsets (76% of node depth
    variance) stay inside the signal -- they are exactly the structure the OU's mu has
    to hold, so the relaxation rate must be measured against them. Scale-free, so it
    does not matter that the arrays are the raw npz values. Per-part, so no lag-1 pair
    ever crosses a part boundary."""
    mu = _np.concatenate([a.ravel() for a in arrs]).mean()
    num = den = 0.0
    for x in arrs:
        xm = x - mu
        num += float((xm[1:] * xm[:-1]).sum())
        den += float((xm * xm).sum())
    return num / den if den else 0.0


class MultiLagLSTMEncoder(nn.Module):
    """Bank of single-layer LSTMs, each firing at its own lag interval.

    At step t:
      - If t == 0 or t % lag == 0: feed the current GAT embedding, advance hidden state.
      - Otherwise: keep hidden state frozen (no-op for that LSTM).

    Output: concatenation of all LSTM hidden states -> LayerNorm -> transition_head.
    """

    def __init__(self, input_dim, hidden_dim=64, latent_dim=21,
                 noise_rank=16, n_lag_lstms=4, dropout=0.0,
                 transition_head_type='mlp_gauss', config=None):
        super().__init__()
        self.lags = ALL_LAGS[:n_lag_lstms]
        self.n_lag_lstms = n_lag_lstms
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.noise_rank = noise_rank

        # Compatibility attr (some base-class helpers read .num_layers).
        self.num_layers = 1

        self.ln_embed = nn.LayerNorm(input_dim)

        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                    num_layers=1, batch_first=True)
            for _ in self.lags
        ])
        # Forget-gate bias -> 1.0 (standard LSTM initialization trick)
        for lstm in self.lstms:
            for name, p in lstm.named_parameters():
                if "bias" in name:
                    n = p.size(0)
                    p.data[n // 4: n // 2].fill_(1.0)

        total_hidden = hidden_dim * n_lag_lstms
        self.ln_out = nn.LayerNorm(total_hidden)

        self.transition_head = build_transition_head(
            kind=transition_head_type,
            input_dim=total_hidden,
            latent_dim=latent_dim,
            noise_rank=noise_rank,
            config=config or {},
        )
        # Not compiled separately -- absorbed into the outer step-function
        # compile (see model_multilag.BrownianModel.__init__).

    def init_state(self, batch_size, device):
        """Return a list of (h, c) tuples — one per lag-LSTM."""
        return [
            (torch.zeros(1, batch_size, self.hidden_dim, device=device),
             torch.zeros(1, batch_size, self.hidden_dim, device=device))
            for _ in self.lags
        ]

    def forward(self, embedded_x, states, step_idx):
        x = self.ln_embed(embedded_x)  # [B, 1, input_dim]

        new_states = []
        hidden_vecs = []
        for i, (lag, lstm) in enumerate(zip(self.lags, self.lstms)):
            h_prev, c_prev = states[i]
            if step_idx == 0 or step_idx % lag == 0:
                _, (h_new, c_new) = lstm(x, (h_prev, c_prev))
                new_states.append((h_new, c_new))
                hidden_vecs.append(h_new.squeeze(0))
            else:
                new_states.append((h_prev, c_prev))
                hidden_vecs.append(h_prev.squeeze(0))

        concat = torch.cat(hidden_vecs, dim=-1)
        concat = self.ln_out(concat)

        params = self.transition_head(concat)
        # Bounded mu prevents runaway drift at init; tanh smooths the bound.
        params = params._replace(mu=10 * torch.tanh(params.mu / 10))
        return params, new_states


class DirectEncoder(nn.Module):
    """No bank (n_lag_lstms=0): the transition head reads the current frame's geometry
    embedding, normalised as one vector like the bank's concatenated states. What memory
    is left is the latent's own OU decay and the membrane module's state."""

    def __init__(self, input_dim, latent_dim=21, noise_rank=16,
                 transition_head_type='mlp_gauss', config=None):
        super().__init__()
        self.lags = []
        self.n_lag_lstms = 0
        self.hidden_dim = 0
        self.latent_dim = latent_dim
        self.noise_rank = noise_rank
        self.num_layers = 1
        self.ln_out = nn.LayerNorm(input_dim)
        self.transition_head = build_transition_head(
            kind=transition_head_type,
            input_dim=input_dim,
            latent_dim=latent_dim,
            noise_rank=noise_rank,
            config=config or {},
        )

    def init_state(self, batch_size, device):
        return []

    def forward(self, embedded_x, states, step_idx):
        params = self.transition_head(self.ln_out(embedded_x[:, 0]))
        params = params._replace(mu=10 * torch.tanh(params.mu / 10))
        return params, states


class BrownianModel(BaseBrownianModel):
    """GAT -> Multi-Lag LSTM bank -> transition head -> stochastic head -> decoder."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        hd = config.get('hidden_dim', 32)
        nh = config.get('num_heads', 8)
        geo_type = config.get('geo_embed_type', 'gat')
        # geo_ic_once: absolute position for the IC frame only, step deltas after, in
        # the embedding and in the tokenized decoder's position keys. geo_ic_axes='xy'
        # leaves z absolute on every frame; 'xyz' drops it with the other two.
        self.geo_ic_once = bool(config.get('geo_ic_once', False))
        self.geo_ic_axes = config.get('geo_ic_axes') or 'xyz'
        if self.geo_ic_axes not in ('xyz', 'xy'):
            raise ValueError(f"geo_ic_axes={self.geo_ic_axes!r}; expected 'xyz' or 'xy'")
        if config.get('geo_ic_axes') is not None and not self.geo_ic_once:
            raise ValueError("geo_ic_axes is set but geo_ic_once is off, so nothing reads it")
        if self.geo_ic_once and self.geom_splat:
            raise ValueError("geo_ic_once with geom_arm='splat': the splat grid is "
                             "rasterized from absolute positions on every step")
        # geo_pair_dir_shells: per-protein x,y directions to the neighbours, by distance
        # shell, in the same node features. Without it a protein sees how far each
        # neighbour is and never which way, so no step can be aimed at one.
        shells = config.get('geo_pair_dir_shells', 0) or 0
        if isinstance(shells, bool) or not isinstance(shells, int) or shells < 0:
            raise ValueError(f"geo_pair_dir_shells={shells!r}; expected a whole number of shells, 0 = off")
        if shells and not self.geo_ic_once:
            raise ValueError("geo_pair_dir_shells is set but geo_ic_once is off; the directions "
                             "are part of the geo_ic_once node features")
        self.geo_pair_dir_shells = shells
        if geo_type == 'gnn':
            if self.geo_ic_once:
                raise ValueError("geo_ic_once is implemented for the GAT embedding only")
            self.geo_embed = GNN_GeometryEmbedding(
                box_size=self.box_size, hidden_dim=hd, num_heads=nh,
                n_particles=self.n_particles)
        else:
            self.geo_embed = GAT_GeometryEmbedding(
                box_size=self.box_size, hidden_dim=hd, num_heads=nh,
                n_particles=self.n_particles, ic_once=self.geo_ic_once,
                ic_keep_z=self.geo_ic_axes == 'xy', pair_dir_shells=self.geo_pair_dir_shells)

        # When arms are on (or geom_arm='moments' appends covariance moments), the
        # geometry embedding fed to the ENCODER is augmented with arm_feat_dim extra
        # features, so arm/shape info flows into the latent z (and thence the decoded
        # step). The decoder context stays the membrane context (geo.output_dim),
        # unchanged -- any arm_feat_dim>0 relies on that separate context path.
        emb_dim = self.geo_embed.output_dim + self.arm_feat_dim
        # Arm C: one CNN over a density grid rasterized from centres+blades. It runs
        # INSIDE the AR loop like every other geometry feature, so it sees the model's
        # own predicted positions and its own advanced blade state, not the GT.
        if self.geom_splat:
            self.splat_encoder = SplatEncoder(
                out_dim=self.splat_out_dim, n_particles=self.n_particles,
                box_size=self.box_size, grid=int(config.get('splat_grid', 69)),
                z_slabs=int(config.get('splat_z_slabs', 5)),
                sigma=float(config.get('splat_sigma', 8.0)),
                hidden=int(config.get('splat_hidden', 32)))
        if self.arm_feat_dim > 0 and not config.get('membrane_mode', False):
            raise ValueError('arms_enabled/geom_arm requires membrane_mode=True '
                             '(decoder context path assumes the membrane branch)')
        # geom_arm capabilities come from the GEOM_ARMS table (geometry_features.py),
        # never from re-testing the arm string here.
        #  * needs_blade: without geom_blade_enabled=True, DataModule/training_step
        #    never deliver `blade`, so the arm would train on the fallback template
        #    for every batch -- i.e. a baseline wearing the arm's name.
        #  * needs_arms: every se3 code path sits under `if self.arms_enabled`, and
        #    se3 does not widen arm_feat_dim, so arms_enabled=False made it silently
        #    inert rather than an error.
        if self.geom_spec['needs_blade'] and not self.geom_blade_enabled:
            raise ValueError(f"geom_arm={self.geom_arm!r} requires geom_blade_enabled=True "
                             f"(set geom_blade_enabled=True, or geom_arm='none')")
        if self.geom_spec['needs_arms'] and not self.arms_enabled:
            raise ValueError(f"geom_arm={self.geom_arm!r} requires arms_enabled=True "
                             f"(its code paths all sit under arms_enabled, so it would "
                             f"be silently inert)")

        # Multi-lag LSTM encoder with configurable transition head
        if config.get('n_lag_lstms', 4) == 0:
            self.encoder = DirectEncoder(
                input_dim=emb_dim,
                latent_dim=self.latent_dim,
                noise_rank=self.noise_rank,
                transition_head_type=config.get('transition_head_type', 'mlp_gauss'),
                config=config,
            )
        else:
            self.encoder = MultiLagLSTMEncoder(
                input_dim=emb_dim,
                hidden_dim=config.get('lag_lstm_width', 64),
                latent_dim=self.latent_dim,
                noise_rank=self.noise_rank,
                n_lag_lstms=config.get('n_lag_lstms', 4),
                dropout=config.get('dropout', 0.1),
                transition_head_type=config.get('transition_head_type', 'mlp_gauss'),
                config=config,
            )

        # Decoder (factory-selected) — context is the membrane ctx (geo.output_dim).
        # xformer_tokens needs position and membrane deltas threaded through the
        # rollout; every other decoder keeps the old two-argument call exactly.
        self.dec_tokenized = config.get('decoder_type', 'mlp') == 'xformer_tokens'
        self.decoder = build_decoder(
            kind=config.get('decoder_type', 'mlp'),
            gat_embed_dim=self.geo_embed.output_dim,
            latent_dim=self.latent_dim,
            n_coords=self.n_coords,
            config=config,
        )
        if self.geo_pair_dir_shells and self.dec_tokenized:
            # the position keys carry the embedding's node features, directions included
            self.decoder.pos_feat_dim = self.geo_embed.node_input_dim
            self.decoder.pos_proj = nn.Linear(self.decoder.pos_feat_dim,
                                              self.decoder.pos_proj.out_features)

        # Radial squash head: contracts the closing component of the proposed step
        # against the nearest neighbour. Off by default -- this is an experimental
        # route to excluded volume that survives at inference without ev_force, and
        # it must not perturb any existing arm until it is asked for. See
        # membrane/squash_head.py for why the frame is radial.
        self.squash_enabled = bool(config.get('squash_enabled', False))
        self.squash_head = None
        self.squash_trace = None          # set to [] to collect (s, target) per step
        # Joint training: [] collects (s_pred, target) with s_pred CARRYING GRAD,
        # so the head learns inside the normal backward instead of in a second,
        # frozen-backbone pass. Distinct from squash_trace, which detaches
        # everything because there the backbone must not move.
        self.squash_aux = None
        if self.squash_enabled:
            self.squash_head = RadialSquashHead(
                self.n_particles, self.box_size,
                latent_dim=(self.latent_dim if config.get('squash_use_latent', False) else 0),
                edge_hidden=int(config.get('squash_edge_hidden', 32)),
                hidden=int(config.get('squash_hidden', 32)),
                pool=config.get('squash_pool', 'max'),
                asymmetric=bool(config.get('squash_asymmetric', True)),
                init_bias=float(config.get('squash_init_bias', 6.0)),
                cutoff_cores=float(config.get('squash_cutoff_cores', 2.0)),
                core_free=bool(config.get('squash_core_free', False)),
                cutoff_frac=float(config.get('squash_cutoff_frac', 0.25)))

        # Per-protein rotational step head (reads the arm-augmented embedding,
        # decoupled from the position latent).
        if self.arms_enabled:
            self.rotation_head = RotationHead(
                emb_dim, self.n_particles, sigma_init=self.arm_sigma_theta,
                dof=self.geom_spec['dof'])

        # --- Membrane conditioning (position-sampled) -------------------------
        # When on, the xformer decoder's cross-attention key/value is a MEMBRANE
        # context (replacing geo curr_emb) sampled at the predicted position from
        # the live mesh; the encoder/latent still see geo. Mesh is a non-persistent
        # buffer (not in checkpoints); indexed cyclically so rollouts of any length
        # / arbitrary IC stay well-defined.
        self.membrane_mode = bool(config.get('membrane_mode', False))
        # New autoregressive hypergraph membrane (decision d4ba47f5): predicts the
        # k=2..7 membrane-depth hypergraph forward and emits the decoder context.
        # Supersedes the GT-mesh-reading MembranePatchEncoder path below.
        self.membrane_hypergraph = bool(config.get('membrane_hypergraph', False))
        if self.z_from_depth:
            if not (self.membrane_mode and self.membrane_hypergraph) or not self.mem_consistency_slope:
                raise ValueError('z_from_depth needs the hypergraph membrane and membrane_consistency_slope = box/node_std')
            self.depth_raw = self.box_size / self.mem_consistency_slope
        if self.mem_loss_units == 'delta' and not (self.membrane_mode and self.membrane_hypergraph):
            # The rescale factors come off the membrane npz, and nothing scores a depth
            # outside hypergraph mode -- so this combination is inert, and an inert
            # sweep arm that looks like data is the failure mode KNOWN_KEYS exists for.
            raise ValueError("membrane_loss_units='delta' is inert without "
                             "membrane_mode+membrane_hypergraph: there is no depth loss "
                             "to change the units of")
        if self.membrane_mode and self.membrane_hypergraph:
            import numpy as _np
            # Pooled runs set membrane_data_paths (plural) and drop the singular key.
            # Only members/ksize are read here and the hypergraph structure is shared
            # across parts (verified byte-identical), so any part serves.
            _mp = resolve(config.get('membrane_data_path') or config['membrane_data_paths'][0])
            md = _np.load(_mp, allow_pickle=True)
            # Delta (residual) mode: `step` predicts the normalized per-frame change
            # instead of the absolute depth. Scales default to the measured
            # consecutive-frame delta std of the stored depths (config can override).
            _delta_mode = bool(config.get('membrane_delta_mode', False))
            _ds_node = config.get('membrane_delta_scale_node')
            _ds_hyper = config.get('membrane_delta_scale_hyper')
            # depth_mode names all three propagators ('absolute'|'delta'|'ou') and
            # supersedes the older membrane_delta_mode bool. Defaulting it FROM that
            # bool keeps every existing config meaning exactly what it meant; the two
            # disagreeing is a config bug, not something to silently resolve.
            _depth_mode = config.get('membrane_depth_mode')
            if _depth_mode is None:
                _depth_mode = 'delta' if _delta_mode else 'absolute'
            if _depth_mode not in DEPTH_MODES:
                raise ValueError(f'membrane_depth_mode must be one of {DEPTH_MODES}, '
                                 f'got {_depth_mode!r}')
            if _delta_mode and _depth_mode != 'delta':
                raise ValueError(f"membrane_delta_mode=True contradicts "
                                 f"membrane_depth_mode={_depth_mode!r}")
            _ou = _depth_mode == 'ou'
            # Gate on _depth_mode, not _delta_mode: membrane_depth_mode='delta' with
            # the old bool left unset must still get a MEASURED scale, or the residual
            # integrates at scale 1.0 -- 15x the real per-frame step, the exact
            # mis-scaling 1968e79 fixed (untrained rollout reaches 1e16, not 3.5e3).
            if (_depth_mode in ('delta', 'ou') or self.dec_tokenized
                    or self.mem_loss_units == 'delta'):
                # Pool the delta stats over every part -- taking them from part 0 alone
                # would mis-scale the others, and diffs must not cross a part boundary.
                # Needed even in ABSOLUTE mode when the tokenized decoder is used: it
                # divides its delta tokens by these, so leaving them at 1.0 would feed
                # raw ~1e-4 values next to unit-scale position keys. The OU head needs
                # the same number for its sigma init, and the lag-1 autocorrelation
                # for its `a`, so the depths are loaded once and reused.
                _parts = resolve_all(config.get('membrane_data_paths') or [_mp])
                _nodes = [_np.load(p)['node_depth'] for p in _parts]
                _hypers = [_np.load(p)['hyper_depth'] for p in _parts]
                _dn = [_np.diff(x, axis=0) for x in _nodes]
                _dh = [_np.diff(x, axis=0) for x in _hypers]
                # UNITS. data/membrane_dataset.py:215-218 divides the depths by their
                # POOLED std before the model ever sees them, so the model works where
                # depth has std 1. These scales were measured on the RAW npz and handed
                # over unconverted -- 27.8x too large for node, 19.9x for hyper. In
                # normalized units the whole node range is -8.2..+1.2, so one step was
                # scaled to move depth by ~1.8 of a ~9.4-wide range against a true
                # per-frame move of 0.065. That is why delta_mode diverged at epoch 0
                # (W&B membrane_lossvariants arm0_delta hit the 1e6 sentinel).
                # Pool exactly as the dataset does -- the per-file meta values
                # (node_std_norm 30.48) differ from the pooled 27.84 by ~10%.
                _nstd = float(_np.concatenate(
                    [x.ravel() for x in _nodes]).std()) or 1.0
                _hstd = float(_np.concatenate(
                    [x.ravel() for x in _hypers]).std()) or 1.0
                if _ds_node is None:
                    _ds_node = float(_np.concatenate(_dn, axis=0).std()) / _nstd
                if _ds_hyper is None:
                    _ds_hyper = float(_np.concatenate(_dh, axis=0).std()) / _hstd
                print(f"  membrane delta scales (normalized units): "
                      f"node {_ds_node:.6f} hyper {_ds_hyper:.6f}", flush=True)
            if self.mem_loss_units == 'delta':
                # value_std/delta_std, i.e. the reciprocal of the same measured
                # per-frame step std the residual head integrates -- one measured
                # constant, two uses, so a membrane_delta_scale_* override moves both
                # together and the loss can never disagree with the integrator about
                # what one frame is worth.
                self.mem_loss_scale_node = 1.0 / _ds_node
                self.mem_loss_scale_hyper = 1.0 / _ds_hyper
                print(f"  membrane loss units=delta: node x{self.mem_loss_scale_node:.3f} "
                      f"hyper x{self.mem_loss_scale_hyper:.3f}", flush=True)
            _ou_kw = {}
            _ou_keys = [k for k in ('membrane_ou_noise', 'membrane_ou_a_node',
                                    'membrane_ou_a_hyper', 'membrane_ou_sigma_node',
                                    'membrane_ou_sigma_hyper') if k in config]
            if _ou_keys and not _ou:
                raise ValueError(f"{_ou_keys} set with membrane_depth_mode="
                                 f"{_depth_mode!r}: nothing reads them outside 'ou'")
            if _ou:
                # a = 1 - rho_1 of the depth VALUE, pooled over the parts with the
                # POOLED mean removed -- the same convention the field is normalized
                # under, and the one that keeps the 76%-of-variance static offsets
                # inside the signal the OU has to hold. Demeaning per channel instead
                # would hand back rho_1 of the fluctuation alone (0.991 / 0.984) and
                # so a 4x faster `a`, which is a different process.
                _a_node = config.get('membrane_ou_a_node')
                _a_hyper = config.get('membrane_ou_a_hyper')
                if _a_node is None:
                    _a_node = 1.0 - _lag1_autocorr(_np, _nodes)
                if _a_hyper is None:
                    _a_hyper = 1.0 - _lag1_autocorr(_np, _hypers)
                _sg_node = config.get('membrane_ou_sigma_node')
                _sg_hyper = config.get('membrane_ou_sigma_hyper')
                if _sg_node is None:
                    _sg_node = _ds_node
                if _sg_hyper is None:
                    _sg_hyper = _ds_hyper
                _ou_kw = dict(ou_a_node=_a_node, ou_a_hyper=_a_hyper,
                              # DEFAULT TRUE. With the noise off the update is a
                              # deterministic EMA of mu with time constant 1/a = 441
                              # frames. An ORACLE test (mu = the exact GT depth, the
                              # best any head could do) gives a per-frame step of
                              # 1.09% of GT and a rollout std of 0.031 vs GT 1.000 --
                              # variance collapse BY CONSTRUCTION, which silently
                              # passes a |max|<20 stability check by barely moving.
                              # sigma is calibrated to the measured per-frame step
                              # std, so the noise is what makes it mean anything.
                              ou_noise=bool(config.get('membrane_ou_noise', True)),
                              edge_latent=bool(config.get('membrane_edge_latent', False)),
                              ou_sigma_node=_sg_node, ou_sigma_hyper=_sg_hyper)
                # sigma/sqrt(2a-a^2) is the stationary std the pair implies; it should
                # come out at 1, because that is what the dataset normalized to.
                _s_n = _sg_node / (2 * _a_node - _a_node ** 2) ** 0.5
                _s_h = _sg_hyper / (2 * _a_hyper - _a_hyper ** 2) ** 0.5
                print(f"  membrane OU: a node {_a_node:.6f} hyper {_a_hyper:.6f}; "
                      f"sigma node {_sg_node:.6f} hyper {_sg_hyper:.6f} "
                      f"(noise {'on' if _ou_kw['ou_noise'] else 'off'}); "
                      f"implied stationary std node {_s_n:.3f} hyper {_s_h:.3f}",
                      flush=True)
            # Per-feature input standardization. The stats have to come from the same
            # pool the dataset normalizes over, and the geometry ones can only be got
            # by actually running _geom on real coords, so the frames are sampled here
            # (this is the only place that owns the npz paths) and handed to the
            # module, which measures them. Strided, not random: the stats are
            # non-persistent buffers rebuilt on every load, so they must be a
            # deterministic function of the config alone.
            _std_in = bool(config.get('membrane_standardize_inputs', False))
            _cc = _cn = _ch = None
            if _std_in:
                _parts = resolve_all(config.get('membrane_data_paths') or [_mp])
                _cc, _cn, _ch = _calib_frames(_np, _parts)
                print(f"  membrane input standardization: {_cc.shape[0]} calibration "
                      f"frames from {len(_parts)} part(s)", flush=True)
            self.membrane_module = MembraneHypergraphModule(
                out_dim=self.geo_embed.output_dim,
                members=md['members'], ksize=md['ksize'],
                n_particles=self.n_particles,
                hidden=config.get('membrane_hidden', 64),
                layers=config.get('membrane_layers', 2),
                latent_dim=self.latent_dim, box=self.box_size,
                delta_mode=_delta_mode,
                delta_scale_node=_ds_node if _ds_node else 1.0,
                delta_scale_hyper=_ds_hyper if _ds_hyper else 1.0,
                depth_mode=_depth_mode, **_ou_kw,
                standardize_inputs=_std_in,
                calib_coords=_cc, calib_node_depth=_cn, calib_hyper_depth=_ch,
            )
            # 'data_mean' starts the depth heads' biases at the pooled mean depth (node -2.43, hyper -3.65 in model
            # units) instead of 0. init_head has to emit that absolute level and z_from_depth takes the slab level from
            # it, but SophiaG moves a bias ~lr per optimizer step: ~1e-3 an epoch at 15 steps, so from 0 it never got
            # there (level ~60 too low). The OU attractor starts there too, so the untrained process doesn't relax across
            # the gap -- the mismatch the zero init above step_node_out exists to avoid.
            _bias_init = config.get('membrane_depth_bias_init', 'zero')
            if _bias_init not in ('zero', 'data_mean'):
                raise ValueError(f"membrane_depth_bias_init must be 'zero' or 'data_mean', got {_bias_init!r}")
            if _bias_init == 'data_mean':
                if not _ou:
                    raise ValueError(f"membrane_depth_bias_init='data_mean' needs membrane_depth_mode='ou', got "
                                     f"{_depth_mode!r}: elsewhere node_out is integrated or shared with step")
                _mn = float(_np.concatenate([x.ravel() for x in _nodes]).mean()) / _nstd
                _mh = float(_np.concatenate([x.ravel() for x in _hypers]).mean()) / _hstd
                _mm = self.membrane_module
                with torch.no_grad():
                    for _head, _v in ((_mm.node_out, _mn), (_mm.step_node_out, _mn),
                                      (_mm.hyper_out, _mh), (_mm.step_hyper_out, _mh)):
                        _head.bias.fill_(_v)
                print(f"  membrane depth heads start at the data mean: node {_mn:.4f} hyper {_mh:.4f}", flush=True)
            # Hyperedge sizes, for the tokenized decoder's hyperedge tokens: a
            # 2-protein group and a 7-protein group are different objects and the
            # token should say which. Buffer so it rides device moves and the
            # checkpoint, and is not a plain tensor attribute that silently stays
            # on CPU under .to(device).
            self.register_buffer('mem_ksize',
                                 torch.as_tensor(md['ksize'], dtype=torch.float32),
                                 persistent=False)
            if self.dec_tokenized:
                with torch.no_grad():
                    self.decoder.delta_scale_node.fill_(float(_ds_node or 1.0))
                    self.decoder.delta_scale_hyper.fill_(float(_ds_hyper or 1.0))
        elif self.membrane_mode:
            import numpy as _np
            self.membrane_encoder = MembranePatchEncoder(
                out_dim=self.geo_embed.output_dim,
                n_particles=self.n_particles,
                radius=config.get('membrane_radius', 400.0),
                spacing=config.get('membrane_spacing', 50.0),
                d_disc=config.get('membrane_disc_dim', 32),
            )
            grids = _np.load(resolve(config['membrane_mesh_path']))      # [N,2,NG,NG]
            self.register_buffer('mesh_grids',
                                 torch.tensor(grids, dtype=torch.float32),
                                 persistent=False)

        # Compile strategy: coarse-grained graphs for max Inductor fusion.
        #   - geo_embed is compiled in isolation (runs outside the per-step
        #     loop, processing the "next-step context" for the following AR
        #     iteration).
        #   - encoder + stochastic_head + decoder are compiled TOGETHER as
        #     the per-step function `_forward_step_multilag` (see below).
        #     This fuses the LSTM bank -> transition head (NSF) -> OU-flow
        #     integration -> per-particle Cholesky decoder path into a
        #     single compiled graph so Inductor can fuse kernels across
        #     module boundaries.
        # dynamic=True on the step function lets the encoder's int-valued
        # step_idx arg flow without triggering per-value recompiles
        # (recompile warnings are enabled in run_training to verify this
        # empirically -- recompile-per-step would show up there).
        if not config.get('disable_torch_compile', False):
            self.geo_embed = torch.compile(self.geo_embed)
            # Membrane mode keeps the step uncompiled: the PBC-bilinear mesh
            # gather + dynamic frame indexing don't fuse cleanly in Inductor.
            if not self.membrane_mode:
                self._forward_step_multilag = torch.compile(
                    self._forward_step_multilag, dynamic=True,
                )

    # ----- single AR step -----

    def _membrane_context(self, curr_x, start_idx, step_idx):
        """Sample + encode the membrane disc at the predicted (detached) position
        from the cyclically-indexed mesh frame -> decoder cross-attn context."""
        B = curr_x.size(0)
        N = self.mesh_grids.size(0)
        fidx = (start_idx + step_idx) % N                       # [B]
        mesh_t = self.mesh_grids[fidx]                          # [B,2,NG,NG]
        pos = curr_x.reshape(B, self.n_particles, 3)[:, :, :2].detach()
        return self.membrane_encoder(mesh_t, pos)               # [B, geo_out_dim]

    def _apply_squash(self, curr_x, ds, latent, step_idx=0):
        """Contract the closing radial component of `ds`. No-op unless enabled.

        Called from BOTH the training step and predict_future's inline rollout --
        those two duplicate the AR step, and a squash wired into only one of them
        would be silently inert exactly where it matters.
        """
        if self.squash_head is None:
            return ds
        core_pos = (self.ev_core * self.box_size).clamp_min(1e-6)
        if self.squash_trace is None:
            out, _ = self.squash_head(curr_x, ds, core_pos, latent)
            if self.squash_aux is not None:
                if not torch.isfinite(core_pos):
                    # Belt-and-braces: _init_config already demands an explicit
                    # excluded_volume_core when w_squash>0. An infinite core makes
                    # minimal_contraction_target 0 everywhere.
                    raise ValueError(
                        'squash supervision needs a finite core; ev_core is +inf. '
                        'Set excluded_volume_core in the config.')
                # Stash DETACHED INPUTS ONLY -- no graph, no head call. The aux loss
                # is built after the rollout, outside the checkpointed region.
                #
                # This step runs inside ckpt_util.checkpoint, so with
                # use_reentrant=False it is REPLAYED during backward. Anything
                # grad-carrying collected here would belong to a discarded graph, and
                # any branch whose condition changed between the two passes desyncs
                # the saved-tensor count outright ("A different number of tensors was
                # saved during the original forward and recomputation").
                # Keying by step makes the replay idempotent rather than duplicating.
                with torch.no_grad():
                    self.squash_aux[int(step_idx)] = (
                        curr_x.detach(), ds.detach(),
                        latent.detach() if latent is not None else None)
            return out
        # Frozen-backbone training. Two rules, both load-bearing:
        #  * the target depends on the RAW proposal, so take it before contracting;
        #  * keep each step's loss graph INDEPENDENT. The trajectory is rolled
        #    on-policy but under no_grad, otherwise x_{t+1} carries the head's graph
        #    into every later step and a long rollout accumulates one graph per step
        #    before the first backward.
        b, n = curr_x.size(0), self.n_particles
        cx, ds_in = curr_x.detach(), ds.detach()
        with torch.no_grad():
            tgt = minimal_contraction_target(
                cx, ds_in, core_pos, self.box_size, n,
                asymmetric=self.squash_head.asymmetric)
        s_pred = self.squash_head.squash(
            cx.view(b, n, 3)[..., :2], ds_in.view(b, n, 3)[..., :2], core_pos, latent)
        self.squash_trace.append((s_pred, tgt))
        with torch.no_grad():
            return self.squash_head(cx, ds_in, core_pos, latent)[0]

    def _prev_kw(self, px):
        # geo_ic_once: every frame after the IC is embedded from its step delta
        return {'prev': px.unsqueeze(1)} if self.geo_ic_once and px is not None else {}

    def _pos_feat_kw(self, cx, px):
        # geo_ic_once: the decoder's position keys get the embedding's own features,
        # so absolute position stops at the IC frame there as well
        if not self.geo_ic_once:
            return {}
        n = self.n_particles
        return {'pos_feat': ic_once_node_features(
            cx.view(-1, n, 3), None if px is None else px.view(-1, n, 3),
            self.box_size, self.geo_ic_axes == 'xy', self.geo_pair_dir_shells)}

    def _forward_step_multilag(self, curr_x, enc_states, z_prev, curr_emb, start_idx, step_idx,
                               ns1=None, ns2=None, active_mask=None, eps_prev=None, mem_ctx=None,
                               node_delta=None, hyper_delta=None, pos_prev=None):
        transition_params, enc_new = self.encoder(curr_emb, enc_states, step_idx)
        noise_state = (ns1, ns2) if ns1 is not None else None
        z_new, latent_entropy, latent_top1_prob, noise_state = self.stochastic_head.step(
            transition_params, z_prev, noise_state=noise_state)
        # Decoder cross-attn context: explicit hypergraph membrane ctx (when threaded
        # in), else the GT-mesh patch membrane, else geometry curr_emb.
        if mem_ctx is not None:
            dec_ctx = mem_ctx
        elif self.membrane_mode:
            dec_ctx = self._membrane_context(curr_x, start_idx, step_idx)
        else:
            dec_ctx = curr_emb
        if self.dec_tokenized:
            # The structural wiring: per-protein CHANGE tokens (latent xyz + the
            # membrane depth delta) attend over per-protein POSITION tokens and the
            # hyperedge depth deltas. curr_x is passed explicitly because in membrane
            # mode dec_ctx is the membrane context and curr_emb never reaches the
            # decoder at all -- the position would otherwise be invisible to it.
            ds, _, _, eps_t = self.decoder(
                z_new, dec_ctx, active_mask=active_mask, eps_prev=eps_prev,
                pos=curr_x, node_delta=node_delta, hyper_delta=hyper_delta,
                ksize=getattr(self, 'mem_ksize', None),
                **self._pos_feat_kw(curr_x, pos_prev))
        else:
            ds, _, _, eps_t = self.decoder(z_new, dec_ctx, active_mask=active_mask,
                                           eps_prev=eps_prev)
        ds = self._drop_z(ds)
        ds = self._apply_squash(curr_x, ds, z_new, step_idx)
        x_tent = curr_x + ds
        # What the NETWORK proposed, before the physics prior corrects it. The EV
        # penalty has to score this: scoring the corrected position is exposure bias
        # and leaves the penalty with no gradient while the force is strong (see
        # base_model.ev_loss_preforce). The trajectory still continues from x_pred.
        # Wrapped only for parity with x_pred -- the penalty's min-image distances are
        # wrap-invariant either way.
        x_pre = torch.remainder(x_tent, self.box_size) if self.ev_capture_preforce else None
        if self.ev_force_enabled:
            x_tent = x_tent + self.ev_force(x_tent)      # in-drift repulsion prior
        x_pred = torch.remainder(x_tent, self.box_size)
        if active_mask is not None:
            # Zero out inactive particle slots in x_pred
            slot_mask = active_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(
                -1, self.n_particles * 3).float()
            x_pred = x_pred * slot_mask
            if x_pre is not None:
                x_pre = x_pre * slot_mask
        # eps_t (decoder AR(1) Markov state; None when white) is always returned last,
        # then x_pre when ev_capture_preforce -- a trace-time constant, so the arity is
        # fixed per model and forward() unpacks off the same attribute.
        if noise_state is not None:
            out = (x_pred, enc_new, z_new, latent_entropy, latent_top1_prob,
                   noise_state[0], noise_state[1], eps_t)
        else:
            out = (x_pred, enc_new, z_new, latent_entropy, latent_top1_prob, eps_t)
        return (out + (x_pre,)) if self.ev_capture_preforce else out

    # ----- forward (training) -----

    def forward(self, x_sequence, target_sequence=None, active_mask=None,
                edge_src=None, edge_dst=None, edge_mask=None, start_frame_idx=None,
                theta_gt=None, blade=None, blade_gt=None):
        if target_sequence is None:
            target_sequence = x_sequence
        x_sequence, target_sequence = self._drop_z(x_sequence), self._drop_z(target_sequence)
        B, S, _ = target_sequence.size()
        # Reset per forward, never append across calls: a stale list would carry a
        # freed graph from the previous batch into this backward.
        self.squash_aux = ({} if (self.squash_head is not None
                                  and getattr(self, 'w_squash', 0.0) > 0
                                  and self.squash_trace is None) else None)
        if self.membrane_mode and start_frame_idx is None:
            start_frame_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        sidx = start_frame_idx
        if self.geom_spec['needs_blade'] and blade is None:
            blade = self.default_blade(B, dtype=x_sequence.dtype)

        enc_states = self.encoder.init_state(B, self.device)
        z = torch.zeros(B, self.latent_dim, device=self.device)
        noise_state = self.stochastic_head.init_noise_state(B, self.device)
        has_ns = noise_state is not None
        # Decoder AR(1) colored-noise state (Markov: only eps_{t-1}). Zeroed per
        # rollout so step 0 sees A.0 = 0 (== white). None when AR is off.
        use_eps = self.decoder.ar1 is not None
        eps_prev = torch.zeros(B, self.n_particles * 3, device=self.device) if use_eps else None

        # Pass mask args to geo_embed (GNN uses them; GAT ignores them)
        geo_kw = {}
        if active_mask is not None:
            geo_kw = dict(active_mask=active_mask, edge_src=edge_src,
                          edge_dst=edge_dst, edge_mask=edge_mask)

        ce, _ = self.geo_embed(x_sequence[:, 0:1, :], **geo_kw)
        cx = x_sequence[:, 0, :]
        px = None
        # geom_arm in {'moments','se3'} feeds a blade state that the 3-DOF rotation
        # head advances every step (rodrigues(blade_state, w) below, gated on
        # geom_spec['advances_blade']); IC is the GT blade for both.
        blade_state = blade
        if self.arms_enabled:
            theta = (theta_gt[:, 0] if theta_gt is not None
                     else torch.zeros(B, self.n_particles, device=self.device))
            if self.geom_rot3d:
                pf = arm_pair_feature_3d(cx, blade_state, self.box_size)
            else:
                pf = self.arm_pair_feature(cx, theta)
            ce = torch.cat([ce, pf.unsqueeze(1)], dim=-1)
            theta_ps = []
        if self.geom_moments:
            ce = torch.cat([ce, cov_from_blades(blade_state, self.arm_radius)
                            .reshape(ce.shape[0], 1, -1)], dim=-1)
        if self.geom_splat:
            ce = torch.cat([ce, self.splat_encoder(cx, blade_state).unsqueeze(1)], dim=-1)
        preds = []
        # Pre-force positions, stacked only when the EV penalty scores them. The bare
        # stack is [B,T,N*3] (7.7 MiB at B=32,T=3000) but the retained pairwise-distance
        # intermediates dominate: measured peak 7986.5 -> 8046.9 MiB, i.e. +60.4 MiB
        # (+0.76%) on the checkpointed evtf_cosine rollout.
        pre_preds = []
        capture_z = getattr(self, 'w_latent_corr', 0.0) > 0
        z_stack = []
        # Predicted blade sequence, for the blade-sequence losses (rot_step, tilt, and
        # the 3-DOF arms' theta_step). Gated on the weights (not just their
        # weighted-sum use downstream) -- otherwise every arms_enabled+advances_blade
        # run pays to build and retain a [B,T,P,3,3] tensor (B=32,T=3000,P=7 at
        # seq_len_end) that no loss reads. rotation_losses_enabled() is shared with
        # training_step's blade_gt threading so the two gates cannot drift apart.
        capture_blades = (self.arms_enabled and self.geom_spec['advances_blade']
                          and self.rotation_losses_enabled())
        blade_ps = []
        latent_entropy_sum, latent_top1_sum, latent_step_count = self._init_latent_diagnostics()
        # Hypergraph membrane: seed membrane_0 from the IC geometry; thread the
        # (node-depth, hyperedge-depth, context) state across the AR loop.
        hg = self.membrane_mode and self.membrane_hypergraph
        if hg:
            m_node, m_hyper, m_ctx = self.membrane_module.init_head(cx)
            zlev = self._z_level(cx, m_node) if self.z_from_depth else None
            m_node_init, m_hyper_init = m_node, m_hyper
            m_node_preds, m_hyper_preds = [], []
            # membrane_module.step runs AFTER the decoder in this loop, so the delta
            # available at decode time is depth_t - depth_{t-1}. Zero on the first
            # step, which is correct: nothing has moved yet.
            m_node_d = torch.zeros_like(m_node)
            m_hyper_d = torch.zeros_like(m_hyper)
        # Gradient checkpointing is REQUIRED for the long curriculum (seq->3000):
        # without it the full unrolled autograd graph OOMs. Both the per-step
        # function and the membrane propagation are checkpointed.
        do_ckpt = self.use_gradient_checkpointing and self.training

        for t in range(S - 1):
            step_args = (cx, enc_states, z, ce, sidx, t)
            # Positional slots after step_idx: ns1, ns2, active_mask, eps_prev.
            # Pad with None so each value lands in its declared slot.
            if has_ns:
                step_args = step_args + (noise_state[0], noise_state[1])
            elif active_mask is not None or use_eps:
                step_args = step_args + (None, None)
            if active_mask is not None or use_eps:
                step_args = step_args + (active_mask,)
            if use_eps:
                step_args = step_args + (eps_prev,)
            mem_ctx = m_ctx if hg else None
            tok = dict(node_delta=m_node_d, hyper_delta=m_hyper_d) if (
                hg and self.dec_tokenized) else {}
            if self.geo_ic_once and px is not None:
                tok['pos_prev'] = px
            if do_ckpt:
                result = ckpt_util.checkpoint(
                    self._forward_step_multilag,
                    *step_args, mem_ctx=mem_ctx, **tok,
                    use_reentrant=False)
            else:
                result = self._forward_step_multilag(*step_args, mem_ctx=mem_ctx, **tok)
            # x_pre is appended after eps_t when captured; peel it first so the
            # unpacking below is unchanged.
            if self.ev_capture_preforce:
                result, x_pre_t = result[:-1], result[-1]
            # eps_t is always the last returned element (None when AR is off).
            if has_ns:
                xp, enc_states, z, latent_entropy, latent_top1, ns1, ns2, eps_t = result
                noise_state = (ns1, ns2)
            else:
                xp, enc_states, z, latent_entropy, latent_top1, eps_t = result
            if use_eps:
                eps_prev = eps_t
            latent_entropy_sum, latent_top1_sum, latent_step_count = self._update_latent_diagnostics(
                latent_entropy_sum, latent_top1_sum, latent_step_count, latent_entropy, latent_top1)

            preds.append(xp)
            if self.ev_capture_preforce:
                pre_preds.append(x_pre_t)
            if capture_z:
                z_stack.append(z)
            if hg:
                # propagate membrane_t -> membrane_{t+1} using the just-computed
                # movement z and the CURRENT geometry cx ("prev membrane + motion").
                # Checkpointed too -> peak membrane activations stay O(1 step).
                m_node_prev, m_hyper_prev = m_node, m_hyper
                if do_ckpt:
                    m_node, m_hyper, m_ctx = ckpt_util.checkpoint(
                        self.membrane_module.step, m_node, m_hyper, z, cx,
                        use_reentrant=False)
                else:
                    m_node, m_hyper, m_ctx = self.membrane_module.step(m_node, m_hyper, z, cx)
                if self.dec_tokenized:
                    m_node_d = m_node - m_node_prev
                    m_hyper_d = m_hyper - m_hyper_prev
                m_node_preds.append(m_node)
                m_hyper_preds.append(m_hyper)
                if self.z_from_depth:
                    xp = self._set_z(xp, m_node, zlev, wrap=True)
                    preds[-1] = xp
            if self.arms_enabled:
                if self.geom_spec['advances_blade']:
                    blade_state = rodrigues(blade_state, self.rotation_head(ce).unsqueeze(2))
                    if capture_blades:
                        blade_ps.append(blade_state)
                else:
                    theta = theta + self.rotation_head(ce)   # ce = this step's arm-augmented emb
                    theta_ps.append(theta)
            px, cx = cx, xp
            ce, _ = self.geo_embed(xp.unsqueeze(1), **geo_kw, **self._prev_kw(px))
            if self.arms_enabled:
                if self.geom_rot3d:
                    pf = arm_pair_feature_3d(xp, blade_state, self.box_size)
                else:
                    pf = self.arm_pair_feature(xp, theta)
                ce = torch.cat([ce, pf.unsqueeze(1)], dim=-1)
            if self.geom_moments:
                ce = torch.cat([ce, cov_from_blades(blade_state, self.arm_radius)
                                .reshape(ce.shape[0], 1, -1)], dim=-1)
            if self.geom_splat:
                ce = torch.cat([ce, self.splat_encoder(xp, blade_state).unsqueeze(1)],
                               dim=-1)

        # --- Loss computation ---
        xps = torch.stack(preds, dim=1)
        del preds
        A = xps.size(1)
        tp = target_sequence[:, 1:A + 1]

        # Pairwise distances — compute once, reuse for pairwise_loss + cohesion_loss
        pred_pw = self.geo_embed.compute_distances_xy_only(xps)
        tgt_pw = self.geo_embed.compute_distances_xy_only(tp)
        qfn = self._get_quantile_fn()
        # For multi-system: build pair-level mask from particle mask
        pair_mask = self._particle_mask_to_pair_mask(active_mask) if active_mask is not None else None
        if pair_mask is not None:
            pl = self.compute_quantile_loss_masked(pred_pw, tgt_pw, pair_mask)
        else:
            pl = qfn(pred_pw, tgt_pw)
        cl = self.compute_cohesion_loss(xps, tp, pred_pw=pred_pw, tgt_pw=tgt_pw,
                                        active_mask=pair_mask)
        ph = self.compute_pairwise_tail_loss(pred_pw, tgt_pw) if self.w_pairwise_tail > 0 else None
        if pair_mask is not None and max(self.w_logpair_kl, self.w_dlogpair_kl, self.w_logpair_q, self.w_dlogpair_q) > 0:
            raise ValueError('the log pair-distance losses pool every pair and do not take a multi-system particle mask')
        # the pair distances are fractions of the box, so the overlap floor of one model unit is 1 / box
        lk = logpair_kl(pred_pw, tgt_pw, floor=1.0 / float(self.box_size)) if self.w_logpair_kl > 0 else None
        dk = dlogpair_kl(pred_pw, tgt_pw, floor=1.0 / float(self.box_size)) if self.w_dlogpair_kl > 0 else None
        lq = self.compute_logpair_q(pred_pw, tgt_pw, floor=1.0 / float(self.box_size)) if self.w_logpair_q > 0 else None
        dq = self.compute_dlogpair_q(pred_pw, tgt_pw, floor=1.0 / float(self.box_size)) if self.w_dlogpair_q > 0 else None
        ev = None
        if self.w_excluded_volume > 0:
            # pred_pw is the POST-force distance -- the force has already removed the
            # overlap there, so the hinge on it is inert. ev_loss_preforce scores the
            # network's raw proposal instead. tgt_pw is still what advances the
            # running-min core, identically either way.
            if self.ev_capture_preforce:
                ev_pw = self.geo_embed.compute_distances_xy_only(
                    torch.stack(pre_preds, dim=1))
            else:
                ev_pw = pred_pw
            ev = self.compute_excluded_volume_loss(ev_pw, tgt_pw)
            del ev_pw
        del pred_pw, tgt_pw, pre_preds

        # z-coordinate quantile loss (wrapped, no unwrap needed)
        pz = tz = None
        if self.predict_z:
            pz = torch.remainder(
                xps[:, :, :self.n_coords].reshape(B, -1, self.n_particles, 3)[:, :, :, 2],
                self.box_size) / self.box_size
            tz = torch.remainder(
                tp[:, :, :self.n_coords].reshape(B, -1, self.n_particles, 3)[:, :, :, 2],
                self.box_size) / self.box_size

        # Unwrap for displacement, angle, VACF
        puw, tuw = self._unwrap_xy(xps, active_mask), self._unwrap_xy(tp, active_mask)
        dl = self.compute_displacement_quantile_loss(
            xps, tp, pred_xy_unwrapped=puw, target_xy_unwrapped=tuw,
            active_mask=active_mask)
        al = self.compute_angle_quantile_loss(
            xps, tp, pred_xy_unwrapped=puw, target_xy_unwrapped=tuw,
            active_mask=active_mask) if self.w_angle > 0 else None
        vf = self.compute_vacf_quantile_loss(
            xps, tp, pred_xy_unwrapped=puw, target_xy_unwrapped=tuw)
        # Lag-wise mean-VACF MSE (only computed if requested via w_vacf_mse > 0;
        # otherwise we skip the work and the training loop won't add it to total).
        if getattr(self, 'w_vacf_mse', 0.0) > 0:
            vf_mse = self.compute_vacf_mse_loss(
                xps, tp, pred_xy_unwrapped=puw, target_xy_unwrapped=tuw)
        else:
            vf_mse = None
        if getattr(self, 'w_alpha', 0.0) > 0:
            alpha_l = self.compute_alpha_loss(puw, tuw)
        else:
            alpha_l = None
        if getattr(self, 'w_vacf_spatial', 0.0) > 0:
            vacf_sp = self.compute_vacf_spatial_l1_loss(puw, tuw)
        else:
            vacf_sp = None
        if getattr(self, 'w_kurtosis', 0.0) > 0:
            kurt_l = self.compute_kurtosis_loss(puw, tuw)
        else:
            kurt_l = None
        del puw, tuw

        # Latent-velocity autocorrelation loss (off by default).
        if capture_z and z_stack:
            from membrane.vacf_loss import compute_latent_corr_loss as _lat_corr
            z_seq = torch.stack(z_stack, dim=1)  # [B, T-1, latent_dim]
            latent_corr_loss = _lat_corr(
                z_seq,
                target_curve=getattr(self, 'latent_corr_target', None),
                lags=getattr(self, 'latent_corr_lags', None),
                kernel=getattr(self, 'latent_corr_kernel', 'tanh'),
                scale=getattr(self, 'latent_corr_scale', 0.05),
            )
        else:
            latent_corr_loss = None
        del z_stack

        pred_blades = torch.stack(blade_ps, dim=1) if capture_blades and blade_ps else None
        del blade_ps

        latent_entropy, latent_top1_prob = self._finalize_latent_diagnostics(
            latent_entropy_sum, latent_top1_sum, latent_step_count)

        out = {
            'displacement_loss': dl,
            'angle_loss': al,
            'pairwise_loss': pl,
        }
        if self.predict_z:
            out['z_quantile_loss'] = self.compute_z_quantile_loss(pz, tz, active_mask)
            out['z_displacement_loss'] = self.compute_z_displacement_quantile_loss(
                xps, tp, active_mask=active_mask)
        out.update({
            'vacf_loss': vf,
            'cohesion_loss': cl,
            'latent_entropy': latent_entropy,
            'latent_top1_prob': latent_top1_prob,
        })
        if al is None:
            del out['angle_loss']
        if lk is not None:
            out['logpair_kl_loss'], out['logpair_kl_mean'], out['logpair_kl_spread'] = lk
        if dk is not None:
            out['dlogpair_kl_loss'], out['dlogpair_kl_mean'], out['dlogpair_kl_spread'] = dk
        if lq is not None:
            out['logpair_q_loss'], out['logpair_q_offset'] = lq
        if dq is not None:
            out['dlogpair_q_loss'], out['dlogpair_q_offset'] = dq
        if self.squash_aux:
            # ONE batched head call over every collected step, here rather than inside
            # the rollout: the per-step call sits in a checkpointed region and would be
            # replayed during backward.
            #
            # Both sides are computed from DETACHED inputs, so w_squash trains the HEAD
            # ONLY. Detaching just the target would not be enough -- s_pred is a
            # function of ds, so an attached one would backprop this term into the
            # backbone and reward it for emitting proposals whose required contraction
            # happens to match what the head already predicts, which has nothing to do
            # with excluded volume. The backbone still couples to the head, but through
            # the intended route: the applied squash carries grad, so every main loss
            # sees the POST-squash step.
            #
            # The target is 1.0 wherever the raw proposal was already valid -- most
            # steps -- so this teaches "stay near 1, contract only when you must"
            # rather than a blanket slowdown.
            steps = [self.squash_aux[k] for k in sorted(self.squash_aux)]
            cx = torch.cat([a[0] for a in steps], dim=0)
            dsx = torch.cat([a[1] for a in steps], dim=0)
            lat = (torch.cat([a[2] for a in steps], dim=0)
                   if steps[0][2] is not None else None)
            nb, npart = cx.size(0), self.n_particles
            core_pos = (self.ev_core * self.box_size).clamp_min(1e-6)
            with torch.no_grad():
                tg = minimal_contraction_target(
                    cx, dsx, core_pos, self.box_size, npart,
                    asymmetric=self.squash_head.asymmetric)
            sp = self.squash_head.squash(cx.view(nb, npart, 3)[..., :2],
                                         dsx.view(nb, npart, 3)[..., :2],
                                         core_pos, lat)
            out['squash_loss'] = ((sp - tg) ** 2).mean()
            self.squash_aux = None
        if ph is not None:
            out['pairwise_tail_loss'] = ph
        if ev is not None:
            out['excluded_volume_loss'] = ev
        if vf_mse is not None:
            out['vacf_mse_loss'] = vf_mse
        if latent_corr_loss is not None:
            out['latent_corr_loss'] = latent_corr_loss
        if alpha_l is not None:
            out['alpha_loss'] = alpha_l
        if vacf_sp is not None:
            out['vacf_spatial_loss'] = vacf_sp
        if kurt_l is not None:
            out['kurtosis_loss'] = kurt_l
        if pred_blades is not None:
            out['pred_blades'] = pred_blades   # [B,A,P,3,3], for compute_rot_step_loss/compute_tilt_loss
            # blade_gt is [B,S,P,3,3] (S = x_sequence's frame count, same window slice
            # as x_sequence -- see data/membrane_dataset.py); frame 0 seeded blade_state
            # (the IC handed in as `blade`), so pred_blades[:,k] is the state at time
            # k+1 and its GT counterpart is blade_gt[:,k+1]. A = S-1 by construction
            # (one AR step per remaining frame), so this slice always has exactly A
            # frames too -- but check rather than assume, since a silent length
            # mismatch would make the sorted-quantile match interpolate instead of
            # comparing the actual observed distributions.
            if blade_gt is not None:
                gt_blades = blade_gt[:, 1:A + 1]
                if gt_blades.shape[1] != pred_blades.shape[1]:
                    raise ValueError(
                        f"gt_blades has {gt_blades.shape[1]} frames but pred_blades has "
                        f"{pred_blades.shape[1]} -- blade_gt must be sliced identically "
                        f"to x_sequence/target_sequence.")
                if self.w_rot_step > 0:
                    out['rot_step_loss'] = self.compute_rot_step_loss(pred_blades, gt_blades)
                if self.w_tilt > 0:
                    out['tilt_loss'] = self.compute_tilt_loss(pred_blades, gt_blades)
                # In-plane spin. The 2D theta block below is skipped for rot3d arms
                # (no theta_ps to stack), which left spin -- the one DOF neither
                # tilt_angle nor rotation_step_angle can see, both reading only the
                # trimer normal -- entirely unsupervised on exactly the arms that
                # predict a full 3D orientation. blade_theta recovers it from the
                # blade state via the dataset's own 3-fold order parameter, so the
                # same w_theta_step means the same thing on both arm families.
                if self.w_theta_step > 0:
                    out['theta_step_loss'] = self.compute_blade_theta_step_loss(
                        pred_blades, gt_blades)
        if hg:
            # predicted membrane states for the aux loss (training_step compares
            # to GT depth windows); [B,A,7] and [B,A,120], plus the init prediction.
            out['mem_node_pred'] = torch.stack(m_node_preds, dim=1)   # [B,A,7]
            out['mem_hyper_pred'] = torch.stack(m_hyper_preds, dim=1)  # [B,A,120]
            out['mem_node_init'] = m_node_init
            out['mem_hyper_init'] = m_hyper_init
            if getattr(self, 'w_mem_coupling', 0.0) > 0:
                # The coupling loss needs positions and depths in the same scope.
                # Depth GT only reaches training_step, so the positions go the other
                # way; both are already the same A frames, so neither side re-slices.
                # Popped there before log_dict, like pred_blades.
                out['pred_positions'] = xps
                out['target_positions'] = tp
            # z-consistency: protein z pulled to the node depth, per frame (base_model._mem_consistency_loss)
            if getattr(self, 'w_mem_consistency', 0.0) > 0 and getattr(self, 'mem_consistency_slope', 0.0):
                out['mem_consistency'] = self._mem_consistency_loss(out['mem_node_pred'], pz)
        if self.arms_enabled and not self.geom_rot3d:
            # theta/2D-arm-template losses only apply to the 1-DOF theta path (arm A
            # /moments/none); se3's rotation head advances blade_state directly (no
            # theta_ps to stack -- see the AR loop above) and its 3D arm geometry
            # already reaches the model as an INPUT feature (arm_pair_feature_3d),
            # not through these 2D extended-point-set losses.
            theta_ps_t = torch.stack(theta_ps, dim=1)                    # [B,A,N]
            theta0 = theta_gt[:, :1] if theta_gt is not None else theta_ps_t[:, :1].detach()
            theta_full = torch.cat([theta0, theta_ps_t], dim=1)          # [B,A+1,N] (theta_0..theta_A)
            theta_tgt = theta_gt[:, :A + 1] if theta_gt is not None else theta_full.detach()
            pred_pts = self.arm_points(xps, theta_ps_t)                  # [B,A,4N,2]
            gt_pts = self.arm_points(tp, theta_tgt[:, 1:A + 1])
            if self.w_arm_pairwise > 0:
                out['arm_pairwise_loss'] = self.compute_arm_pairwise_loss(pred_pts, gt_pts)
            if self.w_arm_excluded_volume > 0:
                out['arm_excluded_volume_loss'] = self.compute_arm_excluded_volume_loss(pred_pts, gt_pts)
            if self.w_theta_step > 0 and theta_gt is not None:
                out['theta_step_loss'] = self.compute_theta_step_loss(theta_full, theta_tgt)
        return out

    # ----- inference -----

    def _z_level(self, x, node):
        """slab level of a frame, [B, 1]: mean over proteins of min-image z minus raw node depth"""
        z = x.reshape(x.shape[0], self.n_particles, 3)[..., 2]
        z = z - self.box_size * torch.round((z - z[..., :1]) / self.box_size)
        return (z - node.detach() * self.depth_raw).mean(-1, keepdim=True)

    def _set_z(self, x, node, level, wrap):
        """x [B, P*3] with each protein's z replaced by level + its raw node depth. The depth is detached: losses on
        positions don't reach the depth module through z, so only the depth losses train it"""
        xyz = x.reshape(x.shape[0], self.n_particles, 3)
        z = level + node.detach() * self.depth_raw
        if wrap:
            z = torch.remainder(z, self.box_size)
        return torch.cat([xyz[..., :2], z.unsqueeze(-1).to(xyz.dtype)], -1).reshape(x.shape)

    def predict_future(self, initial_x, steps, return_unwrapped=False,
                       active_mask=None, edge_src=None, edge_dst=None, edge_mask=None,
                       start_frame_idx=None, return_membrane=False, initial_theta=None,
                       blade=None):
        B = initial_x.size(0)
        if self.membrane_mode and start_frame_idx is None:
            start_frame_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        sidx = start_frame_idx
        # No GT blades at inference time (LongHorizonStabilityCallback and the eval
        # scripts only have coordinates): fall back to default_blade, which builds
        # whatever self.blade_ic names -- 'sampled' (the default) draws per-protein
        # orientations from the measured GT tilt distribution, 'flat' recovers the old
        # 3-fold template. Seeded from blade_ic_seed, so this IC does not add
        # unreproducible noise to the objective the callback monitors.
        if self.geom_spec['needs_blade'] and blade is None:
            blade = self.default_blade(B, dtype=initial_x.dtype)
        enc_states = self.encoder.init_state(B, self.device)
        z = torch.zeros(B, self.latent_dim, device=self.device)
        # Decoder AR(1) state — threaded internally so eval matches the training
        # rollout (the 40k-step LHSC rollout that feeds the z-VACF metric runs here).
        use_eps = self.decoder.ar1 is not None
        eps_prev = torch.zeros(B, self.n_particles * 3, device=self.device) if use_eps else None

        geo_kw = {}
        if active_mask is not None:
            geo_kw = dict(active_mask=active_mask, edge_src=edge_src,
                          edge_dst=edge_dst, edge_mask=edge_mask)

        initial_x = self._drop_z(initial_x)
        cx = initial_x[:, 0, :]
        cu = cx.clone()
        px = None
        if self.arms_enabled:
            theta = (initial_theta if initial_theta is not None
                     else torch.zeros(B, self.n_particles, device=self.device))
        # geom_arm in {'moments','se3'}: blade_state IC is the GT blade (or the
        # default_blade draw above when none is supplied), then the 3-DOF rotation
        # head advances it every step (see the matching comment in forward()).
        blade_state = blade
        ps, pu = [], []
        mns, mhs = [], []   # per-step membrane depths (captured iff return_membrane)
        # Hypergraph membrane: seed membrane_0 from the IC geometry and propagate
        # it forward (no GT mesh needed -> rollout works for any IC / length).
        hg = self.membrane_mode and self.membrane_hypergraph
        if hg:
            m_node, m_hyper, m_ctx = self.membrane_module.init_head(cx)
            zlev = self._z_level(cx, m_node) if self.z_from_depth else None
            # Zero on step 0: nothing has moved yet. Mirrors the train path.
            m_node_d = torch.zeros_like(m_node)
            m_hyper_d = torch.zeros_like(m_hyper)

        for t in range(steps):
            ce, _ = self.geo_embed(cx.unsqueeze(1), **geo_kw, **self._prev_kw(px))
            if self.arms_enabled:
                if self.geom_rot3d:
                    pf = arm_pair_feature_3d(cx, blade_state, self.box_size)
                else:
                    pf = self.arm_pair_feature(cx, theta)
                ce = torch.cat([ce, pf.unsqueeze(1)], dim=-1)
            if self.geom_moments:
                ce = torch.cat([ce, cov_from_blades(blade_state, self.arm_radius)
                                .reshape(ce.shape[0], 1, -1)], dim=-1)
            if self.geom_splat:
                ce = torch.cat([ce, self.splat_encoder(cx, blade_state).unsqueeze(1)],
                               dim=-1)
            transition_params, enc_states = self.encoder(ce, enc_states, t)
            z, _, _, _ = self.stochastic_head.step(transition_params, z)
            if hg:
                dec_ctx = m_ctx
            elif self.membrane_mode:
                dec_ctx = self._membrane_context(cx, sidx, t)
            else:
                dec_ctx = ce
            if self.dec_tokenized:
                # MUST mirror _forward_step_multilag. Left as the two-argument call,
                # this path silently degenerates to a single context key with the
                # delta feature pinned to zero -- and every model-selection decision
                # (ModelCheckpoint, NaNDetection, Optuna, the stability callback) is
                # made through predict_future, so the selected model would not be the
                # trained one. Measured divergence when tokens are dropped: mean
                # |d mu| 0.153 against |mu| 0.144, i.e. unrelated output.
                ds, _, _, eps_t = self.decoder(
                    z, dec_ctx, active_mask=active_mask, eps_prev=eps_prev,
                    pos=cx, node_delta=m_node_d if hg else None,
                    hyper_delta=m_hyper_d if hg else None,
                    ksize=getattr(self, 'mem_ksize', None),
                    **self._pos_feat_kw(cx, px))
            else:
                ds, _, _, eps_t = self.decoder(z, dec_ctx, active_mask=active_mask,
                                               eps_prev=eps_prev)
            ds = self._drop_z(ds)
            if self.arms_enabled:
                if self.geom_spec['advances_blade']:
                    blade_state = rodrigues(blade_state, self.rotation_head(ce).unsqueeze(2))
                else:
                    theta = theta + self.rotation_head(ce)
            if use_eps:
                eps_prev = eps_t
            if hg:
                m_node_prev, m_hyper_prev = m_node, m_hyper
                m_node, m_hyper, m_ctx = self.membrane_module.step(m_node, m_hyper, z, cx)
                if self.dec_tokenized:
                    m_node_d = m_node - m_node_prev
                    m_hyper_d = m_hyper - m_hyper_prev
                if return_membrane:
                    mns.append(m_node)
                    mhs.append(m_hyper)

            ds = self._apply_squash(cx, ds, z)
            nu = cu + ds
            # No pre-force capture here on purpose: nothing computes a loss on this
            # path, and the violation metrics must score the positions the model
            # actually DEPLOYS, i.e. post-force.
            if self.ev_force_enabled:
                nu = nu + self.ev_force(nu)              # in-drift repulsion prior
            if self.z_from_depth:
                nu = self._set_z(nu, m_node, zlev, wrap=False)
            nw = torch.remainder(nu, self.box_size)
            if active_mask is not None:
                slot_mask = active_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(
                    -1, self.n_particles * 3).float()
                nw = nw * slot_mask
                nu = nu * slot_mask
            ps.append(nw)
            pu.append(nu)
            px, cx, cu = cx, nw, nu

        sw = torch.stack(ps, dim=1)
        if return_membrane:
            # coords + concurrently-sampled membrane depths (aligned per step): the
            # membrane at step t is the state that produced coords[t] (both from the
            # same stochastic rollout, so they line up).
            mem = (torch.stack(mns, dim=1), torch.stack(mhs, dim=1))  # [B,steps,7],[B,steps,120]
            return (sw, torch.stack(pu, dim=1), *mem) if return_unwrapped else (sw, *mem)
        return (sw, torch.stack(pu, dim=1)) if return_unwrapped else sw
