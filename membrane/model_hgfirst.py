"""Membrane-first hypergraph model.

Design: docs/membrane/specs/2026-08-07_membrane_first_hypergraph_design.md.

The hypergraph is the WHOLE geometry encoder and the WHOLE decoder -- there is no
GAT/GNN `geo_embed` and no membrane module bolted on downstream. Every AR step:

    coords + blade + (node_depth, hyper_depth)
      -> HyperGeoEncoder            AllSet, every message pass in the readout
      -> Linear -> d_enc            (LayerNorm lives in the bank's ln_embed)
      -> multi-lag LSTM bank        same firing rule as model_multilag
      -> mu head -> z, 3*(P+E)      signed xyz per node AND per hyperedge
      -> HyperDecoder               diagonal Gaussian over DELTAS
      -> protein xyz  = remainder(x + node_delta, box)
         node depth   = depth + node_depth_delta      (its OWN head output)
         hyper depth  = depth + edge_delta_z

Differences from model_multilag that are deliberate, not oversights:

* No `geo_embed`. Pairwise XY distances for the loss battery come from
  `_pairwise_xy` here (same min-image formula, same triu order) so no loss helper
  has to change.
* The membrane is UPSTREAM: its depths are encoder inputs at step t, not a context
  vector produced after the motion was already decided. That is the whole point of
  the architecture.
* z is 3*(P+E) = 381 rather than 21, so the 120 hyperedge depths are no longer
  driven through a 21-dim bottleneck sized for proteins.
* The node DEPTH step is its own head output, not a scaled copy of the protein's z
  step. An earlier version tied them through one learnable gain, on the design's
  original "7x3 + 120x1 = 141 free numbers". That was wrong: node_depth is the
  membrane depth AT the protein's location, coords[...,2] is the protein's own z,
  and they are different quantities in different units (coords box-scaled, depth
  re-normalized to unit variance by the DataModule). Measured over the 3-part sweep
  pool, 18682 frames: corr(d_node_depth, d_protein_z) = 0.621, so a single gain
  matches the depth-step VARIANCE that the distributional losses score while
  discarding the 61.5% of depth-step variance protein z does not explain -- the same
  level-vs-delta trap as design finding 1 on hyperedges. The decoder now emits 4
  means + 4 log-sigmas per node, making the output 7x4 + 120x1 = 148 free numbers.
  Hyperedge depth stays fully free, as finding 1 requires.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt_util

from data.paths import resolve_all
from membrane.base_model import BaseBrownianModel, RotationHead
from membrane.geometry_features import rodrigues
from membrane.hypergraph_geometry import HyperDecoder, HyperGeoEncoder
from membrane.model_multilag import MultiLagLSTMEncoder


def membrane_paths(config):
    """The membrane npz pool, plural key first (pooled runs drop the singular one)."""
    paths = config.get('membrane_data_paths') or None
    if not paths:
        p = config.get('membrane_data_path')
        if not p:
            raise KeyError("hgfirst needs membrane_data_path(s): the hypergraph "
                           "incidence table and the depth targets both come from the "
                           "membrane npz, so there is no default structure to fall "
                           "back on.")
        paths = [p]
    return resolve_all(list(paths))


class BrownianModel(BaseBrownianModel):
    """HyperGeoEncoder -> multi-lag LSTM bank -> mu head -> HyperDecoder."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        if self.z_from_depth or not self.predict_z:
            raise ValueError('z_from_depth and predict_z=False are implemented in model_multilag only')
        for k in ('w_logpair_kl', 'w_dlogpair_kl', 'w_logpair_q', 'w_dlogpair_q'):
            if getattr(self, k) > 0:
                raise ValueError(f'{k} is implemented in model_multilag only')

        # There is no non-membrane mode: the model's inputs and outputs are the
        # membrane state. training_step's batch unpacking and the depth aux-loss
        # block both key off these two flags.
        self.membrane_mode = True
        self.membrane_hypergraph = True
        if config.get('geo_ic_once', False):
            raise ValueError("geo_ic_once does nothing here: this model has no geo_embed "
                             "and no decoder position keys")

        paths = membrane_paths(config)
        md = np.load(paths[0], allow_pickle=True)
        members, ksize = md['members'], md['ksize']
        P = self.n_particles
        if int(members.shape[1]) != P:
            raise ValueError(f"members table is [{members.shape[0]},{members.shape[1]}] "
                             f"but n_particles={P}")
        self.n_hyperedges = E = int(members.shape[0])

        # latent_dim is not a free hyperparameter here: the decoder reads a signed
        # xyz per node AND per hyperedge out of z, so a mismatch would reshape into
        # the wrong tensor (or silently drop edges) rather than fail.
        need = 3 * (P + E)
        if int(self.latent_dim) != need:
            raise ValueError(
                f"latent_dim must be 3*(P+E) = 3*({P}+{E}) = {need} for hgfirst "
                f"(signed xyz per protein node and per hyperedge), got {self.latent_dim}.")

        # The encoder consumes the 9 signed blade offsets as node features. Without
        # geom_blade_enabled the DataModule never returns `blade`, so every batch
        # would train on default_blade -- a baseline wearing the model's name, the
        # same silent failure the geom_arm guards in base_model exist to stop.
        if not self.geom_blade_enabled:
            raise ValueError("hgfirst requires geom_blade_enabled=True (the "
                             "hypergraph node features are the signed blade offsets).")
        # 1-DOF arms hold the blade at its IC for the whole rollout and score the 2D
        # extended-point-set losses, neither of which fits a model whose node feature
        # IS the 3D blade. Reject rather than silently freeze the orientation.
        if self.arms_enabled and not self.geom_spec['advances_blade']:
            raise ValueError(
                f"hgfirst with arms_enabled=True needs a geom_arm that advances a "
                f"blade state (3-DOF); geom_arm={self.geom_arm!r} does not.")
        # The colored-noise filter carries a 2-tensor Markov state through the AR
        # loop that this step function does not thread; dropping it silently would
        # turn a configured AR(2) latent noise into white noise.
        if getattr(self.stochastic_head, 'noise_filter', None) is not None:
            raise NotImplementedError(
                "hgfirst does not thread the latent colored-noise state; set "
                "colored_noise=None (the decoder's diagonal Gaussian is the noise "
                "model in this design).")

        hg_hidden = int(config.get('hg_hidden', 128))
        hg_layers = int(config.get('hg_layers', 3))
        hg_dec_hidden = int(config.get('hg_dec_hidden', 128))
        hg_dec_layers = int(config.get('hg_dec_layers', 2))
        d_enc = int(config.get('d_enc', 512))
        self.d_enc = d_enc

        self.hg_encoder = HyperGeoEncoder(
            hg_hidden, hg_layers, members, ksize,
            box=self.box_size, arm_radius=self.arm_radius)
        # Design Decision 3: concatenate every message pass, project ONCE, feed the
        # same vector to every lag. Pairing layer i with lag i would force
        # hg_layers == n_lag_lstms and tie message-passing depth to timescale.
        self.proj = nn.Linear(self.hg_encoder.readout_dim, d_enc)
        # No LayerNorm here: MultiLagLSTMEncoder.ln_embed already normalizes its
        # input, and stacking two back to back only wastes parameters.
        self.encoder = MultiLagLSTMEncoder(
            input_dim=d_enc,
            hidden_dim=int(config.get('lag_lstm_width', 128)),
            latent_dim=self.latent_dim,
            noise_rank=self.noise_rank,
            n_lag_lstms=int(config.get('n_lag_lstms', 7)),
            dropout=config.get('dropout', 0.1),
            transition_head_type=config.get('transition_head_type', 'mu_only'),
            config=config,
        )
        self.decoder = HyperDecoder(hg_dec_hidden, hg_dec_layers, members,
                                    enc_hidden=hg_hidden, box=self.box_size)

        # Membrane IC. The rollout needs depths at t=0 to feed the encoder, and
        # nothing supplies them at inference (the stability callback and the eval
        # scripts have coordinates only). Read them off an encoder pass with the
        # depth channels zeroed; training_step supervises the result against GT
        # frame 0 via mem_node_init / mem_hyper_init.
        self.init_node_out = nn.Linear(hg_hidden, 1)
        self.init_edge_out = nn.Linear(hg_hidden, 1)

        if self.arms_enabled:
            self.rotation_head = RotationHead(
                d_enc, P, sigma_init=self.arm_sigma_theta, dof=self.geom_spec['dof'])

        r, c = torch.triu_indices(P, P, offset=1)
        self.register_buffer('triu_rows', r, persistent=False)
        self.register_buffer('triu_cols', c, persistent=False)

        # Not compiled. The membrane branch of model_multilag already keeps its step
        # eager (the dynamic hypergraph gathers do not fuse cleanly in Inductor), and
        # the design's timings that sized medium/large were measured eager.

    # ----- geometry helper (replaces geo_embed.compute_distances_xy_only) -----

    def _pairwise_xy(self, x):
        """[B,T,P*3] -> [B,T,n_pairs] min-image XY distances / box, triu order.

        Same formula and same pair ordering as GAT_GeometryEmbedding, so every
        pairwise/cohesion/excluded-volume helper on the base class is unchanged.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, S, _ = x.shape
        xy = x[:, :, :self.n_coords].reshape(B, S, self.n_particles, 3)[:, :, :, :2]
        d = xy.unsqueeze(3) - xy.unsqueeze(2)
        d = d - self.box_size * torch.round(d / self.box_size)
        dm = torch.sqrt((d ** 2).sum(-1) + 1e-8)
        return dm[:, :, self.triu_rows, self.triu_cols] / self.box_size

    # ----- membrane IC -----

    def init_membrane(self, coords, blade):
        """-> (node_depth [B,P], hyper_depth [B,E]) for step 0."""
        B = coords.shape[0]
        zn = coords.new_zeros(B, self.n_particles)
        ze = coords.new_zeros(B, self.n_hyperedges)
        _, (node_h, edge_h) = self.hg_encoder(coords, blade, zn, ze)
        return (self.init_node_out(node_h).squeeze(-1),
                self.init_edge_out(edge_h).squeeze(-1))

    # ----- single AR step -----

    def _forward_step_hg(self, cx, blade_state, m_node, m_hyper, enc_states, z, step_idx):
        B = cx.shape[0]
        readout, (node_h, edge_h) = self.hg_encoder(cx, blade_state, m_node, m_hyper)
        emb = self.proj(readout).unsqueeze(1)                       # [B,1,d_enc]
        params, enc_new = self.encoder(emb, enc_states, step_idx)
        z_new, latent_entropy, latent_top1_prob, _ = self.stochastic_head.step(params, z)
        nz = 3 * self.n_particles
        z_node = z_new[:, :nz].reshape(B, self.n_particles, 3)
        z_edge = z_new[:, nz:].reshape(B, self.n_hyperedges, 3)
        node_delta, edge_delta, node_depth_delta = self.decoder(
            z_node, z_edge, node_h, edge_h, cx)

        x_tent = cx + node_delta.reshape(B, -1).to(cx.dtype)
        if self.ev_force_enabled:
            x_tent = x_tent + self.ev_force(x_tent)                 # in-drift repulsion
        x_pred = torch.remainder(x_tent, self.box_size)
        # The realized step, ev_force included -- returned so a rollout can keep an
        # unwrapped track without having to invert the wrap.
        step_delta = x_tent - cx
        # Depth integration. edge_delta[...,:2] is the recomputed circular-mean
        # displacement and is deliberately unused: the edge POSITION is a closed-form
        # function of the node positions (design Decision 1, revised), so carrying an
        # edge xy state would only give it something to drift away from.
        # Node depth is its OWN head output, not a scaled copy of the protein's z step.
        # They are different quantities (membrane depth AT the protein vs the protein's
        # own z) in different units, and tying them through one gain was measured over
        # the 18682-frame pool at delta corr 0.621 -- 38.5% of the node-depth step
        # variance explained, 61.5% thrown away.
        node_new = m_node + node_depth_delta.squeeze(-1).to(m_node.dtype)
        hyper_new = m_hyper + edge_delta[..., 2].to(m_hyper.dtype)

        if self.arms_enabled and self.geom_spec['advances_blade']:
            blade_new = rodrigues(blade_state, self.rotation_head(emb).unsqueeze(2))
        else:
            blade_new = blade_state
        return (x_pred, blade_new, node_new, hyper_new, enc_new, z_new,
                latent_entropy, latent_top1_prob, step_delta)

    def _run_step(self, do_ckpt, *args):
        if do_ckpt:
            return ckpt_util.checkpoint(self._forward_step_hg, *args, use_reentrant=False)
        return self._forward_step_hg(*args)

    # ----- forward (training) -----

    def forward(self, x_sequence, target_sequence=None, active_mask=None,
                edge_src=None, edge_dst=None, edge_mask=None, start_frame_idx=None,
                theta_gt=None, blade=None, blade_gt=None):
        if target_sequence is None:
            target_sequence = x_sequence
        B, S, _ = target_sequence.size()
        if active_mask is not None:
            raise NotImplementedError(
                "hgfirst has no multi-system path: the hyperedge incidence table is "
                "fixed at P=n_particles, so a padded/masked batch has no meaning.")
        if blade is None:
            blade = self.default_blade(B, dtype=x_sequence.dtype)

        enc_states = self.encoder.init_state(B, self.device)
        z = torch.zeros(B, self.latent_dim, device=self.device)
        cx = x_sequence[:, 0, :]
        blade_state = blade
        m_node, m_hyper = self.init_membrane(cx, blade_state)
        m_node_init, m_hyper_init = m_node, m_hyper

        preds, m_node_preds, m_hyper_preds = [], [], []
        capture_z = getattr(self, 'w_latent_corr', 0.0) > 0
        z_stack = []
        capture_blades = (self.arms_enabled and self.geom_spec['advances_blade']
                          and self.rotation_losses_enabled())
        blade_ps = []
        latent_entropy_sum, latent_top1_sum, latent_step_count = self._init_latent_diagnostics()
        # Gradient checkpointing is REQUIRED for the long curriculum (seq -> 3000):
        # the encoder alone holds [B,E,h] activations per message pass, so the full
        # unrolled tape OOMs without it.
        do_ckpt = self.use_gradient_checkpointing and self.training

        for t in range(S - 1):
            (xp, blade_state, m_node, m_hyper, enc_states, z,
             latent_entropy, latent_top1, _) = self._run_step(
                do_ckpt, cx, blade_state, m_node, m_hyper, enc_states, z, t)
            latent_entropy_sum, latent_top1_sum, latent_step_count = self._update_latent_diagnostics(
                latent_entropy_sum, latent_top1_sum, latent_step_count,
                latent_entropy, latent_top1)
            preds.append(xp)
            m_node_preds.append(m_node)
            m_hyper_preds.append(m_hyper)
            if capture_z:
                z_stack.append(z)
            if capture_blades:
                blade_ps.append(blade_state)
            cx = xp

        # --- Loss computation (same contract as model_multilag.forward) ---
        xps = torch.stack(preds, dim=1)
        del preds
        A = xps.size(1)
        tp = target_sequence[:, 1:A + 1]

        pred_pw = self._pairwise_xy(xps)
        tgt_pw = self._pairwise_xy(tp)
        qfn = self._get_quantile_fn()
        pl = qfn(pred_pw, tgt_pw)
        cl = self.compute_cohesion_loss(xps, tp, pred_pw=pred_pw, tgt_pw=tgt_pw)
        ph = self.compute_pairwise_tail_loss(pred_pw, tgt_pw) if self.w_pairwise_tail > 0 else None
        ev = self.compute_excluded_volume_loss(pred_pw, tgt_pw) if self.w_excluded_volume > 0 else None
        del pred_pw, tgt_pw

        pz = torch.remainder(
            xps[:, :, :self.n_coords].reshape(B, -1, self.n_particles, 3)[:, :, :, 2],
            self.box_size) / self.box_size
        tz = torch.remainder(
            tp[:, :, :self.n_coords].reshape(B, -1, self.n_particles, 3)[:, :, :, 2],
            self.box_size) / self.box_size

        puw, tuw = self._unwrap_xy(xps), self._unwrap_xy(tp)
        dl = self.compute_displacement_quantile_loss(
            xps, tp, pred_xy_unwrapped=puw, target_xy_unwrapped=tuw)
        al = self.compute_angle_quantile_loss(
            xps, tp, pred_xy_unwrapped=puw, target_xy_unwrapped=tuw) if self.w_angle > 0 else None
        vf = self.compute_vacf_quantile_loss(
            xps, tp, pred_xy_unwrapped=puw, target_xy_unwrapped=tuw)
        vf_mse = (self.compute_vacf_mse_loss(xps, tp, pred_xy_unwrapped=puw,
                                             target_xy_unwrapped=tuw)
                  if getattr(self, 'w_vacf_mse', 0.0) > 0 else None)
        alpha_l = self.compute_alpha_loss(puw, tuw) if getattr(self, 'w_alpha', 0.0) > 0 else None
        vacf_sp = (self.compute_vacf_spatial_l1_loss(puw, tuw)
                   if getattr(self, 'w_vacf_spatial', 0.0) > 0 else None)
        kurt_l = (self.compute_kurtosis_loss(puw, tuw)
                  if getattr(self, 'w_kurtosis', 0.0) > 0 else None)
        del puw, tuw

        if capture_z and z_stack:
            from membrane.vacf_loss import compute_latent_corr_loss as _lat_corr
            latent_corr_loss = _lat_corr(
                torch.stack(z_stack, dim=1),
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
            'z_quantile_loss': self.compute_z_quantile_loss(pz, tz),
            'z_displacement_loss': self.compute_z_displacement_quantile_loss(xps, tp),
            'vacf_loss': vf,
            'cohesion_loss': cl,
            'latent_entropy': latent_entropy,
            'latent_top1_prob': latent_top1_prob,
            # [B,A,7] / [B,A,120] + the IC prediction; training_step pops these and
            # compares them to the GT depth windows.
            'mem_node_pred': torch.stack(m_node_preds, dim=1),
            'mem_hyper_pred': torch.stack(m_hyper_preds, dim=1),
            'mem_node_init': m_node_init,
            'mem_hyper_init': m_hyper_init,
        }
        if al is None:
            del out['angle_loss']
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
            out['pred_blades'] = pred_blades
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
                if self.w_theta_step > 0:
                    out['theta_step_loss'] = self.compute_blade_theta_step_loss(
                        pred_blades, gt_blades)
        if getattr(self, 'w_mem_consistency', 0.0) > 0 and getattr(self, 'mem_consistency_slope', 0.0):
            out['mem_consistency'] = self._mem_consistency_loss(out['mem_node_pred'], pz)
        return out

    # ----- inference -----

    def predict_future(self, initial_x, steps, return_unwrapped=False,
                       active_mask=None, edge_src=None, edge_dst=None, edge_mask=None,
                       start_frame_idx=None, return_membrane=False, initial_theta=None,
                       blade=None):
        B = initial_x.size(0)
        if active_mask is not None:
            raise NotImplementedError("hgfirst has no multi-system path.")
        # No GT blades at inference (the stability callback and the eval scripts hand
        # coordinates only): default_blade draws from the measured GT tilt
        # distribution under a fixed seed, so this IC adds no unreproducible noise to
        # the monitored objective.
        if blade is None:
            blade = self.default_blade(B, dtype=initial_x.dtype)

        enc_states = self.encoder.init_state(B, self.device)
        z = torch.zeros(B, self.latent_dim, device=self.device)
        cx = initial_x[:, 0, :]
        cu = cx.clone()
        blade_state = blade
        m_node, m_hyper = self.init_membrane(cx, blade_state)

        ps, pu, mns, mhs = [], [], [], []
        for t in range(steps):
            (xp, blade_state, m_node, m_hyper, enc_states, z,
             _, _, ds) = self._forward_step_hg(cx, blade_state, m_node, m_hyper,
                                               enc_states, z, t)
            # The wrapped position is what the model advances on; the unwrapped track
            # accumulates the same steps without wrapping (MSD / VACF metrics).
            nu = cu + ds
            ps.append(xp)
            pu.append(nu)
            if return_membrane:
                mns.append(m_node)
                mhs.append(m_hyper)
            cx, cu = xp, nu

        sw = torch.stack(ps, dim=1)
        if return_membrane:
            mem = (torch.stack(mns, dim=1), torch.stack(mhs, dim=1))
            return (sw, torch.stack(pu, dim=1), *mem) if return_unwrapped else (sw, *mem)
        return (sw, torch.stack(pu, dim=1)) if return_unwrapped else sw
