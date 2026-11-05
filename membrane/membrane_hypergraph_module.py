"""Autoregressive membrane-hypergraph module (decision d4ba47f5).

Predicts the bilayer membrane state as a k=2..7 hypergraph of z-deformation
DEPTHS (7 protein-node depths + 120 hyperedge junction depths) and emits a
per-protein context vector that REPLACES the geometry embedding (`curr_emb`) in
the decoder's cross-attention.

Two entry points, sharing one AllSet (star/bipartite) message-passing trunk:
  init_head(coords_0)            -> membrane_0 + context   (positions -> hypergraph; seeds rollout)
  step(node_d, hyper_d, z, coords) -> membrane_{t+1} + context  (autoregressive propagator)

All features are rotation/translation-INVARIANT scalars: depths, min-image
pairwise distances, hyperedge order. No raw xy is used as a feature (only to
compute invariant distances). z (the shared transition latent) supplies the
protein-movement signal that resolves the membrane<->motion chicken-and-egg.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEPTH_MODES = ('absolute', 'delta', 'ou')


def _mlp(din, dh, dout, act=nn.SiLU):
    return nn.Sequential(nn.Linear(din, dh), act(), nn.Linear(dh, dout))


def _check_ou_domain(a_node, a_hyper, sg_node, sg_hyper):


    """a in (0,1) and sigma > 0 BEFORE the inverse transforms.



    _logit(1.0) is +inf and _inv_softplus(0.0) is -inf, and both build a model


    that looks fine and NaNs on the first optimizer step. 'a=1' (full reversion)


    and 'sigma=0' (no noise) are both things someone would plausibly write.


    """


    for nm, v in (('a_node', a_node), ('a_hyper', a_hyper)):


        if not (0.0 < float(v) < 1.0):


            raise ValueError(f'membrane_ou_{nm}={v} must be in (0,1); '


                             f'0 and 1 map to -inf/+inf under logit')


    for nm, v in (('sigma_node', sg_node), ('sigma_hyper', sg_hyper)):


        if not float(v) > 0.0:


            raise ValueError(f'membrane_ou_{nm}={v} must be > 0; '


                             f'0 maps to -inf under inverse-softplus. '


                             f'Use membrane_ou_noise=False to disable noise.')



def _logit(p):
    return torch.log(p) - torch.log1p(-p)


def _inv_softplus(x):
    # log(expm1(x)), written so a small x (sigma ~0.065) keeps its precision
    return x + torch.log(-torch.expm1(-x))


class MembraneHypergraphModule(nn.Module):
    def __init__(self, out_dim, members, ksize, n_particles=7, hidden=64,
                 layers=2, latent_dim=21, box=856.196,
                 delta_mode=False, delta_scale_node=1.0, delta_scale_hyper=1.0,
                 depth_mode=None, ou_a_node=0.002269, ou_a_hyper=0.005845,
                 ou_noise=True, ou_sigma_node=0.064967, ou_sigma_hyper=0.106308,
                 edge_latent=False,
                 standardize_inputs=False, calib_coords=None,
                 calib_node_depth=None, calib_hyper_depth=None):
        """out_dim = geo_embed.output_dim (decoder context contract).
        members [120,7] bool incidence; ksize [120] hyperedge orders.

        delta_mode: if True, `step` predicts a NORMALIZED per-frame CHANGE and
        integrates it as a residual: depth_{t+1} = depth_t + scale * head(...).
        The frame-to-frame membrane change is ~10-13x smaller than the absolute
        depth range, so the heads output O(1) numbers (target = delta / scale)
        and `scale` = the measured per-frame delta std maps them back to the tiny
        physical step. `init_head` is ALWAYS absolute (it seeds membrane_0 from
        geometry, no previous state to add to).

        depth_mode supersedes delta_mode and names all three propagators:
          'absolute' depth_{t+1} = head(...)                        (default, as before)
          'delta'    depth_{t+1} = depth_t + scale * head(...)      (== delta_mode)
          'ou'       depth_{t+1} = depth_t + a*(mu - depth_t) + sigma*eps, mu = head
        'ou' is the one that matches what the depth actually IS. It is a CONFINED
        process -- log-log MSD slope 0.62 against position's 1.09, lag-1 value
        autocorrelation 0.998, lag-1000 still 0.735, and 76% of its variance is a
        STATIC between-protein offset. 'absolute' never diverges because the head
        re-emits the level every frame, but then nothing carries state; 'delta'
        carries state but integrates a free-running increment, so any residual bias
        in the head becomes b*t and an untrained 3000-step rollout reaches |depth|
        3523 against a GT range of -8.2..+1.2. OU is both: mu carries the level (and
        so the static structure), while `a` bounds the walk -- the state has a fixed
        point instead of a slope.

        ou_a_*: per-channel relaxation rate, learned through a sigmoid, initialised
        from 1 - (measured lag-1 value autocorrelation) = 0.00227 node / 0.00585
        hyper. ou_sigma_*: only used when ou_noise=True, initialised to the measured
        per-frame step std in the module's own (dataset-normalized) units, 0.065 node
        / 0.106 hyper. Those two measurements are consistent with each other and with
        the normalization: an OU at (a, sigma) has stationary std sigma/sqrt(2a-a^2)
        = 0.965 (node) / 0.985 (hyper), i.e. the unit-variance field the dataset
        hands over. Defaults are the pooled 7piezo numbers, like `box`.

        standardize_inputs: per-FEATURE centre/scale of every DATA channel entering
        the encoders, measured from calib_coords / calib_node_depth /
        calib_hyper_depth -- a sample of real frames, with the depths already divided
        by the dataset's pooled std (that is the unit the model works in)."""
        super().__init__()
        self.P = int(n_particles)
        self.E = int(members.shape[0])
        self.h = int(hidden)
        self.box = float(box)
        # delta_mode stays the bool it was so existing call sites keep working; when
        # both are given depth_mode wins, and model_multilag rejects the pair if they
        # disagree rather than silently picking one.
        self.depth_mode = depth_mode or ('delta' if delta_mode else 'absolute')
        if self.depth_mode not in DEPTH_MODES:
            raise ValueError(f'depth_mode must be one of {DEPTH_MODES}, '
                             f'got {self.depth_mode!r}')
        self.delta_mode = self.depth_mode == 'delta'
        # per-target residual scales. NON-persistent: they are data-derived constants
        # (reconstructed deterministically at build time from the membrane npz), not
        # learned params -- so they stay out of the checkpoint and old absolute-mode
        # checkpoints load without reporting them missing.
        self.register_buffer('delta_scale_node',
                             torch.as_tensor(float(delta_scale_node), dtype=torch.float32),
                             persistent=False)
        self.register_buffer('delta_scale_hyper',
                             torch.as_tensor(float(delta_scale_hyper), dtype=torch.float32),
                             persistent=False)
        H = torch.as_tensor(members, dtype=torch.float32)            # [E,P]
        self.register_buffer('H', H, persistent=False)
        self.register_buffer('ksize', torch.as_tensor(ksize, dtype=torch.float32), persistent=False)
        self.register_buffer('node_deg', H.sum(0).clamp(min=1.0), persistent=False)   # [P]
        self.register_buffer('edge_deg', H.sum(1).clamp(min=1.0), persistent=False)   # [E]

        NODE_GEO, HYPER_GEO, ZC = 3, 2, 8
        # input encoders (init has no depth/z; step has prev depth + z)
        self.init_node = _mlp(NODE_GEO, hidden, hidden)
        self.init_edge = _mlp(HYPER_GEO, hidden, hidden)
        self.z_to_node = _mlp(latent_dim, hidden, self.P * ZC)
        self.step_node = _mlp(1 + ZC + NODE_GEO, hidden, hidden)
        # THE HYPEREDGE LATENT PATH. Without edge_latent, step_edge sees only
        # [hyper_depth, ksize/P, mean_pd/box] -- NO latent at all. z reaches the
        # hyperedges solely through the AllSet node->edge aggregation, measured at
        # 0.15-0.72% of the hyperedge step RMS against 20-84% for nodes, and the
        # hyperedge reversion rate gets a gradient 17,000x smaller than the node one.
        # That matters because the hyperedge LEVEL is largely identifiable (per-edge
        # means correlate r=+0.87 between disjoint blocks of the trajectory) but its
        # AMPLITUDE tracks the configuration (spread 21.5 -> 15.0 as the system
        # aggregates), and the configuration is exactly what z carries.
        # POOLED OVER MEMBERS, not a per-slot projection: data/membrane_dataset.py
        # remaps the 120 hyperedge slots with hyperedge_remap under the same S7 draw
        # as the proteins, so a fixed per-slot map would break permutation
        # equivariance. H @ z_node / edge_deg follows the relabelling exactly.
        self.edge_latent = bool(edge_latent)
        _zc = ZC if self.edge_latent else 0
        self.step_edge = _mlp(1 + HYPER_GEO + _zc, hidden, hidden)

        # shared AllSet trunk
        self.n2e = nn.ModuleList(_mlp(hidden, hidden, hidden) for _ in range(layers))
        self.e_up = nn.ModuleList(_mlp(2 * hidden, hidden, hidden) for _ in range(layers))
        self.e2n = nn.ModuleList(_mlp(hidden, hidden, hidden) for _ in range(layers))
        self.n_up = nn.ModuleList(_mlp(2 * hidden, hidden, hidden) for _ in range(layers))
        self.layers = layers

        # shared output heads
        self.node_out = nn.Linear(hidden, 1)
        self.hyper_out = nn.Linear(hidden, 1)
        # ZERO-MEAN AT INIT. A free bias on a head whose output is INTEGRATED
        # (delta_mode: depth <- depth + scale*raw) is a constant velocity: bias b
        # becomes b*t in the depth. Measured over 30 random inits, the untrained
        # per-step delta was 100% one-signed -- 13/30 seeds all-negative, 12/30
        # all-positive -- correlating 0.78 with node_out.bias, against a ground
        # truth that changes sign 50.1% of the time. Starting at zero removes the
        # init-dependent ramp.
        nn.init.zeros_(self.node_out.bias)
        nn.init.zeros_(self.hyper_out.bias)
        self.context = nn.Sequential(nn.LayerNorm(self.P * hidden),
                                     nn.Linear(self.P * hidden, out_dim), nn.SiLU())

        # --- OU propagator (depth_mode='ou') -------------------------------------
        # SEPARATE output heads for `step`. node_out/hyper_out are shared between
        # init_head and step in the other two modes, and under OU the two roles pull
        # the same weights in opposite directions: init_head has to emit the absolute
        # seed level (node mean -2.43 in normalized units) from geometry alone, while
        # step's mu is read through a*(mu - depth) with a ~ 0.002, i.e. it is the
        # attractor the state creeps toward over a 1/a ~ 440-frame timescale. One head
        # cannot be both the one-shot IC estimator and the slow attractor. 'absolute'
        # and 'delta' keep the shared head, untouched.
        self.ou_noise = bool(ou_noise) and self.depth_mode == 'ou'
        if self.depth_mode == 'ou':
            self.step_node_out = nn.Linear(hidden, 1)
            self.step_hyper_out = nn.Linear(hidden, 1)
            # Zero bias for the same reason the shared heads have one, though the
            # failure it avoids is different: here a bias is not a velocity but the
            # untrained FIXED POINT, and seeding it at the data mean while init_head
            # still seeds near 0 would make every early step a one-signed relaxation
            # across the 440-frame gap. At zero the untrained process sits where
            # init_head puts it and the step is driven by mu's fluctuation, which is
            # sign-balanced; supervision then moves both heads to the real level.
            nn.init.zeros_(self.step_node_out.bias)
            nn.init.zeros_(self.step_hyper_out.bias)
            # Per-channel, and ALL start at one measured value rather than a per-channel
            # fit because both axes are exchangeable under the dataset's permutation
            # augmentation: the 7 node slots obviously, and the 120 hyperedge slots too
            # -- data/membrane_dataset.py remaps them with hyperedge_remap(perm, ...)
            # under the same S7 draw, so they are NOT canonical identities.
            _check_ou_domain(ou_a_node, ou_a_hyper, ou_sigma_node, ou_sigma_hyper)
            self.ou_a_logit_node = nn.Parameter(
                _logit(torch.full((self.P,), float(ou_a_node))))
            self.ou_a_logit_hyper = nn.Parameter(
                _logit(torch.full((self.E,), float(ou_a_hyper))))
            if self.ou_noise:
                self.ou_sigma_raw_node = nn.Parameter(
                    _inv_softplus(torch.full((self.P,), float(ou_sigma_node))))
                self.ou_sigma_raw_hyper = nn.Parameter(
                    _inv_softplus(torch.full((self.E,), float(ou_sigma_hyper))))

        # --- per-feature input standardization (default OFF) ---------------------
        # The channels entering step_node/step_edge (and init_node/init_edge) are
        # wildly unequal in scale. data/membrane_dataset.py divides the depths by
        # their POOLED VALUE std and never subtracts the mean, so depth arrives at
        # std 1.0 with mean -2.43 (node) / -3.65 (hyper), while _geom's own channels
        # measure std 0.091 / 0.100 / 0.037 (mean_d, min_d, std_d) and 0.167 / 0.101
        # (ksize/P, mean_pd/box) over the 18.7k-frame pool. That is 27x across the
        # node block and 6x across the edge block, on top of a -2.4 DC offset the
        # first Linear must cancel with its bias before any geometry can register at
        # all. Centring and scaling per FEATURE puts every one at mean 0 std 1.
        # z_node is deliberately NOT standardized: it is an MLP output (measured std
        # 0.199 at init), not data, so its scale is z_to_node's to set.
        self.standardize_inputs = bool(standardize_inputs)
        if self.standardize_inputs:
            if calib_coords is None or calib_node_depth is None or calib_hyper_depth is None:
                raise ValueError(
                    'standardize_inputs=True needs calib_coords/calib_node_depth/'
                    'calib_hyper_depth: the stats are MEASURED (the geometry ones by '
                    'running _geom over real frames), never assumed')
            self._calibrate_inputs(calib_coords, calib_node_depth, calib_hyper_depth)

    def _calibrate_inputs(self, coords, node_depth, hyper_depth):
        """Measure (mean, std) per input feature and park them as NON-persistent
        buffers. Non-persistent for the same reason delta_scale_* is: they are
        data-derived constants reconstructed deterministically at every build, and
        persistent=True would let a checkpoint reinstate stale stats over freshly
        measured ones."""
        coords = torch.as_tensor(coords, dtype=torch.float32)
        node_depth = torch.as_tensor(node_depth, dtype=torch.float32)
        hyper_depth = torch.as_tensor(hyper_depth, dtype=torch.float32)
        with torch.no_grad():
            node_geo, hyper_geo = self._geom(coords)                 # [F,P,3], [F,E,2]

        def reg(name, x, dim):
            # Pool over every axis but the feature axis: the node/hyperedge index is
            # exchangeable under the dataset's permutation augmentation, so a
            # per-INDEX statistic would not survive relabelling. clamp_min keeps a
            # degenerate channel (a constant one in a tiny calibration sample) from
            # dividing by zero.
            flat = x.reshape(-1, dim)
            self.register_buffer(f'in_mu_{name}', flat.mean(0), persistent=False)
            self.register_buffer(f'in_sd_{name}', flat.std(0).clamp_min(1e-6),
                                 persistent=False)

        reg('node_geo', node_geo, 3)
        reg('hyper_geo', hyper_geo, 2)
        reg('node_depth', node_depth, 1)
        reg('hyper_depth', hyper_depth, 1)

    def _std_node_in(self, node_depth, node_geo):
        if not self.standardize_inputs:
            return node_depth, node_geo
        node_geo = (node_geo - self.in_mu_node_geo) / self.in_sd_node_geo
        if node_depth is not None:
            node_depth = (node_depth - self.in_mu_node_depth) / self.in_sd_node_depth
        return node_depth, node_geo

    def _std_edge_in(self, hyper_depth, hyper_geo):
        if not self.standardize_inputs:
            return hyper_depth, hyper_geo
        hyper_geo = (hyper_geo - self.in_mu_hyper_geo) / self.in_sd_hyper_geo
        if hyper_depth is not None:
            hyper_depth = (hyper_depth - self.in_mu_hyper_depth) / self.in_sd_hyper_depth
        return hyper_depth, hyper_geo

    # ---- invariant geometry features from current coords ----
    def _geom(self, coords):
        B = coords.shape[0]
        xy = coords.view(B, self.P, 3)[:, :, :2]                     # [B,P,2]
        d = xy[:, :, None, :] - xy[:, None, :, :]                    # [B,P,P,2]
        d = d - self.box * torch.round(d / self.box)                # min-image
        # +eps inside sqrt so the zero diagonal (self-distance) has finite grad.
        dist = torch.sqrt((d ** 2).sum(-1) + 1e-12)                 # [B,P,P]
        eye = torch.eye(self.P, device=coords.device)
        mask = 1.0 - eye                                            # [P,P] off-diagonal
        denom = (self.P - 1)
        mean_d = (dist * mask).sum(2) / denom                      # [B,P]
        min_d = (dist + eye * 1e9).min(dim=2).values               # [B,P] (skip self)
        var_d = (((dist - mean_d[..., None]) ** 2) * mask).sum(2) / denom
        std_d = torch.sqrt(var_d + 1e-6)
        node_geo = torch.stack([mean_d, min_d, std_d], -1) / self.box   # [B,P,3]
        # hyperedge geometry: ksize, mean intra-subset pair distance (diag dist~0)
        npair = (self.ksize * (self.ksize - 1)).clamp(min=1.0)      # [E]
        mean_pd = torch.einsum('ei,bij,ej->be', self.H, dist, self.H) / npair[None]
        hyper_geo = torch.stack([(self.ksize / self.P).expand(B, -1),
                                 mean_pd / self.box], -1)            # [B,E,2]
        return node_geo, hyper_geo

    # ---- AllSet trunk: node/hyperedge embeddings -> (node_emb, edge_emb) ----
    def _allset(self, node_h, edge_h):
        for l in range(self.layers):
            em = torch.einsum('en,bnh->beh', self.H, self.n2e[l](node_h)) / self.edge_deg[None, :, None]
            edge_h = edge_h + self.e_up[l](torch.cat([edge_h, em], -1))
            nm = torch.einsum('en,beh->bnh', self.H, self.e2n[l](edge_h)) / self.node_deg[None, :, None]
            node_h = node_h + self.n_up[l](torch.cat([node_h, nm], -1))
        return node_h, edge_h

    def _finish(self, node_h, edge_h, node_head=None, hyper_head=None):
        """Raw head outputs. In 'delta' these are the NORMALIZED residual
        (node/hyper) and `step` integrates them; in 'ou' they are mu, the level the
        state relaxes toward; otherwise they are absolute depths."""
        node_head = self.node_out if node_head is None else node_head
        hyper_head = self.hyper_out if hyper_head is None else hyper_head
        node_raw = node_head(node_h).squeeze(-1)                   # [B,P]
        hyper_raw = hyper_head(edge_h).squeeze(-1)                # [B,E]
        ctx = self.context(node_h.reshape(node_h.shape[0], -1))    # [B,out_dim]
        return node_raw, hyper_raw, ctx

    def init_head(self, coords):
        node_geo, hyper_geo = self._geom(coords)
        _, node_geo = self._std_node_in(None, node_geo)
        _, hyper_geo = self._std_edge_in(None, hyper_geo)
        node_h, edge_h = self._allset(self.init_node(node_geo), self.init_edge(hyper_geo))
        return self._finish(node_h, edge_h)

    def step(self, node_depth, hyper_depth, z, coords):
        node_geo, hyper_geo = self._geom(coords)
        B = coords.shape[0]
        z_node = self.z_to_node(z).view(B, self.P, -1)             # [B,P,ZC]
        # Standardize only what goes INTO the encoders; the residual/OU integration
        # below runs on the raw node_depth/hyper_depth, which are the model's I/O
        # contract with the dataset and every eval script.
        node_d_in, node_geo = self._std_node_in(node_depth, node_geo)
        hyper_d_in, hyper_geo = self._std_edge_in(hyper_depth, hyper_geo)
        node_in = torch.cat([node_d_in[..., None], z_node, node_geo], -1)
        if self.edge_latent:
            z_edge = torch.einsum('en,bnh->beh', self.H, z_node) / self.edge_deg[None, :, None]
            edge_in = torch.cat([hyper_d_in[..., None], hyper_geo, z_edge], -1)
        else:
            edge_in = torch.cat([hyper_d_in[..., None], hyper_geo], -1)
        node_h, edge_h = self._allset(self.step_node(node_in), self.step_edge(edge_in))
        if self.depth_mode == 'ou':
            node_raw, hyper_raw, ctx = self._finish(node_h, edge_h,
                                                    self.step_node_out, self.step_hyper_out)
            # Mean reversion: the increment is PROPORTIONAL to the distance from mu,
            # so the state cannot run away the way a free-running residual does -- the
            # walk has a fixed point, not a slope. sigmoid keeps a in (0,1), which is
            # exactly the stability band of the AR(1) this is (|1-a|<1).
            a_node = torch.sigmoid(self.ou_a_logit_node)
            a_hyper = torch.sigmoid(self.ou_a_logit_hyper)
            node_out = node_depth + a_node * (node_raw - node_depth)
            hyper_out = hyper_depth + a_hyper * (hyper_raw - hyper_depth)
            if self.ou_noise:
                # The diffusion half of the OU. Off by default so the first experiment
                # isolates mean reversion: with it off the propagator is deterministic
                # given (depth, z, coords) and any change in the rollout is the
                # reversion, not fresh noise.
                node_out = node_out + F.softplus(self.ou_sigma_raw_node) * torch.randn_like(node_out)
                hyper_out = hyper_out + F.softplus(self.ou_sigma_raw_hyper) * torch.randn_like(hyper_out)
        else:
            node_raw, hyper_raw, ctx = self._finish(node_h, edge_h)
            if self.delta_mode:
                # residual integration: depth_{t+1} = depth_t + scale * normalized_delta
                node_out = node_depth + self.delta_scale_node * node_raw
                hyper_out = hyper_depth + self.delta_scale_hyper * hyper_raw
            else:
                node_out, hyper_out = node_raw, hyper_raw
        return node_out, hyper_out, ctx


if __name__ == '__main__':
    import numpy as np
    d = np.load('archive/data/membrane/membrane_train.npz', allow_pickle=True)
    members, ksize = d['members'], d['ksize']
    out_dim = 7 * 63 + 21                                          # geo_embed.output_dim
    mod = MembraneHypergraphModule(out_dim, members, ksize, hidden=64, layers=2)
    B = 4
    coords = torch.rand(B, 21) * 856.196
    z = torch.randn(B, 21)
    n0, h0, c0 = mod.init_head(coords)
    print('init:', n0.shape, h0.shape, c0.shape)
    assert n0.shape == (B, 7) and h0.shape == (B, 120) and c0.shape == (B, out_dim)
    n1, h1, c1 = mod.step(n0, h0, z, coords)
    print('step:', n1.shape, h1.shape, c1.shape)
    assert n1.shape == (B, 7) and h1.shape == (B, 120) and c1.shape == (B, out_dim)
    loss = (n1.sum() + h1.sum() + c1.sum())
    loss.backward()
    nparam = sum(p.numel() for p in mod.parameters())
    print(f'OK forward+backward; params={nparam/1e3:.1f}K')

    # delta (residual) mode: step must integrate scaled residual onto the input.
    dn = float(np.diff(d['node_depth'], axis=0).std())
    dh = float(np.diff(d['hyper_depth'], axis=0).std())
    modd = MembraneHypergraphModule(out_dim, members, ksize, hidden=64, layers=2,
                                    delta_mode=True, delta_scale_node=dn, delta_scale_hyper=dh)
    n0d, h0d, _ = modd.init_head(coords)                 # init is absolute
    n1d, h1d, _ = modd.step(n0d, h0d, z, coords)         # residual: n1d = n0d + scale*raw
    resid_node = ((n1d - n0d) / dn)                       # should be O(1) normalized head output
    assert n1d.shape == (B, 7) and h1d.shape == (B, 120)
    (n1d.sum() + h1d.sum()).backward()
    print(f'OK delta-mode; scale_node={dn:.3f} scale_hyper={dh:.3f} '
          f'|normalized residual| mean={resid_node.abs().mean().item():.3f}')

    # OU mode: bounded by construction. Free-run it far past the training horizon on
    # an UNTRAINED net -- delta mode blows past the -8.2..+1.2 GT range here, OU
    # cannot, because the increment shrinks as the state leaves mu.
    modo = MembraneHypergraphModule(out_dim, members, ksize, hidden=64, layers=2,
                                    depth_mode='ou')
    n, h, _ = modo.init_head(coords)
    with torch.no_grad():
        for _ in range(3000):
            n, h, _ = modo.step(n, h, torch.randn(B, 21), coords)
    print(f'OK ou-mode; 3000 untrained steps -> |node| max {n.abs().max().item():.3f} '
          f'|hyper| max {h.abs().max().item():.3f} '
          f'(a_node {torch.sigmoid(modo.ou_a_logit_node)[0].item():.5f})')
    assert n.abs().max().item() < 20 and h.abs().max().item() < 20
    print('ALL MODULE SELF-TESTS PASSED')
