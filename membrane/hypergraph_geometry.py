"""Membrane-first hypergraph geometry: the hypergraph IS the encoder and the decoder.

Design: docs/membrane/specs/2026-08-07_membrane_first_hypergraph_design.md.
Replaces the GAT/GNN geometry embedding (protein-only, with the membrane bolted on
downstream) and the protein decoder with two AllSet trunks over the P protein nodes
and the E k=2..P hyperedges. The encoder hands the LSTM bank a readout of EVERY
message pass; the decoder turns the latent back into a diagonal Gaussian over
DELTAS -- never absolute positions.

Node inputs are translation-invariant scalars plus the signed blade offsets (the
SE(3) shape/orientation channel the model would otherwise have to derive);
hyperedge inputs are order, intra-subset min-image distances, member tilt moments
and the edge's own depth. Position enters only through a periodic sin/cos encoding
of the wrapped xyz -- the encoding the GAT baseline also used -- and through
minimum-image differences: no raw unwrapped coordinate and no absolute pair vector
is ever a feature. That encoding is invariant under whole-box shifts but is NOT an
invariant of position: it is a bijection of in-box position, so a sub-box
translation moves it (and moves nothing else -- every other node channel and every
hyperedge channel is translation-invariant).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from membrane.geometry_features import blade_normal, cov_from_blades, tilt_angle

# sin/cos xyz (6) + blade offsets (9, signed) + normal (3) + tilt (1)
# + covariance moments (6) + own node depth (1) + invariant neighbour summary (3)
NODE_FEATS = 29
# k/P (1) + mean/min/std intra-subset pair distance (3) + mean/std member tilt (2)
# + own hyperedge depth (1) + mean member-to-centroid distance (1)
EDGE_FEATS = 8


def _mlp(din, dh, dout, act=nn.SiLU):
    return nn.Sequential(nn.Linear(din, dh), act(), nn.Linear(dh, dout))


def _min_image(d, box):
    return d - box * torch.round(d / box)


def _soft_bound(x, lo, hi):
    """Monotone, everywhere-differentiable map R -> (lo, hi). Replaces torch.clamp.

    clamp has EXACTLY zero gradient per saturated element, so a head whose whole
    pre-activation distribution has left the range never comes back: measured with
    node_out.bias[log_sigma rows] = -20 against lo = -8, |dL/dbias| and |dL/dW| are
    both 0.0 and stay 0.0. A bias just outside the bound is not fatal -- at +4.5
    against hi = +4 the weight term still straddles it and |dL/dbias| is 2.8e+02 --
    so this is a hazard when the whole distribution exits, not an unconditional trap.
    tanh keeps a usable gradient far outside: 9.9e-3 at 12 past either default bound,
    reaching exactly 0 only past ~50 (where fp32 tanh rounds to +-1).

    The price is a mild interior distortion: identity to 9.2e-3 within +-1 of the
    midpoint (0.9% in sigma) and 7.1e-2 within +-2 (7.4%). That is the argument for
    keeping the bounds WIDE -- tightening them onto the working range would start
    biasing sigma instead of just bounding it.
    """
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    return mid + half * torch.tanh((x - mid) / half)


def _geom_dtype(t):
    """Geometry runs at >= fp32 even under bf16 autocast: min-image differencing at
    box ~856 in bf16 has ~4 length units of resolution, which would wipe out the
    pairwise distances the trunk keys on. autocast is supposed to lower the Linears,
    not the coordinates."""
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


def _circular_mean_xy(xy, Hmean, box):
    """PBC circular mean of each hyperedge's member xy -- the closed form the dataset
    stores as centroid_xy (scripts/membrane/data_prep/membrane_hypergraph.py), evaluated as
    c = arg(mean_j exp(i theta_j)) independently per axis.

    This is the ONLY correct way to move an edge with its members. The incidence mean
    (1/k weights) is not it: the true differential is
    dc = sum_j cos(theta_j - c)/(k*R) dtheta_j with R = |mean_j exp(i theta_j)|, and
    the cos(theta_j - c)/R factor equals 1 only when every member is equally far in
    angle from the mean -- automatic at k=2 by symmetry, false at k>=3. Measured over
    7103 real frames the uniform-weight approximation misses 34-67% of the true edge
    displacement and discards 47-87% of its variance. Recomputing the POSITION here
    sidesteps the weights entirely and cannot drift over a rollout.

    Those weights are NOT bounded by 1. Over the 7103 frames the largest single
    weight is 501.1 and the largest row L1 norm is 1001.2 (1.3% of member weights
    exceed 1), because 1/R diverges as the members approach an evenly-spread
    configuration -- the SIGNED row sum stays exactly 1, as a mean must, so it is the
    L1 norm that bounds the amplification. Nothing here backpropagates through them
    today: the edge POSITION is what is used, and edge_delta[...,:2] is a convenience
    output no loss reads. A loss that DOES backprop through edge_delta[...,:2] would
    inherit that 1e3 amplification on those frames and needs its own conditioning cut
    -- see the R > 0.2 cut the tests use.
    """
    th = 2 * math.pi * xy / box
    Hb = Hmean[None, :, :, None]
    S = (Hb * torch.sin(th)[:, None]).sum(2)                     # [B,E,2]
    C = (Hb * torch.cos(th)[:, None]).sum(2)
    # R -> 0 means the members are spread so evenly around the box that the circular
    # mean is genuinely undefined -- an ambiguity in the TARGET's definition (the GT
    # centroid flips there too), not a numerical artifact. atan2 itself is safe there:
    # torch.atan2(0., 0.) is 0.0 forward and (dS, dC) = (0.0, -0.0) backward in both
    # float32 and float64, NOT NaN. What this guard buys is the NEAR-degenerate case:
    # d(atan2)/dS = C/R^2 diverges as R -> 0, and at R^2 ~ 1e-14 the gradient reaching
    # xy is ~1e6 unguarded versus 0 with the guard. Sanitizing atan2's ARGUMENTS
    # rather than its result is what makes that branch contribute zero gradient.
    #
    # Side effect, accepted: when the guard fires S,C become (0,1), so the edge snaps
    # to x = 0 -- the box corner, not the neighbourhood of its members. Acceptable
    # because the configuration it fires on has no defined answer to snap TO (any
    # point in the box is equally the circular mean) and it does not occur in the
    # data: over 7103 real frames x 120 edges x 2 axes min R^2 is 2.0e-10, i.e. 200x
    # the threshold, and the guard fires 0 times. The near-degenerate regime IS
    # reached (1683 of those R^2 land under 1e-4, a ~1e4 amplification), which is the
    # regime the threshold is really sized for.
    ok = (S * S + C * C) > 1e-12
    S = torch.where(ok, S, torch.zeros_like(S))
    C = torch.where(ok, C, torch.ones_like(C))
    return torch.remainder(torch.atan2(S, C), 2 * math.pi) * (box / (2 * math.pi))


def _masked_mean_std(v, mask, count, eps):
    """Mask-weighted mean/std over the last axis.

    Two-pass (centre, then square) rather than E[x^2] - E[x]^2: the one-pass form
    cancels catastrophically exactly where this has to stay well behaved -- a subset
    whose members are nearly coincident -- and can land slightly negative, which
    sqrt turns into NaN. eps inside the sqrt keeps the gradient finite at var == 0
    (sqrt'(0) = inf); dvar/dx is itself 0 there, so unguarded that is 0 * inf = NaN
    reaching backward, the same failure mode tilt_angle's atan2 formulation avoids.
    """
    m = (v * mask).sum(-1) / count
    var = ((v - m.unsqueeze(-1)) ** 2 * mask).sum(-1) / count
    return m, torch.sqrt(var + eps)


class AllSetTrunk(nn.Module):
    """Star/bipartite hypergraph message passing (node -> edge -> node), the same
    scheme as MembraneHypergraphModule._allset, but returning EVERY layer's
    (node_h, edge_h) instead of only the last so a caller can read each message
    pass (design Decision 3: all layers feed all lags)."""

    def __init__(self, hidden, layers, members):
        super().__init__()
        self.hidden, self.layers = int(hidden), int(layers)
        H = torch.as_tensor(members, dtype=torch.float32)
        if H.dim() != 2:
            raise ValueError(f'members must be [E,P], got {tuple(H.shape)}')
        self.E, self.P = int(H.shape[0]), int(H.shape[1])
        # persistent=False: H and the degrees are data-derived constants rebuilt
        # deterministically from the membrane npz at build time, not learned params,
        # so they stay out of the checkpoint (MembraneHypergraphModule's convention).
        self.register_buffer('H', H, persistent=False)
        # clamp_min(1): an isolated node or an empty hyperedge would divide by 0.
        # Neither occurs in the complete k=2..P table, but this trunk is also built
        # on synthetic incidence tables.
        self.register_buffer('node_deg', H.sum(0).clamp_min(1.0), persistent=False)
        self.register_buffer('edge_deg', H.sum(1).clamp_min(1.0), persistent=False)
        self.n2e = nn.ModuleList(_mlp(hidden, hidden, hidden) for _ in range(self.layers))
        self.e_up = nn.ModuleList(_mlp(2 * hidden, hidden, hidden) for _ in range(self.layers))
        self.e2n = nn.ModuleList(_mlp(hidden, hidden, hidden) for _ in range(self.layers))
        self.n_up = nn.ModuleList(_mlp(2 * hidden, hidden, hidden) for _ in range(self.layers))

    def forward(self, node_h, edge_h):
        """-> [(node_h, edge_h)] * layers, outermost list ordered by message pass."""
        # cast the structure buffers to the activation dtype: under autocast the MLPs
        # emit bf16 and dividing by an fp32 degree would promote the [B,E,h] message
        # back to fp32 on every layer, which is neither what autocast was asked for
        # nor free at E=120.
        dt = node_h.dtype
        H, edeg, ndeg = self.H.to(dt), self.edge_deg.to(dt), self.node_deg.to(dt)
        outs = []
        for l in range(self.layers):
            m = torch.einsum('en,bnh->beh', H, self.n2e[l](node_h)) / edeg[None, :, None]
            edge_h = edge_h + self.e_up[l](torch.cat([edge_h, m], -1))
            m = torch.einsum('en,beh->bnh', H, self.e2n[l](edge_h)) / ndeg[None, :, None]
            node_h = node_h + self.n_up[l](torch.cat([node_h, m], -1))
            outs.append((node_h, edge_h))
        return outs


class HyperGeoEncoder(nn.Module):
    """Hypergraph geometry embedding. Replaces GAT/GNN entirely.

    forward(coords, blade, node_depth, hyper_depth) -> (readout, (node_h, edge_h)),
    where readout concatenates, per message pass, the P node vectors and one
    mean-pooled edge vector per hyperedge order k. The per-order pool (rather than
    one pool over all E) keeps the k-dependent offset ladder the data shows
    (+13.00 at k=2 down to -16.20 at k=7) legible to the bank; it is also
    permutation-invariant, since relabelling proteins permutes edges within an
    order and never across orders.
    """

    def __init__(self, hidden, layers, members, ksize, box=856.1959882301632,
                 arm_radius=31.63368034362793):
        super().__init__()
        self.trunk = AllSetTrunk(hidden, layers, members)
        self.P, self.E = self.trunk.P, self.trunk.E
        self.hidden, self.layers = self.trunk.hidden, self.trunk.layers
        if self.P < 2:
            raise ValueError('need P >= 2: the neighbour summary and every k>=2 '
                             'hyperedge are undefined for a single protein')
        self.box, self.arm_radius = float(box), float(arm_radius)

        k = torch.as_tensor(ksize, dtype=torch.float32).reshape(-1)
        # ksize is carried separately in the npz from members; if the two disagree the
        # order features and the incidence table describe different hypergraphs and
        # nothing downstream would notice. Fail at build time instead.
        if k.shape[0] != self.E or not torch.equal(k, self.trunk.H.sum(1)):
            raise ValueError('ksize does not match the members incidence table')
        self.register_buffer('ksize', k, persistent=False)
        orders = torch.unique(k)                                     # ascending
        kg = torch.stack([(k == o).to(k.dtype) for o in orders])     # [K,E]
        self.register_buffer('kg', kg / kg.sum(1, keepdim=True).clamp_min(1.0),
                             persistent=False)
        self.n_orders = int(orders.shape[0])

        H = self.trunk.H
        eye = torch.eye(self.P)
        # ordered intra-subset pairs (i != j, both members), flattened so one
        # mask-weighted reduction covers mean/std and one sentinel covers min.
        pm = (H[:, :, None] * H[:, None, :] * (1.0 - eye)).reshape(self.E, self.P * self.P)
        self.register_buffer('pair_mask', pm, persistent=False)
        self.register_buffer('n_pair', pm.sum(-1).clamp_min(1.0), persistent=False)
        self.register_buffer('off_diag', 1.0 - eye, persistent=False)
        self.register_buffer('Hmean', H / H.sum(1, keepdim=True).clamp_min(1.0),
                             persistent=False)

        self.in_node = _mlp(NODE_FEATS, hidden, hidden)
        self.in_edge = _mlp(EDGE_FEATS, hidden, hidden)
        self.readout_dim = self.layers * (self.P + self.n_orders) * hidden

    def geom_features(self, coords, blade, node_depth, hyper_depth):
        """-> (node [B,P,NODE_FEATS], hyperedge [B,E,EDGE_FEATS]) raw inputs."""
        B = coords.shape[0]
        xyz = _geom_dtype(coords).reshape(B, self.P, 3)
        dt, box = xyz.dtype, self.box
        blade = _geom_dtype(blade).to(dt)
        # Depths are consumed RAW (no /box, no rescale) because they arrive already
        # normalized to ~unit variance by data/membrane_dataset.py:140-143. A caller
        # feeding the npz's own node_depth/hyper_depth instead would hand this block
        # values at |mean| ~65, std ~28 -- 170-310x every other channel.
        node_depth = node_depth.to(dt).reshape(B, self.P)
        hyper_depth = hyper_depth.to(dt).reshape(B, self.E)
        H, kf = self.trunk.H.to(dt), self.ksize.to(dt)

        sc = 2 * math.pi * torch.remainder(xyz, box) / box
        trig = torch.cat([torch.sin(sc), torch.cos(sc)], -1)                # [B,P,6]
        # blade offsets normalized by arm_radius so the signed SE(3) block lands O(1)
        # beside the sin/cos block; raw offsets run ~30 in these length units (the
        # same reason cov_from_blades divides by arm_radius^2).
        boff = (blade / self.arm_radius).reshape(B, self.P, 9)
        normal = blade_normal(blade)                                        # [B,P,3]
        tilt = tilt_angle(blade)                                            # [B,P]
        cov = cov_from_blades(blade, self.arm_radius)                       # [B,P,6]

        xy = xyz[..., :2]
        d = _min_image(xy[:, :, None, :] - xy[:, None, :, :], box)          # [B,P,P,2]
        # +eps under the sqrt: the diagonal is exactly 0 and sqrt'(0) = inf, so an
        # unmargined sqrt NaNs backward on EVERY sample, not just degenerate ones.
        dist = torch.sqrt((d ** 2).sum(-1) + 1e-12)                         # [B,P,P]

        off = self.off_diag.to(dt)
        mean_d, std_d = _masked_mean_std(dist, off, float(self.P - 1), 1e-6)
        # Finite sentinel rather than inf: inf survives a later reduction as
        # (inf - inf) = NaN, and it only has to beat the largest reachable min-image
        # distance (box*sqrt(2)/2 in xy).
        min_d = (dist + (1.0 - off) * (4.0 * box)).amin(-1)                 # [B,P]

        dflat = dist.reshape(B, 1, self.P * self.P)
        pm = self.pair_mask.to(dt)
        mean_pd, std_pd = _masked_mean_std(dflat, pm, self.n_pair.to(dt), 1e-6)
        min_pd = (dflat + (1.0 - pm) * (4.0 * box)).amin(-1)                # [B,E]
        # radians, not lengths: the variance guard scales with the quantity.
        mean_t, std_t = _masked_mean_std(tilt[:, None, :], H, kf, 1e-8)     # [B,E]

        # Same closed form the decoder recomputes the edge POSITION with, so the
        # feature and the target cannot drift apart.
        cxy = _circular_mean_xy(xy, self.Hmean.to(dt), box)                 # [B,E,2]
        dc = _min_image(xy[:, None] - cxy[:, :, None, :], box)              # [B,E,P,2]
        mean_cd = (torch.sqrt((dc ** 2).sum(-1) + 1e-12) * H).sum(-1) / kf  # [B,E]

        nf = torch.cat([trig, boff, normal, tilt[..., None], cov,
                        node_depth[..., None],
                        torch.stack([mean_d, min_d, std_d], -1) / box], -1)
        ef = torch.stack([(kf / self.P)[None].expand(B, -1),
                          mean_pd / box, min_pd / box, std_pd / box,
                          mean_t, std_t, hyper_depth, mean_cd / box], -1)
        return nf, ef

    def forward(self, coords, blade, node_depth, hyper_depth):
        nf, ef = self.geom_features(coords, blade, node_depth, hyper_depth)
        outs = self.trunk(self.in_node(nf), self.in_edge(ef))
        parts = []
        for node_h, edge_h in outs:
            parts.append(node_h.reshape(node_h.shape[0], -1))
            pooled = torch.einsum('ke,beh->bkh', self.kg.to(edge_h.dtype), edge_h)
            parts.append(pooled.reshape(edge_h.shape[0], -1))
        return torch.cat(parts, -1), outs[-1]


class HyperDecoder(nn.Module):
    """Latent (node xyz + edge xyz) plus the encoder's final embeddings -> a diagonal
    Gaussian over DELTAS. No absolute position is ever emitted.

    Edge xy is not predicted: a hyperedge's xy is the PBC circular mean of its
    members' xy, so it has zero degrees of freedom given the nodes (design
    Decision 1) -- predicting it freely would spend 2E outputs on a determined
    quantity and then need a consistency penalty to drag it back.

    It is also not PROPAGATED as a displacement. The design's `Hmean @ node_delta_xy`
    is the uniform-weight approximation to the circular mean's differential and is
    exact only at k=2; at k>=3 it misses 34-67% of the measured edge displacement
    (see _circular_mean_xy). Instead the edge POSITION is recomputed in closed form
    from the predicted node positions, and the displacement callers get is the
    min-image difference of the two positions -- exact at every k, with no drift to
    accumulate over a rollout.

    Edge DEPTH stays free: measured corr(d_hyper, mean d_member) is 0.060, so tying
    it to the members would impose a coupling the data does not have.

    Node DEPTH is free too, and that is a correction to the design, which says
    "7x3 + 120x1 = 141 free numbers" and thereby assumes a protein's z-coordinate IS
    its membrane node depth. It is not: node_depth is the membrane's depth sampled at
    the protein's location, coords[...,2] is the protein's own z, and they are in
    different units (coords box-scaled, depth renormalized to unit variance by the
    DataModule). Pooled over the 18682-frame 3-part sweep pool the per-protein
    delta corr(d_node_depth, d_protein_z) is 0.621 -- 38.5% of the depth-step
    variance explained, 61.5% not -- while the std ratio is 1.7101. A single scalar
    gain on the z step therefore matches the step VARIANCE the distributional losses
    score while getting 62% of the per-step signal wrong: the same level-vs-delta
    trap as the hyperedges (level corr 0.476, delta corr 0.060). So the node head
    emits 4 means and 4 log_sigmas and the free-parameter count is 7x4 + 120x1 = 148.
    """

    def __init__(self, hidden, layers, members, enc_hidden,
                 box=856.1959882301632, log_sigma_min=-8.0, log_sigma_max=4.0):
        super().__init__()
        self.trunk = AllSetTrunk(hidden, layers, members)
        self.P, self.E = self.trunk.P, self.trunk.E
        self.box = float(box)
        self.log_sigma_min, self.log_sigma_max = float(log_sigma_min), float(log_sigma_max)
        # _soft_bound needs a non-empty interval; an inverted/degenerate one would
        # divide by 0 (or silently flip the map) rather than fail.
        if not self.log_sigma_max > self.log_sigma_min:
            raise ValueError(f'log_sigma_max must exceed log_sigma_min, got '
                             f'{self.log_sigma_min} .. {self.log_sigma_max}')
        self.in_node = _mlp(3 + int(enc_hidden), hidden, hidden)
        self.in_edge = _mlp(3 + int(enc_hidden), hidden, hidden)
        # mu (dx, dy, dz, d_depth) then log_sigma for the same four. The depth column
        # is the node's OWN degree of freedom, not a rescaled dz -- see the class
        # docstring for the measurement that forced it.
        self.node_out = nn.Linear(hidden, 8)
        self.edge_out = nn.Linear(hidden, 2)          # mu_z,   log_sigma_z
        H = self.trunk.H
        self.register_buffer('Hmean', H / H.sum(1, keepdim=True).clamp_min(1.0),
                             persistent=False)

    def _sample(self, mu, log_sigma, eps):
        # Reparameterized, so the gradient reaches BOTH mu and log_sigma. Bound before
        # exp: a transient large log_sigma overflows to inf and NaNs the whole rollout,
        # and the floor stops sigma underflowing to a hard 0 (dead noise, dead
        # gradient). Soft rather than clamp so a saturated head can still recover --
        # see _soft_bound.
        sigma = torch.exp(_soft_bound(log_sigma, self.log_sigma_min, self.log_sigma_max))
        if eps is None:
            eps = torch.randn_like(mu)
        elif eps.shape != mu.shape:
            # Broadcasting here is silent and changes the OUTPUT WIDTH: eps [B,E,3]
            # against mu [B,E,1] makes forward return edge_delta [B,E,5], so a
            # downstream edge_delta[...,2] reads a noise channel, not the depth delta.
            raise ValueError(f'eps shape {tuple(eps.shape)} does not match mu shape '
                             f'{tuple(mu.shape)}: broadcasting would silently widen '
                             f'the sampled output')
        return mu + sigma * eps.to(mu.dtype)

    def edge_xy(self, node_xy):
        """Hyperedge positions [B,E,2] in [0,box) from protein positions [B,P,2].
        Exposed because a caller that keeps edge positions across a rollout must use
        this identical closed form rather than integrating the returned delta."""
        xy = _geom_dtype(node_xy)[..., :2]
        return _circular_mean_xy(xy, self.Hmean.to(xy.dtype), self.box)

    def forward(self, z_node, z_edge, enc_node_h, enc_edge_h, node_xy,
                eps_node=None, eps_edge=None, eps_depth=None):
        """-> (node_delta [B,P,3], edge_delta [B,E,3], node_depth_delta [B,P,1]).

        `node_xy` is the CURRENT protein position, [B,P,2] or [B,P,3] or [B,P*3]; the
        new one is node_xy + node_delta[...,:2]. Passing the node position (rather
        than the previous edge position) keeps one source of truth: both edge
        positions come from the same closed form, so their difference cannot drift
        away from the target's own definition.

        edge_delta[...,:2] is the min-image displacement between the recomputed edge
        positions -- a displacement for the caller's convenience, while the POSITION
        is what is computed exactly. edge_delta[...,2] is the predicted depth delta.

        node_depth_delta is a SEPARATE return rather than a fourth column of
        node_delta: callers index node_delta[...,:3] as a displacement and integrate
        it into positions, so widening it would broadcast a depth into the geometry
        exactly the way a mis-shaped eps does.

        eps_* override the sampling noise (tests, and any caller needing a
        reproducible draw); each must match its mu exactly -- eps_node [B,P,3],
        eps_depth [B,P,1], eps_edge [B,E,1]."""
        node_h, edge_h = self.trunk(
            self.in_node(torch.cat([z_node, enc_node_h], -1)),
            self.in_edge(torch.cat([z_edge, enc_edge_h], -1)))[-1]
        no, eo = self.node_out(node_h), self.edge_out(edge_h)
        node_delta = self._sample(no[..., :3], no[..., 4:7], eps_node)
        node_depth_delta = self._sample(no[..., 3:4], no[..., 7:8], eps_depth)
        edge_dz = self._sample(eo[..., :1], eo[..., 1:], eps_edge)

        xy0 = _geom_dtype(node_xy).reshape(node_delta.shape[0], self.P, -1)[..., :2]
        xy1 = xy0 + node_delta[..., :2].to(xy0.dtype)
        Hmean = self.Hmean.to(xy0.dtype)
        # min-image the difference of two positions that each live in [0,box): a step
        # across the periodic seam is a small move, not a box-sized one.
        edge_dxy = _min_image(_circular_mean_xy(xy1, Hmean, self.box)
                              - _circular_mean_xy(xy0, Hmean, self.box), self.box)
        # Cast the DEPTH up, never the geometry down. Under bf16 autocast edge_dz
        # comes back bf16 and edge_dxy fp32; casting edge_dxy to edge_dz's dtype threw
        # away the precision _geom_dtype exists to protect (measured median relative
        # error 1.4e-3, max abs 0.99 on the box/2 spikes) with the cast direction
        # decided by which head happened to be autocast.
        return (node_delta,
                torch.cat([edge_dxy, edge_dz.to(edge_dxy.dtype)], -1),
                node_depth_delta)
