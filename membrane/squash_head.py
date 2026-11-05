"""Radial squash head: contract a proposed step against its nearest neighbour.

WHY THIS SHAPE. Measured on GT (scripts/membrane/analysis/measure_contact_conditioned_steps.py,
results/membrane/evaluation/2026-07-13_rerank-eval-battery/contact_conditioned_gt.json, both trajectories agreeing):
as a pair closes to 1.05-1.15 core the mean RADIAL step is pushed outward
(+0.59 / +1.05) and the radial std drops ~25-32% (5.2 -> 3.6-3.9), while the
TANGENTIAL std does not move (3.2-3.5 at every separation). So excluded volume
acts along the separation axis and is not a general slowdown, and the effect has
decayed to zero by ~1.5 core. Three consequences, all baked in here:

  * work in the radial/tangential frame of the nearest neighbour, not in (x, y).
    A per-axis squash would scale the noise covariance to diag(s) L L^T diag(s),
    a directional distortion that lands in NGP and VACF -- both already weak.
  * leave the tangential component and z alone. ev_force and
    compute_excluded_volume_loss are both xy-only, so squashing z buys no EV.
  * contract only the INWARD (closing) radial component. Scaling both signs
    equally shrinks the spread but leaves the mean at zero; suppressing only the
    closing half reproduces BOTH measured effects at once. Headroom checks out:
    fully suppressing inward steps on sigma=5.2 gives a mean of sigma/sqrt(2pi)
    = 2.07 against the 0.59-1.05 GT asks for, so partial suppression suffices.

The head is deliberately a one-layer edge MLP over pairs plus a symmetric pool,
i.e. the minimal GNN. Attention would spend capacity deciding WHICH neighbour
matters; GT says it is the closest one, which max-pooling gives for free.

Bounded to [0, 1] by a sigmoid, and biased so it starts at s ~ 1 (identity). The
bound is what makes the module sound rather than merely discouraging: if the
current configuration is valid then x + s*ds -> x as s -> 0, and pair distance is
continuous, so some admissible s always restores validity. A signed multiplier
would break that -- reversing a component can drive a particle INTO a neighbour.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _min_image(d, box):
    return d - box * torch.round(d / box)


def neighbour_frame(xy, box):
    """Nearest-neighbour direction and distance per particle.

    xy:  [B, N, 2] wrapped positions.
    returns u [B, N, 2] unit vector i -> nearest j (min image), d [B, N, 1].
    """
    n = xy.size(1)
    diff = _min_image(xy.unsqueeze(1) - xy.unsqueeze(2), box)      # [B,N,N,2]  j - i
    dist = diff.norm(dim=-1)                                       # [B,N,N]
    eye = torch.eye(n, device=xy.device, dtype=torch.bool).unsqueeze(0)
    dist = dist.masked_fill(eye, float('inf'))
    dmin, arg = dist.min(dim=-1)                                   # [B,N]
    u = torch.gather(diff, 2, arg[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
    return u / dmin.unsqueeze(-1).clamp_min(1e-9), dmin.unsqueeze(-1)


class RadialSquashHead(nn.Module):
    """[B,N*3] proposed step -> same shape, inward radial component contracted.

    edge_hidden/hidden are small on purpose: the input is a distance and a signed
    projection, and the target decays to 1 beyond ~1.5 core.
    """

    def __init__(self, n_particles, box_size, latent_dim=0, edge_hidden=32,
                 hidden=32, pool='max', asymmetric=True, init_bias=6.0,
                 cutoff_cores=2.0, core_free=False, cutoff_frac=0.25):
        super().__init__()
        if pool not in ('max', 'sum', 'mean'):
            raise ValueError(f"pool must be max|sum|mean, got {pool!r}")
        self.n_particles = n_particles
        self.box_size = float(box_size)
        self.pool = pool
        self.asymmetric = bool(asymmetric)
        self.cutoff_cores = float(cutoff_cores)
        # core_free: normalise the edge features by the BOX rather than the hard core,
        # and use a generous fixed cutoff, so the head consults no GT-derived contact
        # constant at inference and has to learn where the wall is from data. The
        # core-normalised default conditions better but does read one scalar at
        # rollout, which matters when the whole point is that nothing analytic runs.
        self.core_free = bool(core_free)
        self.cutoff_frac = float(cutoff_frac)
        self.latent_dim = int(latent_dim)
        # edge features: d/core, (d/core)^-1 clamped, radial projection of i's step
        # onto u_ij scaled by core, and a closing indicator.
        self.edge = nn.Sequential(
            nn.Linear(4, edge_hidden), nn.SiLU(),
            nn.Linear(edge_hidden, edge_hidden), nn.SiLU())
        self.out = nn.Sequential(
            nn.Linear(edge_hidden + self.latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))
        # start at identity: sigmoid(6.0) = 0.9975, so the first epochs are
        # the frozen backbone's own dynamics. A fresh sigmoid would sit at 0.5
        # and halve every displacement before the head has learned anything.
        nn.init.zeros_(self.out[-1].weight)
        nn.init.constant_(self.out[-1].bias, init_bias)

    def squash(self, xy, ds_xy, core, latent=None):
        """s in [0,1] per particle. xy/ds_xy [B,N,2], core scalar tensor."""
        b, n, _ = xy.shape
        diff = _min_image(xy.unsqueeze(1) - xy.unsqueeze(2), self.box_size)  # j-i
        dist = diff.norm(dim=-1).clamp_min(1e-9)                             # [B,N,N]
        eye = torch.eye(n, device=xy.device, dtype=torch.bool).unsqueeze(0)
        u = diff / dist.unsqueeze(-1)
        radial = (ds_xy.unsqueeze(2) * u).sum(-1)        # [B,N,N] + = i moves toward j
        scale = self.box_size if self.core_free else core
        dn = (dist / scale).clamp(0.0, 20.0)
        feats = torch.stack([dn,
                             (1.0 / dn.clamp_min(0.02 if self.core_free else 0.25)),
                             radial / scale,
                             (radial > 0).to(xy.dtype)], dim=-1)
        h = self.edge(feats)                                                  # [B,N,N,H]
        # neighbours beyond the measured interaction range carry no signal
        cutoff = (self.cutoff_frac * self.box_size if self.core_free
                  else self.cutoff_cores * core)
        mask = eye | (dist > cutoff)
        if self.pool == 'max':
            h = h.masked_fill(mask.unsqueeze(-1), float('-inf'))
            h = h.max(dim=2).values
            h = torch.nan_to_num(h, neginf=0.0)
        else:
            h = h.masked_fill(mask.unsqueeze(-1), 0.0)
            h = h.sum(dim=2)
            if self.pool == 'mean':
                h = h / (~mask).sum(-1, keepdim=True).clamp_min(1).to(h.dtype)
        if self.latent_dim:
            h = torch.cat([h, latent], dim=-1)
        return torch.sigmoid(self.out(h)).squeeze(-1)                         # [B,N]

    def forward(self, x_flat, ds_flat, core, latent=None):
        """Apply the contraction. x_flat/ds_flat [B, N*3]; z is never touched."""
        b, n = x_flat.size(0), self.n_particles
        x = x_flat.view(b, n, 3)
        ds = ds_flat.view(b, n, 3)
        xy, ds_xy = x[..., :2], ds[..., :2]
        u, _ = neighbour_frame(xy, self.box_size)               # i -> nearest j
        s = self.squash(xy, ds_xy, core, latent)                # [B,N]
        rad = (ds_xy * u).sum(-1, keepdim=True)                 # + = closing
        tan = ds_xy - rad * u
        if self.asymmetric:
            # contract only the closing half: this is what moves the MEAN outward
            scale = torch.where(rad > 0, s.unsqueeze(-1), torch.ones_like(rad))
        else:
            scale = s.unsqueeze(-1)
        out = ds.clone()
        out[..., :2] = tan + (rad * scale) * u
        return out.reshape(b, n * 3), s


def minimal_contraction_target(x_flat, ds_flat, core, box_size, n_particles,
                               asymmetric=True):
    """Smallest s in [0,1] per particle that keeps every xy pair outside `core`.

    Dense supervision: defined at EVERY step and exactly 1.0 whenever the raw
    proposal was already fine, which is most steps -- so "stay near 1" is taught
    rather than hoped for, and the target is always inside the sigmoid's range.
    Needs `core` only at TRAINING time.

    Solved per pair. Scaling the closing radial component by s makes the next
    separation r + a + s*b with a the un-scaled part and b the scaled part, so
    |r + a + s b|^2 = core^2 is a quadratic in s; the admissible s is its smaller
    root clamped to [0,1]. A particle's target is the min over its violating
    pairs (the binding constraint), and 1.0 if it has none.
    """
    b_, n = x_flat.size(0), n_particles
    x = x_flat.view(b_, n, 3)[..., :2]
    ds = ds_flat.view(b_, n, 3)[..., :2]
    u, _ = neighbour_frame(x, box_size)
    rad = (ds * u).sum(-1, keepdim=True)
    tan = ds - rad * u
    if asymmetric:
        closing = (rad > 0).to(ds.dtype)
    else:
        closing = torch.ones_like(rad)
    fixed = tan + rad * (1.0 - closing) * u        # part s does not scale
    scaled = rad * closing * u                     # part s scales

    i, j = torch.triu_indices(n, n, offset=1, device=x.device)
    r = _min_image(x[:, i] - x[:, j], box_size)                 # [B,P,2]
    a = fixed[:, i] - fixed[:, j]
    bb = scaled[:, i] - scaled[:, j]
    p, q = r + a, bb
    A = (q * q).sum(-1)
    B = 2.0 * (p * q).sum(-1)
    C = (p * p).sum(-1) - core ** 2
    disc = B * B - 4 * A * C
    ok_at_one = (p + q).norm(dim=-1) >= core                    # no violation at s=1
    safe = (A > 1e-12) & (disc > 0) & ~ok_at_one
    root = (-B - torch.sqrt(disc.clamp_min(0))) / (2 * A.clamp_min(1e-12))
    s_pair = torch.where(safe, root.clamp(0.0, 1.0), torch.ones_like(root))

    tgt = x.new_ones(b_, n)
    tgt = tgt.scatter_reduce(1, i.expand(b_, -1), s_pair, reduce='amin')
    tgt = tgt.scatter_reduce(1, j.expand(b_, -1), s_pair, reduce='amin')
    return tgt.clamp(0.0, 1.0)
