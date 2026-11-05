"""Per-protein 3D shape/orientation features derived from the stored blade vectors.

The main model has always seen each protein as a point: sin/cos of its wrapped xyz
plus pairwise distances. These functions turn the 3 blade offsets into the shape and
orientation signal that theta (atan2 of xy only) throws away.
"""
import math

import torch

TRI = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]   # xx, xy, xz, yy, yz, zz

# Single source of truth for what each geom_arm implies. Every decision site keys
# off this table rather than re-testing the arm string, so adding an arm is one
# row and an unknown/typo'd arm is a construction-time error instead of a silent
# fall-through to the baseline under the arm's wandb group name.
#   needs_blade           -- reads `blade` in the AR loop (requires geom_blade_enabled)
#   extra_dim_per_particle-- features appended per protein to the encoder embedding
#   needs_arms            -- meaningless without arms_enabled (its code sits under it)
#   dof                   -- rotation-head output DOF (1 = theta, 3 = rotation vector)
#   advances_blade         -- AR loop rodrigues-advances blade_state every step (vs
#                             holding it at the IC for the whole rollout)
# moments and se3 now differ ONLY in the feature derived from the blade state --
# both predict orientation the same way (a 3-DOF rotation head advancing blade_state
# via rodrigues), so the comparison isolates the ENCODING.
#   needs_splat           -- feeds centres+blades through SplatEncoder and appends its
#                            GLOBAL embedding (config['splat_out_dim'] wide, NOT
#                            per-particle) to the encoder input every AR step
GEOM_ARMS = {
    'none':    dict(needs_blade=False, extra_dim_per_particle=0, needs_arms=False,
                    dof=1, advances_blade=False, needs_splat=False),
    'moments': dict(needs_blade=True,  extra_dim_per_particle=6, needs_arms=True,
                    dof=3, advances_blade=True,  needs_splat=False),
    'se3':     dict(needs_blade=True,  extra_dim_per_particle=0, needs_arms=True,
                    dof=3, advances_blade=True,  needs_splat=False),
    # Arm C. Same blade state and same 3-DOF rotation head as moments/se3, so the
    # comparison isolates the ENCODING exactly as it does for those two: the only
    # difference is that the blade state is rasterized to a density grid and read by
    # a CNN instead of being reduced to covariance moments or pairwise distances.
    'splat':   dict(needs_blade=True,  extra_dim_per_particle=0, needs_arms=True,
                    dof=3, advances_blade=True,  needs_splat=True),
}

# Derived alias kept for callers that only care about the blade question.
BLADE_ARMS = frozenset(k for k, v in GEOM_ARMS.items() if v['needs_blade'])


def flat_blade_template(n_particles, arm_radius, batch, device=None, dtype=torch.float32):
    """[batch,n_particles,3,3] flat symmetric 3-fold trimer: blades at 120deg
    spacing, radius arm_radius, dz=0.

    This is the exact 3D lift of theta=0 through base_model.reconstruct_arms, i.e.
    the same default the 1-DOF theta path already falls back to when no GT
    orientation is supplied. Blade-consuming arms use it when a caller (the
    stability callback, the eval scripts) has no GT blades to hand -- an unsupplied
    artifact gets a defined default here, never a silently dropped concat."""
    ang = 2 * math.pi * torch.arange(3, device=device, dtype=dtype) / 3
    off = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)],
                      dim=-1) * arm_radius                       # [3,3]
    return off.expand(batch, n_particles, 3, 3).contiguous()


def cov_from_blades(blade, arm_radius):
    """blade [...,P,3,3] (3 offsets per protein) -> [...,P,6] normalized covariance.

    The body is the 4 markers {centre, b0, b1, b2}; the centre offset is 0 by
    construction, so the covariance is translation-invariant with no extra work.
    Normalized by arm_radius^2 to land O(1) -- raw entries run ~1e2-1e3 in
    normalized length units and would swamp the sin/cos node features."""
    zero = blade.new_zeros(blade.shape[:-2] + (1, 3))
    pts = torch.cat([zero, blade], dim=-2)                    # [...,P,4,3]
    pts = pts - pts.mean(dim=-2, keepdim=True)
    cov = pts.transpose(-1, -2) @ pts / pts.shape[-2]         # [...,P,3,3]
    cov = cov / (arm_radius ** 2)
    return torch.stack([cov[..., i, j] for i, j in TRI], dim=-1)


def blade_normal(blade):
    """blade [...,P,3,3] -> [...,P,3] unit normal of the trimer plane.

    Built from two in-plane edge vectors so it is independent of where the centre
    sits; the sign is arbitrary (the trimer has no distinguished face), which is why
    every consumer takes |n . z| rather than n . z. An exactly collinear/zero-area
    blade has an undefined normal; the cross product is then the zero vector and
    clamp_min keeps the divide finite, so this silently returns the zero vector
    rather than a unit one -- not raised, since three offsets at fixed arm_radius
    going exactly collinear does not occur in the trimer geometry this is fit on.
    Downstream, tilt_angle/rotation_step_angle read atan2(0,0)=0 for that sample."""
    e0 = blade[..., 0, :] - blade[..., 2, :]
    e1 = blade[..., 1, :] - blade[..., 2, :]
    n = torch.cross(e0, e1, dim=-1)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def tilt_angle(blade):
    """blade [...,P,3,3] -> [...,P] tilt off the membrane normal, radians in [0, pi/2].

    Folded via |n_z| because the trimer is face-agnostic: a protein rotated 170 deg is
    physically tilted 10 deg, and treating it as 170 would put a spurious mode at the
    top of the distribution this loss matches. Computed as atan2(|n_xy|, |n_z|) rather
    than arccos(|n_z|): the two agree everywhere (exactly, since |n_xy| = sqrt(1-n_z^2)
    for a unit normal) so this is not a precision fix. It is a gradient fix -- arccos'(1)
    is -inf, and n_z = +-1 (a flat trimer, e.g. flat_blade_template's own output) is the
    common case at initialization, where d(n_z)/d(param) is also exactly 0, so unmargined
    that is 0 * -inf = NaN reaching backward(). atan2 is exact at that point and its
    gradient there is finite (0), which is the correct value: that sample is a flat
    kink (like torch.abs at 0), not a NaN source."""
    n = blade_normal(blade)
    return torch.atan2(n[..., :2].norm(dim=-1), n[..., 2].abs())


def rotation_step_angle(blade_t, blade_tp1):
    """Consecutive blade states -> [...,P] angle between their normals, radians.

    Same |.| folding as tilt_angle, for the same reason (a 170-deg step is a 10-deg
    step). Computed as atan2(|n_t x n_t+1|, |n_t . n_t+1|) rather than
    arccos(|n_t . n_t+1|), for the same gradient reason as tilt_angle: two identical
    consecutive blades (cos=1, the common case at initialization before a rotation head
    has learned anything) drive arccos' to -inf and NaN the gradient through the
    d(cos)/d(param)=0 kink; atan2 is exact there with a finite (0) gradient."""
    n0 = blade_normal(blade_t)
    n1 = blade_normal(blade_tp1)
    cross_mag = torch.cross(n0, n1, dim=-1).norm(dim=-1)
    dot_mag = (n0 * n1).sum(-1).abs()
    return torch.atan2(cross_mag, dot_mag)


def blade_theta(blade):
    """blade [...,P,3,3] -> [...,P] in-plane 3-fold orientation, radians in (-pi/3, pi/3].

    This is the SAME estimator that produced the stored `theta` field --
    scripts/membrane/data_prep/build_membrane_arms.py computes q = mean_k exp(3i*phi_k) over the three
    blade xy angles and takes angle(q)/3 -- so a 3-DOF arm's predicted blade state can
    be scored against GT with the observable the dataset already defines, instead of a
    second, differently-normalized notion of "spin".

    The tilt/rotation-step observables read only the trimer NORMAL and are therefore
    blind to rotation ABOUT that normal: measured on GT blades, spinning every protein
    45deg about its own normal moves this by 45.25deg while rotation_step_angle reads
    1.5e-6deg and tilt_angle moves 9.9e-7deg. This recovers exactly that blind DOF.

    It is a PROJECTED observable, so a pure tilt change leaks into it a little: a 2.6deg
    rotation about an in-plane axis (the mean GT normal step, zero spin) moves this by
    0.22deg against a GT spin step of 1.53deg mean. That cross-talk is acceptable
    because the identical estimator is applied to prediction and GT, so the two
    distributions being matched are the same functional of the blade state.

    exp(3i*phi) is formed algebraically as ((x+iy)/|v_xy|)^3 rather than via
    atan2(y, x): an arm pointing exactly along +-z has v_xy = 0, where atan2(0,0) is
    defined (0) but its gradient is 0/0 = NaN. clamp_min on |v_xy|^3 makes that sample
    contribute 0 with a finite gradient instead -- the same class of fix as
    tilt_angle's atan2 formulation.

    The outer atan2 is degenerate only if the three arms' third-harmonic phasors cancel
    exactly (|q| = 0). They do not for this geometry: over all 7103 GT frames
    |q| >= 0.746 (mean 0.973), because the rigid trimer keeps its arms ~120deg apart in
    projection for every tilt the bilayer permits (p99 30deg, max 38.4deg)."""
    x, y = blade[..., 0], blade[..., 1]                   # [...,P,3]
    r3 = (x * x + y * y).clamp_min(1e-12) ** 1.5          # |v_xy|^3
    c = (x ** 3 - 3 * x * y * y) / r3                     # cos(3 phi)
    s = (3 * x * x * y - y ** 3) / r3                     # sin(3 phi)
    return torch.atan2(s.mean(-1), c.mean(-1)) / 3.0


def rodrigues(v, w):
    """Rotate vectors v [...,3] by rotation vector w [...,3] (axis*angle).

    Used to advance blade orientation autoregressively: the 3-DOF rotation head
    emits w, and the whole rigid trimer turns by it. w is cast to v's dtype --
    torch.cross raises on mixed dtypes, and under autocast the head's output and
    the blade state need not agree (safe today only because the fp32 policy
    happens to promote the head)."""
    w = w.to(v.dtype)
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / th
    kv = (k * v).sum(-1, keepdim=True)
    return (v * torch.cos(th) + torch.cross(k.expand_as(v), v, dim=-1) * torch.sin(th)
            + k * kv * (1.0 - torch.cos(th)))


def sample_orientation(n_batch, n_particles, arm_radius,
                       tilt_mean_deg=15.50, tilt_std_deg=7.13, generator=None):
    """Random per-protein orientations matching the measured GT tilt distribution.

    Replaces the flat template as the blade-less IC. All-proteins-aligned-and-flat is
    a measure-zero configuration; this codebase already rejects its 2D form (see
    --init-theta random). Defaults are the measured GT moments over 7103 frames.

    Tilt is sampled from a folded normal so it stays in [0, pi/2] with no rejection
    loop; azimuth of the tilt direction and the in-plane spin are both uniform."""
    ang = 2 * math.pi * torch.arange(3, dtype=torch.float32) / 3
    tmpl = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros(3)], -1) * arm_radius
    b = tmpl.view(1, 1, 3, 3).repeat(n_batch, n_particles, 1, 1)

    shape = (n_batch, n_particles, 1)
    spin = torch.rand(shape, generator=generator) * (2 * math.pi)
    b = rodrigues(b, torch.cat([torch.zeros_like(spin), torch.zeros_like(spin), spin], -1)
                  .unsqueeze(2))

    tilt = (torch.randn(shape, generator=generator) * math.radians(tilt_std_deg)
            + math.radians(tilt_mean_deg)).abs().clamp(max=math.pi / 2)
    azi = torch.rand(shape, generator=generator) * (2 * math.pi)
    axis = torch.cat([torch.cos(azi), torch.sin(azi), torch.zeros_like(azi)], -1)
    return rodrigues(b, (axis * tilt).unsqueeze(2))


def _flat_orientation(n_batch, n_particles, arm_radius, generator=None):
    """sample_orientation's signature over flat_blade_template, so both IC modes are
    callable through one dispatch. Deterministic, so `generator` is accepted and
    ignored."""
    return flat_blade_template(n_particles, arm_radius, n_batch)


# The blade-less IC modes, keyed by the `blade_ic` config value. Single source of
# truth: base_model validates membership here AND dispatches through here, so the
# accepted set and the implemented set cannot drift apart (they were a literal tuple
# and an `== 'flat'` branch in two different places). Every entry takes
# (n_batch, n_particles, arm_radius, generator=None) -> [B,P,3,3].
BLADE_ICS = {
    'sampled': sample_orientation,
    'flat': _flat_orientation,
}
