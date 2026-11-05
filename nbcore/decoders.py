"""Decoders — map (latent z, context embedding) -> per-step displacement Delta_x.

All decoders share the interface:
    forward(z, context_embedding, active_mask=None, eps_prev=None)
        -> (delta_sample, delta_mu, noise_params, eps_t)
`eps_t` is the per-step output-noise component (delta_sample - delta_mu); it is
`None` unless AR(1) colored noise is active (decoder_ar1 != 'none'), in which case
it is the Markov state threaded to the next step as `eps_prev`.

`noise_type` selects the *instantaneous* (per-step) noise covariance structure
(the innovation L in eps_t = A.eps_{t-1} + L.eta_t):
  - 'cholesky' (default): per-particle 3x3 lower-triangular Cholesky, block-
    diagonal across particles. Params per particle: 6. Total params: 7*6=42.
  - 'diagonal': per-particle per-axis std, no cross-axis correlation. 3 per
    particle. Total: 21. Matches the pre-Cholesky gayso5zk-era decoder.
  - 'cholesky_full': single lower-triangular Cholesky over the full
    n_particles*3 = D dimension (cross-particle xyz correlated). Params:
    D*(D+1)/2. For D=21 that's 231 total noise params per timestep.

`decoder_ar1` adds an orthogonal TEMPORAL AR(1) coupling A on top of *any*
noise_type (output-space colored noise: eps_t = A.eps_{t-1} + L.eta_t). A is a
single GLOBAL learned parameter (constant across the rollout), constrained
contractive (largest singular value < 1) so the eps recursion is BIBO-stable
over long (40k-step) rollouts. Innovation form (no stationary-variance renorm):
at the target coupling (~0.17) the variance inflation 1/(1-a^2) ~ 3% is
negligible; magnitude is handled separately by the z-displacement loss.
  - 'none'  (default): temporally white (fresh randn each step).
  - 'diag'  : per-particle per-axis A. a = tanh(raw), raw [n_particles, 3].
  - 'block' : per-particle 3x3 A (cross-axis lag coupling). raw [n_particles,3,3].
  - 'full'  : single DxD A (cross-particle lag coupling). raw [D, D].
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P


# =========================================================================
#  Noise samplers
# =========================================================================

_NOISE_EPS = 1e-3     # floor on per-axis scale to keep sampling strictly PD
_PER_PARTICLE_CHOL_PARAMS = 6
_PER_PARTICLE_DIAG_PARAMS = 3
_VALID_NOISE_TYPES = ('cholesky', 'diagonal', 'cholesky_full')


def _full_cholesky_param_count(n_particles):
    D = n_particles * 3
    return D * (D + 1) // 2


def _noise_param_count(noise_type, n_particles):
    """Total noise param count emitted by the decoder noise head (not per-particle)."""
    if noise_type == 'cholesky':
        return n_particles * _PER_PARTICLE_CHOL_PARAMS
    if noise_type == 'diagonal':
        return n_particles * _PER_PARTICLE_DIAG_PARAMS
    if noise_type == 'cholesky_full':
        return _full_cholesky_param_count(n_particles)
    raise ValueError(f"Unknown noise_type={noise_type!r}; expected one of {_VALID_NOISE_TYPES}")


def _per_particle_param_count(noise_type):
    """Legacy helper; only valid for block-diagonal types."""
    if noise_type == 'cholesky':
        return _PER_PARTICLE_CHOL_PARAMS
    if noise_type == 'diagonal':
        return _PER_PARTICLE_DIAG_PARAMS
    raise ValueError(
        f"_per_particle_param_count not defined for noise_type={noise_type!r}; "
        f"use _noise_param_count instead.")


# ---- Innovation samplers: return the noise component L@eta only (not + delta_mu).
# These are the building blocks shared by the white path (_sample_noise) and the
# AR(1) path (_sample_noise_ar1). `ref` supplies batch size / device / dtype.

def _per_particle_cholesky_noise(chol_params, n_particles, ref):
    """blockdiag(L_i) @ eps, eps ~ N(0, I). chol_params: [B, n_particles*6]."""
    B = ref.size(0)
    params = chol_params.reshape(B, n_particles, _PER_PARTICLE_CHOL_PARAMS)
    d11 = F.softplus(params[..., 0]) + _NOISE_EPS
    d22 = F.softplus(params[..., 1]) + _NOISE_EPS
    d33 = F.softplus(params[..., 2]) + _NOISE_EPS
    o21 = params[..., 3]
    o31 = params[..., 4]
    o32 = params[..., 5]

    eps = torch.randn(B, n_particles, 3, device=ref.device, dtype=ref.dtype)
    # L @ eps, where L is 3x3 lower-triangular per particle.
    # Computed as per-row dot products without materializing the [B, N, 3, 3] tensor.
    e0 = eps[..., 0]
    e1 = eps[..., 1]
    e2 = eps[..., 2]
    n0 = d11 * e0
    n1 = o21 * e0 + d22 * e1
    n2 = o31 * e0 + o32 * e1 + d33 * e2
    return torch.stack([n0, n1, n2], dim=-1).reshape(B, n_particles * 3)


def _per_particle_diagonal_noise(diag_params, n_particles, ref):
    """sigma * eps, sigma per-particle per-axis. diag_params: [B, n_particles*3]."""
    sigma = F.softplus(diag_params) + _NOISE_EPS
    eps = torch.randn_like(ref)
    return sigma * eps


def _full_cholesky_noise(chol_params, n_particles, ref):
    """L @ eps, L a single lower-triangular over D=n_particles*3 (cross-particle)."""
    B = ref.size(0)
    D = n_particles * 3
    diag_raw = chol_params[:, :D]
    off_raw = chol_params[:, D:]  # [B, D*(D-1)/2]
    diag = F.softplus(diag_raw) + _NOISE_EPS  # [B, D]

    tril_idx = torch.tril_indices(D, D, offset=-1, device=ref.device)
    # Build L (and the matmul) in fp32. Under bf16 autocast `diag`/`off_raw`
    # are fp32 (softplus promotes), so an in-place index-put into a bf16 L
    # would dtype-mismatch; fp32 also keeps the full DxD covariance stable.
    ar = torch.arange(D, device=ref.device)
    L = torch.zeros(B, D, D, device=ref.device, dtype=torch.float32)
    L[:, ar, ar] = diag.float()
    L[:, tril_idx[0], tril_idx[1]] = off_raw.float()

    eps = torch.randn(B, D, device=ref.device, dtype=torch.float32)
    return torch.einsum('bij,bj->bi', L, eps)


def _innovation_noise(noise_type, noise_params, n_particles, ref):
    """The instantaneous noise component L@eta for the given noise_type."""
    if noise_type == 'cholesky':
        return _per_particle_cholesky_noise(noise_params, n_particles, ref)
    if noise_type == 'diagonal':
        return _per_particle_diagonal_noise(noise_params, n_particles, ref)
    if noise_type == 'cholesky_full':
        return _full_cholesky_noise(noise_params, n_particles, ref)
    raise ValueError(f"Unknown noise_type={noise_type!r}; expected one of {_VALID_NOISE_TYPES}")


def _sample_noise(noise_type, delta_mu, noise_params, n_particles):
    """White (temporally uncorrelated) path: delta_mu + L@eta."""
    return delta_mu + _innovation_noise(noise_type, noise_params, n_particles, delta_mu)


# =========================================================================
#  AR(1) temporal coupling (output-space colored noise)
# =========================================================================

_VALID_AR1 = ('none', 'diag', 'block', 'full')


class AR1Coupling(nn.Module):
    """Global, contractive lag-1 coupling A for output-noise AR(1).

    eps_t = A . eps_{t-1} + innovation.  A is one learned parameter set (constant
    across the rollout), constrained so its largest singular value is < 1, which
    makes the eps recursion BIBO-stable (Var(eps) -> Var(innov)/(1-||A||^2),
    bounded) over arbitrarily long rollouts.

    structure:
      'diag'  : per-particle per-axis. a = SPECTRAL_CAP*tanh(raw), raw
                [n_particles, 3]. Each entry IS a singular value => |a| <= cap < 1.
      'block' : per-particle 3x3. A_i = clip of raw_i to the spectral ball of
                radius SPECTRAL_CAP (A_i = raw_i while ||raw_i||_2 <= cap, else
                cap*raw_i/||raw_i||_2). Cross-axis lag coupling.
      'full'  : single DxD, same spectral-ball clip. Cross-particle lag coupling.
    diag zero-inits cleanly (tanh smooth at 0); block/full take a tiny nonzero
    init so the spectral-norm (SVD) backward is well-defined (SVD of an exact-zero
    matrix is singular) — A still ~0, so the on-ramp is effectively white. The
    clip (vs. always-normalize) is what gives the white on-ramp: ||A|| grows from
    ~0 with training and only saturates at cap. Realized A and all arithmetic are
    fp32 (the ea76c8f bf16 discipline).
    """
    SPECTRAL_CAP = 0.95

    def __init__(self, structure, n_particles):
        super().__init__()
        if structure not in ('diag', 'block', 'full'):
            raise ValueError(f"AR1 structure={structure!r}; expected diag|block|full")
        self.structure = structure
        self.n_particles = n_particles
        self.D = n_particles * 3
        if structure == 'diag':
            self.raw = nn.Parameter(torch.zeros(n_particles, 3))
        else:
            shape = (n_particles, 3, 3) if structure == 'block' else (self.D, self.D)
            self.raw = nn.Parameter(torch.zeros(shape))
            with torch.no_grad():
                self.raw.normal_(0.0, 1e-2)  # avoid SVD-at-zero singularity; A~0

    def matrix(self):
        """Realized contractive A in fp32: [N,3] (diag) | [N,3,3] (block) | [D,D] (full)."""
        cap = self.SPECTRAL_CAP
        if self.structure == 'diag':
            # cap*tanh keeps |a| <= cap < 1 strictly (plain tanh saturates to
            # exactly 1.0 in fp32 for large raw -> marginally stable over 40k).
            return cap * torch.tanh(self.raw.float())
        ah = self.raw.float()
        sn = torch.linalg.matrix_norm(ah, ord=2)             # [] (full) | [N] (block)
        scale = cap / sn.clamp_min(cap)                      # ==1 while ||ah||<=cap
        if self.structure == 'block':
            scale = scale.unsqueeze(-1).unsqueeze(-1)         # [N,1,1]
        return ah * scale

    def apply_to(self, eps_prev):
        """A . eps_prev. eps_prev: [B, D] (any dtype). Returns [B, D] fp32."""
        B = eps_prev.size(0)
        ep = eps_prev.float()
        if self.structure == 'diag':
            a = self.matrix()  # [N,3]
            return (ep.reshape(B, self.n_particles, 3) * a.unsqueeze(0)).reshape(B, self.D)
        if self.structure == 'block':
            A = self.matrix()  # [N,3,3]
            ep3 = ep.reshape(B, self.n_particles, 3)
            return torch.einsum('nij,bnj->bni', A, ep3).reshape(B, self.D)
        A = self.matrix()  # [D,D]
        return torch.einsum('ij,bj->bi', A, ep)

    @torch.no_grad()
    def spectral_radius(self):
        """Largest singular value of the realized A (diagnostics/tests)."""
        if self.structure == 'diag':
            return self.matrix().abs().max()
        A = self.matrix()
        sn = torch.linalg.matrix_norm(A, ord=2)
        return sn.max() if self.structure == 'block' else sn


def build_ar1(decoder_ar1, n_particles):
    """Return an AR1Coupling for decoder_ar1 in {diag,block,full}, or None for 'none'."""
    if decoder_ar1 in (None, 'none'):
        return None
    if decoder_ar1 not in _VALID_AR1:
        raise ValueError(f"decoder_ar1={decoder_ar1!r}; expected one of {_VALID_AR1}")
    return AR1Coupling(decoder_ar1, n_particles)


def _sample_noise_ar1(noise_type, delta_mu, noise_params, n_particles, ar1, eps_prev):
    """Colored path: eps_t = A.eps_{t-1} + L.eta_t; returns (delta_sample, eps_t).

    eps_t is kept in fp32 and threaded as the Markov state (constant dtype across
    steps avoids torch.compile recompiles on a dtype flip). delta_sample follows
    delta_mu's dtype.
    """
    innovation = _innovation_noise(noise_type, noise_params, n_particles, delta_mu).float()
    eps_t = ar1.apply_to(eps_prev) + innovation                       # [B, D] fp32
    delta_sample = delta_mu + eps_t.to(delta_mu.dtype)
    return delta_sample, eps_t


def _decoder_finalize(noise_type, ar1, delta_mu, noise_params, n_particles,
                      active_mask, eps_prev):
    """Shared decoder tail: draw the noise (white or AR1) and apply active_mask.

    Returns (delta_sample, delta_mu, noise_params, eps_t). eps_t is None unless
    AR(1) is active, in which case it is the fp32 Markov state for the next step.
    """
    if ar1 is not None:
        if eps_prev is None:
            eps_prev = torch.zeros(delta_mu.size(0), n_particles * 3,
                                   device=delta_mu.device)
        delta_sample, eps_t = _sample_noise_ar1(
            noise_type, delta_mu, noise_params, n_particles, ar1, eps_prev)
    else:
        delta_sample = _sample_noise(noise_type, delta_mu, noise_params, n_particles)
        eps_t = None
    if active_mask is not None:
        slot_mask = active_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(
            -1, n_particles * 3).float()
        delta_sample = delta_sample * slot_mask
        delta_mu = delta_mu * slot_mask
        if eps_t is not None:
            eps_t = eps_t * slot_mask
    return delta_sample, delta_mu, noise_params, eps_t


# =========================================================================
#  MLP decoder
# =========================================================================

class MLPDecoder(nn.Module):
    def __init__(self, gat_embed_dim, hidden_dim=512, latent_dim=21,
                 dropout=0.0, num_layers=3, n_coords=21, n_particles=7,
                 noise_type='cholesky', use_context=True, decoder_ar1='none'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_particles = n_particles
        self.noise_type = noise_type
        self.use_context = use_context
        self.ar1 = build_ar1(decoder_ar1, n_particles)
        n_noise_total = _noise_param_count(noise_type, n_particles)

        if use_context:
            self.context_norm = nn.LayerNorm(gat_embed_dim)
            self.latent_proj = nn.Sequential(
                P.spectral_norm(nn.Linear(latent_dim, hidden_dim // 2)), nn.SiLU())
            self.context_proj = nn.Linear(gat_embed_dim, hidden_dim // 2)
            input_dim = (hidden_dim // 2) * 2
        else:
            self.latent_proj = nn.Sequential(
                P.spectral_norm(nn.Linear(latent_dim, hidden_dim)), nn.SiLU())
            input_dim = hidden_dim

        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.extend([
                P.spectral_norm(nn.Linear(in_d, hidden_dim)),
                nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            ])
        self.trunk = nn.Sequential(*layers)

        self.mu_head = nn.Sequential(
            P.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.SiLU(), nn.Linear(hidden_dim // 2, n_coords))
        self.noise_head = nn.Sequential(
            P.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_noise_total))

    def forward(self, z, context_embedding, active_mask=None, eps_prev=None):
        z_feat = self.latent_proj(z)
        if self.use_context:
            ctx = context_embedding.squeeze(1) if context_embedding.dim() == 3 else context_embedding
            ctx_feat = self.context_proj(self.context_norm(ctx))
            x = self.trunk(torch.cat([z_feat, ctx_feat], dim=-1))
        else:
            x = self.trunk(z_feat)

        delta_mu = self.mu_head(x)
        noise_params = self.noise_head(x)
        return _decoder_finalize(self.noise_type, self.ar1, delta_mu, noise_params,
                                 self.n_particles, active_mask, eps_prev)


# =========================================================================
#  Cross-attention (transformer) decoder
# =========================================================================

class XFormerDecoder(nn.Module):
    """DEGENERATE ATTENTION -- kept only so existing checkpoints reproduce.

    This builds a ONE-token query and a ONE-token key/value, and softmax over a single
    key is identically 1.0. MEASURED on this class: cross-attention weights are exactly
    1.0 in every layer, and two completely different queries against the same key/value
    give BIT-IDENTICAL output (max |diff| = 0.0). The query is ignored. The self-attn
    inside TransformerDecoderLayer is equally inert -- a length-1 target attending to
    itself.

    So this is an MLP: z -> linear, plus a fixed linear projection of the context
    through the residual stream, plus the FFN. In the shipped 7piezo diffusion config
    that is 526,336 parameters (8.4% of the 6.29M model, 52.9% of the decoder) computing
    nothing. The docstring used to flag the degeneracy only for use_context=False; it is
    equally degenerate with context ON.

    DO NOT USE FOR NEW RUNS -- use ParticleXFormerDecoder ('xformer_particles'), which
    gives the attention real tokens. This class stays because decoder_type is frozen in
    checkpoint hyper_parameters and old checkpoints must still load.
    """

    def __init__(self, gat_embed_dim, hidden_dim=512, latent_dim=21,
                 num_layers=2, num_heads=8, dim_feedforward=None,
                 dropout=0.0, n_coords=21, n_particles=7,
                 noise_type='cholesky', use_context=True, decoder_ar1='none'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_particles = n_particles
        self.noise_type = noise_type
        self.use_context = use_context
        self.ar1 = build_ar1(decoder_ar1, n_particles)
        n_noise_total = _noise_param_count(noise_type, n_particles)
        dff = dim_feedforward if dim_feedforward is not None else hidden_dim * 4

        self.z_proj = nn.Sequential(
            P.spectral_norm(nn.Linear(latent_dim, hidden_dim)), nn.SiLU())
        if use_context:
            self.ctx_proj = nn.Sequential(
                nn.LayerNorm(gat_embed_dim),
                nn.Linear(gat_embed_dim, hidden_dim))
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=dff,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        else:
            self.ctx_proj = None
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=dff,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln_out = nn.LayerNorm(hidden_dim)

        self.mu_head = nn.Sequential(
            P.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.SiLU(), nn.Linear(hidden_dim // 2, n_coords))
        self.noise_head = nn.Sequential(
            P.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_noise_total))

    def forward(self, z, context_embedding, active_mask=None, eps_prev=None):
        z_hidden = self.z_proj(z).unsqueeze(1)          # [B, 1, H]
        if self.use_context:
            ctx = context_embedding.squeeze(1) if context_embedding.dim() == 3 else context_embedding
            ctx_hidden = self.ctx_proj(ctx).unsqueeze(1)    # [B, 1, H] key/value
            out = self.transformer(z_hidden, ctx_hidden)
        else:
            out = self.transformer(z_hidden)
        out = self.ln_out(out.squeeze(1))

        delta_mu = self.mu_head(out)
        noise_params = self.noise_head(out)
        return _decoder_finalize(self.noise_type, self.ar1, delta_mu, noise_params,
                                 self.n_particles, active_mask, eps_prev)



class ParticleXFormerDecoder(nn.Module):
    """Cross-attention that actually attends: one token PER PARTICLE.

    The bug this replaces: XFormerDecoder gave the attention a single query and a
    single key/value, so softmax ran over one element and returned 1.0 -- the query was
    provably ignored. Attention needs more than one token on at least one side.

    The tokens were already there, just flattened. GAT_GeometryEmbedding returns
    cat([gat, pw]) where gat is [B, S, P*hidden] PARTICLE-CONTIGUOUS and pw is the
    n_pairs upper-triangle of normalized distances, so:

        key/value = P per-particle geometry tokens  +  1 token for the pairwise block
        query     = P per-particle latent tokens    (latent_dim split P ways)

    NO positional or identity embedding is added, deliberately. Without one the decoder
    is permutation-EQUIVARIANT over particles: permuting the particles permutes the
    outputs identically. That matches the physics and matches the pooled-quantile losses,
    which are invariant to relabelling anyway.

    Heads are per-token where the noise parameterization allows it. 'cholesky' and
    'diagonal' are BLOCK-DIAGONAL per particle (_noise_param_count = n_particles * per),
    so each token emits its own 3 coordinates and its own noise block. 'cholesky_full'
    couples every particle to every other, so it cannot be produced per token and gets a
    pooled head over the concatenated tokens instead.
    """

    def __init__(self, gat_embed_dim, hidden_dim=512, latent_dim=42,
                 num_layers=2, num_heads=8, dim_feedforward=None,
                 dropout=0.0, n_coords=21, n_particles=7,
                 noise_type='cholesky', use_context=True, decoder_ar1='none',
                 latent_skip=False):
        super().__init__()
        if latent_dim % n_particles:
            raise ValueError(
                f"ParticleXFormerDecoder splits the latent into one token per particle, "
                f"so latent_dim ({latent_dim}) must divide by n_particles "
                f"({n_particles}). Got remainder {latent_dim % n_particles}.")
        if n_coords != n_particles * 3:
            raise ValueError(
                f"n_coords ({n_coords}) must be n_particles*3 ({n_particles * 3}).")
        self.hidden_dim = hidden_dim
        self.n_particles = n_particles
        self.n_coords = n_coords
        self.noise_type = noise_type
        self.use_context = use_context
        self.ar1 = build_ar1(decoder_ar1, n_particles)
        self.z_per = latent_dim // n_particles
        self.n_pairs = n_particles * (n_particles - 1) // 2
        # gat_embed_dim is geo_embed.output_dim = P*gat_hidden + n_pairs
        gat_hidden = (gat_embed_dim - self.n_pairs) // n_particles
        if gat_hidden * n_particles + self.n_pairs != gat_embed_dim:
            raise ValueError(
                f"context width {gat_embed_dim} is not P*h + n_pairs for P="
                f"{n_particles}, n_pairs={self.n_pairs}; cannot split it per particle.")
        self.gat_hidden = gat_hidden
        dff = dim_feedforward if dim_feedforward is not None else hidden_dim * 4

        self.q_proj = nn.Sequential(
            P.spectral_norm(nn.Linear(self.z_per, hidden_dim)), nn.SiLU())
        if use_context:
            self.kv_particle = nn.Sequential(
                nn.LayerNorm(gat_hidden), nn.Linear(gat_hidden, hidden_dim))
            self.kv_pair = nn.Sequential(
                nn.LayerNorm(self.n_pairs), nn.Linear(self.n_pairs, hidden_dim))
            layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=dff,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        else:
            self.kv_particle = self.kv_pair = None
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=dff,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.ln_out = nn.LayerNorm(hidden_dim)
        self.mu_head = nn.Sequential(
            P.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.SiLU(), nn.Linear(hidden_dim // 2, 3))
        # Start at zero drift: the AR loop wraps x + delta, so a biased init walks every
        # particle off in a fixed direction before training can correct it. BIAS ONLY,
        # matching TokenizedXFormerDecoder -- zeroing the weight too makes delta_mu
        # identically 0 for any input, which also sends zero gradient back through the
        # transformer on the first step.
        nn.init.zeros_(self.mu_head[-1].bias)

        self._per_token_noise = noise_type in ('cholesky', 'diagonal')
        n_noise_total = _noise_param_count(noise_type, n_particles)
        if self._per_token_noise:
            self.noise_head = nn.Sequential(
                P.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
                nn.SiLU(), nn.Linear(hidden_dim // 2, n_noise_total // n_particles))
        else:
            self.noise_head = nn.Sequential(
                P.spectral_norm(nn.Linear(hidden_dim * n_particles, hidden_dim)),
                nn.SiLU(), nn.Linear(hidden_dim, n_noise_total))

        # Additive latent -> mean path. Without it the only route from z to delta_mu is
        # the attention QUERY: z sets softmax(q.k) over geometry VALUES, so latent noise
        # can only reshuffle which geometry is mixed, never contribute a drift of its own.
        # Training then switches the OU latent off (decay and innov -> 0 at the incumbent
        # LR), which is what killed step-to-step persistence on 7piezo. This restores the
        # full-vector additive access the old single-token decoder had. Zero-initialised,
        # so step 0 is identical to the plain decoder; built last, so every other module
        # initialises exactly as it does without the flag.
        self.z_skip = None
        if latent_skip:
            self.z_skip = nn.Linear(latent_dim, n_coords)
            nn.init.zeros_(self.z_skip.weight)
            nn.init.zeros_(self.z_skip.bias)

    def forward(self, z, context_embedding, active_mask=None, eps_prev=None):
        B = z.size(0)
        q = self.q_proj(z.view(B, self.n_particles, self.z_per))      # [B, P, H]
        if self.use_context:
            ctx = (context_embedding.squeeze(1) if context_embedding.dim() == 3
                   else context_embedding)
            gat = ctx[..., :self.n_particles * self.gat_hidden].view(
                B, self.n_particles, self.gat_hidden)
            pair = ctx[..., self.n_particles * self.gat_hidden:]
            kv = torch.cat([self.kv_particle(gat),
                            self.kv_pair(pair).unsqueeze(1)], dim=1)   # [B, P+1, H]
            out = self.transformer(q, kv)
        else:
            out = self.transformer(q)
        out = self.ln_out(out)                                         # [B, P, H]

        delta_mu = self.mu_head(out).reshape(B, self.n_coords)
        if self.z_skip is not None:
            delta_mu = delta_mu + self.z_skip(z)
        if self._per_token_noise:
            noise_params = self.noise_head(out).reshape(B, -1)
        else:
            noise_params = self.noise_head(out.reshape(B, -1))
        return _decoder_finalize(self.noise_type, self.ar1, delta_mu, noise_params,
                                 self.n_particles, active_mask, eps_prev)


class GeomQueryDecoder(nn.Module):
    """Cross-attention with the roles swapped: geometry decides WHO couples, the latent
    supplies WHAT moves.

    ParticleXFormerDecoder puts z in the attention QUERY, so z sets softmax(q.k) over
    geometry-only values and latent noise can only reshuffle which geometry is mixed.
    Training answers by switching the OU latent off. Here queries and keys come from the
    per-particle geometry tokens and the VALUES come from the latent, so z never
    touches the weights and enters the drift additively.

        alpha_i = softmax over {other particles, pair token} of q(g_i).k(g_j)

    Self is masked out and particle i's own latent is added directly: all seven 7piezo
    particles are one species, so their geometry tokens are close and an unmasked
    softmax would spread toward uniform, dragging every particle toward the group mean.
    The pair token carries a ZERO value -- attention mass on it means "borrow nothing".

    values='proj': latent tokens go through a spectral Linear+SiLU (the query
        projection of ParticleXFormerDecoder), then the usual residual stream seeded
        with the geometry token, FFN, LayerNorm and per-token heads.
        drift_ln=False feeds mu_head the raw stream instead of ln_out(stream); the
        noise head keeps the LayerNorm. ln_out is scale-invariant, and the projected
        latent dominates the stream, so with it the drift sees z's direction but not its
        size (measured: corr(|z_i|, |mu_i|) = -0.09) -- a Gaussian latent then gives a
        drift of too-uniform size and sub-Gaussian displacements (NGP < 0, lags 5-10).
        latent_skip adds a zero-initialised Linear(z) -> delta_mu, linear in z.
    values='raw': no projection. The latent's 6 per-particle dims are read as
        [drift xyz | log-diffusivity xyz]:
            mu_i = s * z_i[:3] + sum_h c_h * sum_j alpha^h_ij z_j[:3]
            ds_i = mu_i + exp(k * z_i[3:]) * (L_i eps)
        s, c_h, k are per axis with x and y tied (lateral isotropy). A coordinate only
        ever drives the same coordinate, mu(-z) = -mu(z), and the latent's magnitude
        reaches the drift. c and k start at 0, so step 0 is independent OU-drift
        particles with the geometry noise head. L_i comes from the geometry token alone.
        The scalars are named _dyn_* so configure_optimizers puts them in the
        dynamics group: under SophiaG's per-step cap of lr, a scalar at the base LR can
        move only ~lr*steps over a whole run, i.e. it would stay at init.

    Both variants use ONE attention layer: a second layer's queries would come from
    a stream that already contains z.
    """

    def __init__(self, gat_embed_dim, hidden_dim=512, latent_dim=42, num_layers=1,
                 num_heads=8, dim_feedforward=None, dropout=0.0, n_coords=21,
                 n_particles=7, noise_type='cholesky', use_context=True,
                 decoder_ar1='none', values='proj', raw_gain_init=0.1, drift_ln=True,
                 latent_skip=False, species_groups=None):
        super().__init__()
        if values not in ('proj', 'raw'):
            raise ValueError(f"values={values!r}; expected 'proj' or 'raw'")
        # species_groups (>= 2 groups): the drift and noise heads are built once per
        # species. Shared heads make the decoder equivariant under ANY particle
        # permutation -- same latent and geometry in, same step out, whichever protein it
        # is -- so two species can differ only as far as their inputs do. Attention and the
        # residual stream stay shared; inside a species particles remain exchangeable.
        self.species_groups = None
        if species_groups is not None and len(species_groups) > 1:
            if values != 'proj' or noise_type not in ('cholesky', 'diagonal'):
                raise ValueError("per-species heads need values='proj' and a per-particle "
                                 "noise type ('cholesky' or 'diagonal')")
            self.species_groups = [list(g) for g in species_groups]
            order = [p for g in self.species_groups for p in g]
            if sorted(order) != list(range(n_particles)):
                raise ValueError(f"species_groups must partition range({n_particles}); got {species_groups}")
            self.register_buffer('_unsort', torch.argsort(torch.tensor(order)), persistent=False)
        if not use_context:
            raise ValueError("GeomQueryDecoder takes its queries and keys from the "
                             "geometry; it cannot run with dec_use_context=False")
        if num_layers != 1:
            raise ValueError(f"GeomQueryDecoder has exactly one attention layer (a "
                             f"second would query a stream containing z); got "
                             f"num_layers={num_layers}")
        if latent_dim % n_particles:
            raise ValueError(f"latent_dim ({latent_dim}) must divide by n_particles "
                             f"({n_particles}).")
        if n_coords != n_particles * 3:
            raise ValueError(f"n_coords ({n_coords}) must be n_particles*3.")
        if hidden_dim % num_heads:
            raise ValueError(f"hidden_dim ({hidden_dim}) must divide by num_heads "
                             f"({num_heads}).")
        self.values = values
        self.hidden_dim = hidden_dim
        self.n_particles = n_particles
        self.n_coords = n_coords
        self.noise_type = noise_type
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.ar1 = build_ar1(decoder_ar1, n_particles)
        self.z_per = latent_dim // n_particles
        if values == 'raw':
            if self.z_per != 6:
                raise ValueError(f"values='raw' reads each particle's latent as 3 drift + "
                                 f"3 log-diffusivity dims; needs latent_dim = 6*n_particles, "
                                 f"got {self.z_per} per particle")
            if self.ar1 is not None:
                raise ValueError("values='raw' scales the white innovation directly; "
                                 "decoder_ar1 must be 'none'")
        self.n_pairs = n_particles * (n_particles - 1) // 2
        gat_hidden = (gat_embed_dim - self.n_pairs) // n_particles
        if gat_hidden * n_particles + self.n_pairs != gat_embed_dim:
            raise ValueError(
                f"context width {gat_embed_dim} is not P*h + n_pairs for P="
                f"{n_particles}, n_pairs={self.n_pairs}; cannot split it per particle.")
        self.gat_hidden = gat_hidden
        dff = dim_feedforward if dim_feedforward is not None else hidden_dim * 4

        self.kv_particle = nn.Sequential(
            nn.LayerNorm(gat_hidden), nn.Linear(gat_hidden, hidden_dim))
        self.kv_pair = nn.Sequential(
            nn.LayerNorm(self.n_pairs), nn.Linear(self.n_pairs, hidden_dim))
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.q_lin = nn.Linear(hidden_dim, hidden_dim)
        self.k_lin = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        # [P, P+1]: particle i may not attend to its own key; the pair token is always open
        self.register_buffer('self_mask', torch.cat(
            [torch.eye(n_particles, dtype=torch.bool),
             torch.zeros(n_particles, 1, dtype=torch.bool)], dim=1), persistent=False)

        self.ln_out = nn.LayerNorm(hidden_dim)
        n_noise_total = _noise_param_count(noise_type, n_particles)
        self._per_token_noise = noise_type in ('cholesky', 'diagonal')

        def _noise_head():
            return nn.Sequential(
                P.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
                nn.SiLU(), nn.Linear(hidden_dim // 2, n_noise_total // n_particles))

        def _mu_head():
            head = nn.Sequential(
                P.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
                nn.SiLU(), nn.Linear(hidden_dim // 2, 3))
            nn.init.zeros_(head[-1].bias)
            return head

        if self.species_groups is not None:
            self.noise_heads = nn.ModuleList([_noise_head() for _ in self.species_groups])
        elif self._per_token_noise:
            self.noise_head = nn.Sequential(
                P.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
                nn.SiLU(), nn.Linear(hidden_dim // 2, n_noise_total // n_particles))
        else:
            self.noise_head = nn.Sequential(
                P.spectral_norm(nn.Linear(hidden_dim * n_particles, hidden_dim)),
                nn.SiLU(), nn.Linear(hidden_dim, n_noise_total))

        if values == 'proj':
            self.v_proj = nn.Sequential(
                P.spectral_norm(nn.Linear(self.z_per, hidden_dim)), nn.SiLU())
            self.o_proj = nn.Linear(hidden_dim, hidden_dim)
            self.norm_ff = nn.LayerNorm(hidden_dim)
            self.ff = nn.Sequential(
                nn.Linear(hidden_dim, dff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(dff, hidden_dim), nn.Dropout(dropout))
            if self.species_groups is not None:
                self.mu_heads = nn.ModuleList([_mu_head() for _ in self.species_groups])
            else:
                self.mu_head = nn.Sequential(
                    P.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
                    nn.SiLU(), nn.Linear(hidden_dim // 2, 3))
                nn.init.zeros_(self.mu_head[-1].bias)
        else:
            # [lateral, normal]; expanded to (x, y, z) as (l, l, n)
            self._dyn_log_gain = nn.Parameter(torch.full((2,), math.log(raw_gain_init)))
            self._dyn_couple = nn.Parameter(torch.zeros(num_heads, 2))
            self._dyn_noise_gain = nn.Parameter(torch.zeros(2))

        if values != 'proj' and (not drift_ln or latent_skip):
            raise ValueError("drift_ln / latent_skip apply to values='proj' only")
        self.drift_ln = drift_ln
        self.z_skip = None
        if latent_skip:
            self.z_skip = nn.Linear(latent_dim, n_coords)
            nn.init.zeros_(self.z_skip.weight)
            nn.init.zeros_(self.z_skip.bias)

    @staticmethod
    def _xyz(a):
        return torch.stack([a[..., 0], a[..., 0], a[..., 1]], dim=-1)

    def _geometry(self, context_embedding, B):
        ctx = (context_embedding.squeeze(1) if context_embedding.dim() == 3
               else context_embedding)
        gat = ctx[..., :self.n_particles * self.gat_hidden].view(
            B, self.n_particles, self.gat_hidden)
        pair = ctx[..., self.n_particles * self.gat_hidden:]
        return self.kv_particle(gat), self.kv_pair(pair).unsqueeze(1)

    def attention(self, g, pair_tok):
        """alpha: [B, heads, P, P+1] over (particles..., pair token), self masked."""
        B, Pn, H = g.shape
        q = self.q_lin(self.norm_q(g)).view(B, Pn, self.num_heads, self.head_dim)
        k = self.k_lin(self.norm_k(torch.cat([g, pair_tok], dim=1))).view(
            B, Pn + 1, self.num_heads, self.head_dim)
        scores = torch.einsum('bphd,bkhd->bhpk', q, k) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.self_mask, float('-inf'))
        return self.attn_drop(torch.softmax(scores.float(), dim=-1).to(g.dtype))

    def _by_species(self, heads, tokens):
        """Each particle's token [B, P, H] through its own species' head -> [B, P, out]."""
        return torch.cat([h(tokens[:, g]) for h, g in zip(heads, self.species_groups)],
                         dim=1)[:, self._unsort]

    def forward(self, z, context_embedding, active_mask=None, eps_prev=None):
        B = z.size(0)
        Pn = self.n_particles
        g, pair_tok = self._geometry(context_embedding, B)            # [B,P,H], [B,1,H]
        alpha = self.attention(g, pair_tok)                           # [B,h,P,P+1]
        zt = z.view(B, Pn, self.z_per)

        if self.values == 'proj':
            v = self.v_proj(zt)                                       # [B,P,H]
            v_all = torch.cat([v, v.new_zeros(B, 1, self.hidden_dim)], dim=1)
            v_all = v_all.view(B, Pn + 1, self.num_heads, self.head_dim)
            m = torch.einsum('bhpk,bkhd->bphd', alpha, v_all).reshape(B, Pn, -1)
            x = g + v + self.o_proj(m)
            x = x + self.ff(self.norm_ff(x))
            out = self.ln_out(x)
            if self.species_groups is not None:
                src = out if self.drift_ln else x
                delta_mu = self._by_species(self.mu_heads, src).reshape(B, self.n_coords)
            else:
                delta_mu = self.mu_head(out if self.drift_ln else x).reshape(B, self.n_coords)
            if self.z_skip is not None:
                delta_mu = delta_mu + self.z_skip(z)
            if self.species_groups is not None:
                noise_params = self._by_species(self.noise_heads, out).reshape(B, -1)
            elif self._per_token_noise:
                noise_params = self.noise_head(out).reshape(B, -1)
            else:
                noise_params = self.noise_head(out.reshape(B, -1))
            return _decoder_finalize(self.noise_type, self.ar1, delta_mu, noise_params,
                                     Pn, active_mask, eps_prev)

        drift, logd = zt[..., :3], zt[..., 3:]
        vals = torch.cat([drift, drift.new_zeros(B, 1, 3)], dim=1)   # [B,P+1,3]
        m = torch.einsum('bhpk,bkc->bhpc', alpha, vals)               # [B,h,P,3]
        couple = (m * self._xyz(self._dyn_couple)[None, :, None, :]).sum(1)
        mu = self._xyz(self._dyn_log_gain.exp()) * drift + couple
        delta_mu = mu.reshape(B, self.n_coords)

        out = self.ln_out(g)
        if self._per_token_noise:
            noise_params = self.noise_head(out).reshape(B, -1)
        else:
            noise_params = self.noise_head(out.reshape(B, -1))
        scale = torch.exp(self._xyz(self._dyn_noise_gain) * logd).reshape(B, self.n_coords)
        delta_sample = delta_mu + scale * _innovation_noise(
            self.noise_type, noise_params, Pn, delta_mu)
        if active_mask is not None:
            slot_mask = active_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(
                -1, self.n_coords).float()
            delta_sample = delta_sample * slot_mask
            delta_mu = delta_mu * slot_mask
        return delta_sample, delta_mu, noise_params, None


# =========================================================================
#  KAN (B-spline Kolmogorov-Arnold) decoder
# =========================================================================

class KANLayer(nn.Module):
    """Single KAN layer: learnable B-spline activation per edge + residual SiLU."""
    def __init__(self, in_dim, out_dim, grid_size=5, spline_order=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.register_buffer('grid', torch.linspace(-1.5, 1.5, grid_size + spline_order + 1))
        n_bases = grid_size + spline_order
        self.spline_weight = nn.Parameter(torch.randn(out_dim, in_dim, n_bases) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(out_dim, in_dim) * (1.0 / math.sqrt(in_dim)))
        self.ln = nn.LayerNorm(out_dim)

    def _b_spline_basis(self, x):
        g = self.grid
        xe = x.unsqueeze(-1)
        bases = ((xe >= g[:-1]) & (xe < g[1:])).float()
        for k in range(1, self.spline_order + 1):
            ln = xe - g[:-(k + 1)]
            ld = g[k:-1] - g[:-(k + 1)]
            rn = g[k + 1:] - xe
            rd = g[k + 1:] - g[1:(-k if k > 0 else None)]
            bases = ln / (ld + 1e-8) * bases[:, :, :-1] + rn / (rd + 1e-8) * bases[:, :, 1:]
        return bases

    def forward(self, x):
        bases = self._b_spline_basis(x)
        spline = torch.einsum('oib,bib->bo', self.spline_weight, bases)
        base = F.silu(F.linear(x, self.base_weight))
        return self.ln(spline + base)


class KANDecoder(nn.Module):
    def __init__(self, gat_embed_dim, hidden_dim=512, latent_dim=21,
                 dropout=0.0, grid_size=5, spline_order=3, n_coords=21, n_particles=7,
                 noise_type='cholesky', decoder_ar1='none'):
        super().__init__()
        self.n_particles = n_particles
        self.noise_type = noise_type
        self.ar1 = build_ar1(decoder_ar1, n_particles)
        n_noise_total = _noise_param_count(noise_type, n_particles)

        self.context_norm = nn.LayerNorm(gat_embed_dim)
        self.latent_proj = nn.Sequential(
            P.spectral_norm(nn.Linear(latent_dim, hidden_dim // 2)), nn.SiLU())
        self.context_proj = nn.Linear(gat_embed_dim, hidden_dim // 2)
        self.trunk = nn.Sequential(
            KANLayer(hidden_dim, hidden_dim, grid_size, spline_order),
            nn.Dropout(dropout),
            KANLayer(hidden_dim, hidden_dim, grid_size, spline_order))
        self.mu_head = nn.Sequential(
            P.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.SiLU(), nn.Linear(hidden_dim // 2, n_coords))
        self.noise_head = nn.Sequential(
            P.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_noise_total))

    def forward(self, z, context_embedding, active_mask=None, eps_prev=None):
        zf = self.latent_proj(z)
        ctx = context_embedding.squeeze(1) if context_embedding.dim() == 3 else context_embedding
        cf = self.context_proj(self.context_norm(ctx))
        x = self.trunk(torch.cat([zf, cf], dim=-1))

        delta_mu = self.mu_head(x)
        noise_params = self.noise_head(x)
        return _decoder_finalize(self.noise_type, self.ar1, delta_mu, noise_params,
                                 self.n_particles, active_mask, eps_prev)


# =========================================================================
#  Factory
# =========================================================================

class TokenizedXFormerDecoder(nn.Module):
    """Per-protein cross-attention: CHANGE attends to POSITION.

    WHY THIS EXISTS. XFormerDecoder builds exactly ONE query token and ONE key/value
    token (`z_proj(z).unsqueeze(1)`, `ctx_proj(ctx).unsqueeze(1)`). Softmax over a
    single key is identically 1.0, so no selection happens between anything -- the
    2 layers x 8 heads are an expensive MLP and the context enters as a fixed
    projection. Nothing in it is relational.

    Here the sequence dimension carries the system's actual structure:

        query  : the CHANGE. P protein tokens = [ z_xyz (3) , node_depth_delta (1) ]
                 + E hyperedge tokens = [ hyper_depth_delta , ksize ].
                 Latent drift+diffusion and the membrane module's output together,
                 both as DELTAS -- never absolute depth. The module integrates
                 internally for its own recurrence; the decoder sees the change.
        key/val: POSITION. P tokens, one per protein = [ x, y, z ], plus one
                 pooled-context token so nothing the old decoder saw is lost.

    Attention is then literal on both axes: self-attention lets a protein's change
    attend to every group's depth change, and cross-attention lets all of them
    attend to where the proteins currently are. Only the first P query tokens are
    read out -- the hyperedge tokens exist to inform the protein tokens, not to
    emit coordinates.

    The deltas are the PREVIOUS step's: membrane_module.step runs after the decoder
    in the rollout, so depth_t - depth_{t-1} is what exists at decode time. That is
    causally correct for an AR model and needs no reordering.

    Heads are per-token and weight-shared across proteins, so mu for protein i is
    driven by protein i's token. A PRIOR, not a bottleneck: TransformerDecoderLayer
    self-attends over the query tokens before cross-attention, so other proteins
    still leak in (measured ~1% at init, and trainable upward).
    """

    QUERY_FEATS = 4          # z_xyz (3) + node depth delta (1)
    HYPER_FEATS = 2          # hyper depth delta + ksize

    def __init__(self, gat_embed_dim, hidden_dim=256, latent_dim=21,
                 num_layers=2, num_heads=8, dim_feedforward=None, dropout=0.0,
                 n_coords=21, n_particles=7, noise_type='cholesky',
                 use_context=True, decoder_ar1='none', use_hyper_tokens=True,
                 box_size=856.196, pos_feat_dim=6):
        super().__init__()
        self.n_particles = n_particles
        self.noise_type = noise_type
        self.use_context = use_context
        self.use_hyper_tokens = use_hyper_tokens
        self.box_size = float(box_size)
        # 6 = sin/cos of absolute position, built here from pos. Anything else is
        # geo_ic_once: the model builds the key features itself (absolute for the IC
        # frame, step deltas after) and hands them over as pos_feat.
        self.pos_feat_dim = pos_feat_dim
        self.ar1 = build_ar1(decoder_ar1, n_particles)
        if latent_dim != n_particles * 3:
            raise ValueError(
                f"TokenizedXFormerDecoder needs a per-protein latent: latent_dim="
                f"{latent_dim} is not n_particles*3={n_particles * 3}. The token "
                f"layout slices z as [P,3]; a latent that is not laid out that way "
                f"would be silently mis-grouped.")
        dff = dim_feedforward if dim_feedforward is not None else hidden_dim * 4

        self.q_proj = nn.Sequential(
            P.spectral_norm(nn.Linear(self.QUERY_FEATS, hidden_dim)), nn.SiLU())
        # NO LayerNorm on either of these. LayerNorm over a 3-vector removes the
        # per-token mean and scale, so a uniform translation is INVISIBLE -- measured:
        # shifting every protein by 257 A moved mu by 1.5e-07. Over a 2-vector it is
        # worse: the output is algebraically [+-1, -+1], i.e. sign(a-b) and nothing
        # else, which both destroys magnitude AND amplifies near-zero deltas to full
        # scale. With 120 such keys against 7 positions, 92% of the cross-attention
        # mass went to constant keys built from numerical noise.
        # Position instead gets the repo's existing periodic encoding (see
        # membrane/base_model.py:48-49): sin/cos of 2*pi*x/box, which is continuous
        # across the box seam where x/box is not -- 0.999 and 0.001 are neighbours.
        self.pos_proj = nn.Linear(pos_feat_dim, hidden_dim)
        # Deltas are divided by their MEASURED per-frame GT std, so a token value of
        # 1.0 means "a typical one-frame move". Buffers because the membrane module
        # owns the measurement and sets them after construction.
        self.hyper_proj = nn.Linear(self.HYPER_FEATS, hidden_dim)
        # NON-persistent, matching membrane_hypergraph_module's own copies. Persistent
        # buffers are restored by load_state_dict, which runs AFTER __init__ fills
        # these -- so a --resume or checkpoint eval would silently reinstate whatever
        # scales the checkpoint was written with (e.g. the pre-1968e79 raw units
        # 1.8087/2.1155) over the corrected ones, permanently and without a warning.
        self.register_buffer('delta_scale_node', torch.ones(()), persistent=False)
        self.register_buffer('delta_scale_hyper', torch.ones(()), persistent=False)
        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(gat_embed_dim), nn.Linear(gat_embed_dim, hidden_dim))
        # Learned per-protein identity, so token order is information rather than
        # an arbitrary permutation the model has to infer.
        self.q_id = nn.Parameter(torch.randn(n_particles, hidden_dim) * 0.02)
        self.kv_id = nn.Parameter(torch.randn(n_particles, hidden_dim) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=dff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.ln_out = nn.LayerNorm(hidden_dim)

        # Per-token heads, shared across proteins: 3 coords and the per-particle
        # noise block for THAT protein.
        self.mu_head = nn.Sequential(
            P.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.SiLU(), nn.Linear(hidden_dim // 2, 3))
        if noise_type == 'cholesky_full':
            raise ValueError(
                "TokenizedXFormerDecoder emits a per-particle noise block; "
                "cholesky_full is a joint 21x21 factor and has no per-token form.")
        self.per_particle_noise = _per_particle_param_count(noise_type)
        self.noise_head = nn.Sequential(
            P.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.SiLU(), nn.Linear(hidden_dim // 2, self.per_particle_noise))
        # ZERO-MEAN DISPLACEMENT AT INIT. mu is a per-step displacement, and GT
        # displacement is sign-symmetric on every axis (measured 50.3/49.8/50.1%
        # negative, COM step mean ~0.01 against std 0.38). A free bias here is a
        # constant velocity: it grows as b*t while the diffusive part grows as
        # sigma*sqrt(t), so it is negligible early and can dominate a 40k-step eval
        # rollout. Training can unlearn it; nothing guaranteed it, so start at zero.
        nn.init.zeros_(self.mu_head[-1].bias)

    def forward(self, z, context_embedding, active_mask=None, eps_prev=None,
                pos=None, node_delta=None, hyper_delta=None, ksize=None, pos_feat=None):
        B = z.size(0)
        P = self.n_particles
        if self.pos_feat_dim != 6 and pos_feat is None:
            raise ValueError(
                "this decoder was built for geo_ic_once: its position keys come from "
                "pos_feat ([B,P,9]), not from a bare pos")
        zt = z.view(B, P, 3)
        nd = (torch.zeros(B, P, 1, device=z.device, dtype=z.dtype)
              if node_delta is None
              else (node_delta.view(B, P, 1).to(z.dtype)
                    / self.delta_scale_node.clamp_min(1e-6)))
        q = self.q_proj(torch.cat([zt, nd], dim=-1)) + self.q_id.unsqueeze(0)
        if self.use_hyper_tokens and hyper_delta is not None:
            hd = (hyper_delta.unsqueeze(-1).to(z.dtype)
                  / self.delta_scale_hyper.clamp_min(1e-6))
            ks = (torch.zeros_like(hd) if ksize is None
                  else ksize.view(1, -1, 1).to(hd.dtype).expand_as(hd) / float(P))
            q = torch.cat([q, self.hyper_proj(torch.cat([hd, ks], dim=-1))], dim=1)

        mem = []
        if pos_feat is not None:
            mem.append(self.pos_proj(pos_feat.to(z.dtype)) + self.kv_id.unsqueeze(0))
        elif pos is not None:
            # Periodic, so the encoding is continuous across the box boundary and
            # absolute position survives (a plain x/box under LayerNorm did not).
            ang = 2.0 * math.pi * torch.remainder(
                pos.view(B, P, 3), self.box_size) / self.box_size
            pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).to(z.dtype)
            mem.append(self.pos_proj(pe) + self.kv_id.unsqueeze(0))
        if self.use_context and context_embedding is not None:
            ctx = (context_embedding.squeeze(1) if context_embedding.dim() == 3
                   else context_embedding)
            mem.append(self.ctx_proj(ctx).unsqueeze(1))
        if not mem:
            raise ValueError("TokenizedXFormerDecoder got no key/value tokens: pass "
                             "pos, hyper_delta or a context embedding.")

        out = self.ln_out(self.transformer(q, torch.cat(mem, dim=1)))   # [B,P+E,H]
        out = out[:, :P]                       # protein tokens only carry coordinates
        delta_mu = self.mu_head(out).reshape(B, P * 3)
        noise_params = self.noise_head(out).reshape(B, P * self.per_particle_noise)
        return _decoder_finalize(self.noise_type, self.ar1, delta_mu, noise_params,
                                 self.n_particles, active_mask, eps_prev)


def build_decoder(kind, gat_embed_dim, latent_dim, n_coords, config):
    hidden_dim = config.get('dec_hidden_dim', 256)
    dropout = config.get('dropout', 0.1)
    n_particles = config.get('n_particles', n_coords // 3)
    noise_type = config.get('decoder_noise_type', 'cholesky')
    decoder_ar1 = config.get('decoder_ar1', 'none')
    use_context = config.get('dec_use_context', True)

    if kind == 'mlp':
        return MLPDecoder(
            gat_embed_dim=gat_embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            num_layers=config.get('dec_mlp_layers', 3),
            n_coords=n_coords,
            n_particles=n_particles,
            noise_type=noise_type,
            use_context=use_context,
            decoder_ar1=decoder_ar1,
        )
    if kind == 'xformer':
        return XFormerDecoder(
            gat_embed_dim=gat_embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=config.get('dec_xformer_layers', 2),
            num_heads=config.get('dec_xformer_heads', 8),
            dim_feedforward=config.get('dec_xformer_dim_feedforward', None),
            dropout=dropout,
            n_coords=n_coords,
            n_particles=n_particles,
            noise_type=noise_type,
            use_context=use_context,
            decoder_ar1=decoder_ar1,
        )
    if kind == 'xformer_particles' and config.get('dec_latent_values', 'none') != 'none':
        return GeomQueryDecoder(
            gat_embed_dim=gat_embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=config.get('dec_xformer_layers', 2),
            num_heads=config.get('dec_xformer_heads', 8),
            dim_feedforward=config.get('dec_xformer_dim_feedforward', None),
            dropout=dropout,
            n_coords=n_coords,
            n_particles=n_particles,
            noise_type=noise_type,
            use_context=use_context,
            decoder_ar1=decoder_ar1,
            values=config['dec_latent_values'],
            raw_gain_init=config.get('dec_raw_gain_init', 0.1),
            drift_ln=config.get('dec_drift_ln', True),
            latent_skip=config.get('dec_latent_skip', False),
            species_groups=(config.get('species_groups')
                            if config.get('dec_per_species', False) else None),
        )
    if config.get('dec_per_species', False):
        raise ValueError("dec_per_species is implemented for decoder_type 'xformer_particles' "
                         "with dec_latent_values set (GeomQueryDecoder) only")
    if kind == 'xformer_particles':
        return ParticleXFormerDecoder(
            gat_embed_dim=gat_embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=config.get('dec_xformer_layers', 2),
            num_heads=config.get('dec_xformer_heads', 8),
            dim_feedforward=config.get('dec_xformer_dim_feedforward', None),
            dropout=dropout,
            n_coords=n_coords,
            n_particles=n_particles,
            noise_type=noise_type,
            use_context=use_context,
            decoder_ar1=decoder_ar1,
            latent_skip=config.get('dec_latent_skip', False),
        )
    if kind == 'xformer_tokens':
        return TokenizedXFormerDecoder(
            gat_embed_dim=gat_embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=config.get('dec_xformer_layers', 2),
            num_heads=config.get('dec_xformer_heads', 8),
            dim_feedforward=config.get('dec_xformer_dim_feedforward', None),
            dropout=dropout,
            n_coords=n_coords,
            n_particles=n_particles,
            noise_type=noise_type,
            use_context=use_context,
            decoder_ar1=decoder_ar1,
            use_hyper_tokens=config.get('dec_hyper_tokens', True),
            box_size=config.get('box_size', 856.196),
            pos_feat_dim=9 if config.get('geo_ic_once', False) else 6,
        )
    if kind == 'kan':
        return KANDecoder(
            gat_embed_dim=gat_embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            grid_size=config.get('kan_grid_size', 5),
            spline_order=config.get('kan_spline_order', 3),
            n_coords=n_coords,
            n_particles=n_particles,
            noise_type=noise_type,
            decoder_ar1=decoder_ar1,
        )
    raise ValueError(f"Unknown decoder_type={kind!r}")
