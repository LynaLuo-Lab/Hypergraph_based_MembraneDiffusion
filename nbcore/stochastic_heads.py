"""Stochastic latent heads selected by config['latent_type'].

The head's job is to advance z_{t-1} -> z_t given transition_params from an
encoder transition head. Three noise-path shapes are supported:

  - MLPGaussTransitionHead  -> sigma_flat is a packed full 21x21 Cholesky factor
    (diagonal entries via softplus, strict lower-triangle unconstrained).
    Consumed by BrownianHead / OUHead / OUGMMHead.

  - NSFTransitionHead       -> sigma_flat is a REPARAMETERIZED INNOVATION SAMPLE
    from a conditional NSF, shape [B, latent_dim]. Consumed by OUFlowHead.
    The NSF has already sampled per-step; the stochastic head only integrates.

latent_type selects the head; the matching transition_head_type (see
nbcore.transition_heads) must produce the corresponding sigma_flat payload.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nbcore.transition_heads import LatentTransitionParams


def _safe_exp_neg(x):
    """exp(-x) that works for both Tensor and float."""
    if isinstance(x, torch.Tensor):
        return torch.exp(-x)
    return math.exp(-x)


# =========================================================================
#  Base + Cholesky helpers
# =========================================================================

_CHOL_DIAG_EPS = 1e-3


def _inverse_softplus(x, threshold=20.0):
    """Inverse of F.softplus: returns y such that softplus(y) ≈ x."""
    if x > threshold:
        return float(x)
    return float(torch.tensor(x).expm1().log())


def _full_cholesky_noise(sigma_flat, latent_dim, device):
    """Sample N(0, L L^T) for a batched lower-triangular Cholesky factor.

    sigma_flat: [B, latent_dim * (latent_dim + 1) // 2]
      First latent_dim entries -> diagonal (passed through softplus + eps).
      Remaining latent_dim*(latent_dim-1)//2 entries -> strict lower triangle.
    returns:    [B, latent_dim]
    """
    B = sigma_flat.size(0)
    diag_raw = sigma_flat[:, :latent_dim]
    offdiag = sigma_flat[:, latent_dim:]
    diag = F.softplus(diag_raw) + _CHOL_DIAG_EPS

    L = torch.zeros(B, latent_dim, latent_dim, device=sigma_flat.device, dtype=sigma_flat.dtype)
    diag_idx = torch.arange(latent_dim, device=sigma_flat.device)
    L[:, diag_idx, diag_idx] = diag
    row_idx, col_idx = torch.tril_indices(latent_dim, latent_dim, offset=-1, device=sigma_flat.device)
    L[:, row_idx, col_idx] = offdiag

    eps = torch.randn(B, latent_dim, 1, device=device, dtype=sigma_flat.dtype)
    return torch.bmm(L, eps).squeeze(-1)


def _lowrank_diag_noise(sigma_flat, latent_dim, noise_rank, device):
    """Sample N(0, L L^T + D^2) for low-rank factor L plus diagonal D.

    Layout (matches pre-Apr-20 diffusion/model_multilag_mlp_7x3d.py training):
      sigma_flat: [B, (noise_rank + 1) * latent_dim]
        First latent_dim entries -> diagonal D_raw (softplus + eps).
        Remaining noise_rank * latent_dim entries -> L flat, reshape to
        [B, latent_dim, noise_rank].
    Sampling:  x = L @ z_r + diag(D) @ z_d,  z_r ~ N(0, I_r), z_d ~ N(0, I_d).
    """
    B = sigma_flat.size(0)
    diag_raw = sigma_flat[:, :latent_dim]
    L_flat = sigma_flat[:, latent_dim:]
    diag = F.softplus(diag_raw) + _CHOL_DIAG_EPS

    L = L_flat.reshape(B, latent_dim, noise_rank)
    z_r = torch.randn(B, noise_rank, 1, device=device, dtype=sigma_flat.dtype)
    z_d = torch.randn(B, latent_dim, device=device, dtype=sigma_flat.dtype)
    return torch.bmm(L, z_r).squeeze(-1) + diag * z_d


def _sample_noise(sigma_flat, latent_dim, noise_rank, sigma_layout, device):
    if sigma_layout == 'cholesky':
        return _full_cholesky_noise(sigma_flat, latent_dim, device)
    if sigma_layout == 'lowrank_diag':
        return _lowrank_diag_noise(sigma_flat, latent_dim, noise_rank, device)
    raise ValueError(f"Unknown sigma_layout={sigma_layout!r}")


class BaseStochasticHead(nn.Module):
    def __init__(self, latent_dim, noise_rank, dt, sigma_layout='cholesky'):
        super().__init__()
        self.latent_dim, self.noise_rank, self.dt = latent_dim, noise_rank, dt
        self.sigma_layout = sigma_layout

    def _default_stats(self, device):
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    def _require_standard_params(self, params: LatentTransitionParams):
        if params.mixture_logits is not None:
            raise ValueError("Mixture latent params require latent_type='ou_gmm'.")
        return params.mu, params.sigma_flat

    def init_noise_state(self, batch_size, device, dtype=torch.float32):
        return None

    def step(self, transition_params, z_prev=None, noise_state=None):
        raise NotImplementedError


# =========================================================================
#  Gaussian-innovation heads (consume Cholesky-packed sigma_flat)
# =========================================================================

class BrownianHead(BaseStochasticHead):
    """dz = mu*dt + sigma*dW — pure integration, drift accumulates."""
    def step(self, transition_params, z_prev=None, noise_state=None):
        mu, sigma_flat = self._require_standard_params(transition_params)
        orig = mu.dtype
        mu = mu.float()
        sigma_flat = sigma_flat.float()
        diff = _sample_noise(sigma_flat, self.latent_dim, self.noise_rank, self.sigma_layout, mu.device)
        delta_z = mu * self.dt + diff * (self.dt ** 0.5)
        z_new = (z_prev.float() + delta_z) if z_prev is not None else delta_z
        return z_new.to(orig), *self._default_stats(mu.device), noise_state


class OUHead(BaseStochasticHead):
    """dz = theta*(mu - z)*dt + sigma*dW — mean-reverts toward encoder's mu."""
    def __init__(self, latent_dim, noise_rank, dt, theta=None, sigma_layout='cholesky',
                 learnable_theta=False, learnable_dt=False):
        super().__init__(latent_dim, noise_rank, dt, sigma_layout=sigma_layout)
        theta_val = theta if theta is not None else (1.0 / dt)
        if learnable_theta:
            self._log_theta = nn.Parameter(torch.tensor(float(theta_val)).log())
        else:
            self._log_theta = None
            self.theta = theta_val
        if learnable_dt:
            self._log_dt = nn.Parameter(torch.tensor(float(dt)).log())
        else:
            self._log_dt = None

    @property
    def effective_theta(self):
        if self._log_theta is not None:
            return self._log_theta.exp()
        return self.theta

    @property
    def effective_dt(self):
        if self._log_dt is not None:
            return self._log_dt.exp()
        return self.dt

    def step(self, transition_params, z_prev=None, noise_state=None):
        mu, sigma_flat = self._require_standard_params(transition_params)
        orig = mu.dtype
        mu = mu.float()
        sigma_flat = sigma_flat.float()
        diff = _sample_noise(sigma_flat, self.latent_dim, self.noise_rank, self.sigma_layout, mu.device)
        theta = self.effective_theta
        dt = self.effective_dt
        decay = _safe_exp_neg(theta * dt)
        if z_prev is not None:
            z_f = z_prev.float()
            z_new = mu + (z_f - mu) * decay + diff * (dt ** 0.5)
        else:
            z_new = mu + diff * (dt ** 0.5)
        return z_new.to(orig), *self._default_stats(mu.device), noise_state


class OUGMMHead(BaseStochasticHead):
    """Sample one latent mixture component per step, then apply an OU update.

    Kept for ablation vs OUFlowHead. Note: torch.multinomial on the component
    selection is non-reparameterizable; gradient does not flow through comp_idx.
    """

    def __init__(self, latent_dim, noise_rank, dt, theta=None, sigma_layout='cholesky',
                 learnable_theta=False, learnable_dt=False):
        super().__init__(latent_dim, noise_rank, dt, sigma_layout=sigma_layout)
        theta_val = theta if theta is not None else (1.0 / dt)
        if learnable_theta:
            self._log_theta = nn.Parameter(torch.tensor(float(theta_val)).log())
        else:
            self._log_theta = None
            self.theta = theta_val
        if learnable_dt:
            self._log_dt = nn.Parameter(torch.tensor(float(dt)).log())
        else:
            self._log_dt = None

    @property
    def effective_theta(self):
        if self._log_theta is not None:
            return self._log_theta.exp()
        return self.theta

    @property
    def effective_dt(self):
        if self._log_dt is not None:
            return self._log_dt.exp()
        return self.dt

    def step(self, transition_params, z_prev=None, noise_state=None):
        if transition_params.mixture_logits is None:
            raise ValueError("latent_type='ou_gmm' requires mixture logits from the encoder.")

        logits = transition_params.mixture_logits.float()
        probs = torch.softmax(logits, dim=-1)
        comp_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        batch_idx = torch.arange(probs.size(0), device=probs.device)

        mu = transition_params.mu.float()[batch_idx, comp_idx]
        sigma_flat = transition_params.sigma_flat.float()[batch_idx, comp_idx]
        diff = _sample_noise(sigma_flat, self.latent_dim, self.noise_rank, self.sigma_layout, mu.device)

        theta = self.effective_theta
        dt = self.effective_dt
        decay = _safe_exp_neg(theta * dt)
        if z_prev is not None:
            z_f = z_prev.float()
            z_new = mu + (z_f - mu) * decay + diff * (dt ** 0.5)
        else:
            z_new = mu + diff * (dt ** 0.5)

        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        top1_prob = probs.max(dim=-1).values.mean()
        return z_new.to(transition_params.mu.dtype), entropy, top1_prob, noise_state


# =========================================================================
#  NSF-innovation head (consumes pre-sampled innovation in sigma_flat)
# =========================================================================

class ColoredNoiseFilter(nn.Module):
    """AR noise filter applied to white innovations before OU integration.

    Tiers (increasing expressiveness):
      'scalar'   — single ρ shared across all dims.           1 param.
      'scalar_ar2' — single (ρ1, ρ2) AR(2) shared across dims. 2 params.
      'per_dim'  — independent ρ_d per dimension.             D params.
      'ar2'      — per-dim AR(2): two coefficients per dim.   2D params.
      'vector'   — diag(ρ) + low-rank UV' cross-dim coupling. D + 2Dr params.
      'vector_ar2' — vector AR(1) on lag-1 + lag-2 states.    2(D + 2Dr) params.

    The filter is: n_t = A · n_{t-1} [+ B · n_{t-2}] + scale · ε_t
    where scale = √(1 - ||effective_coeffs||²) to preserve unit variance.
    State resets when z_prev is None (first AR step).
    """

    def __init__(self, latent_dim, mode='scalar', rank=4, init_corr=0.5):
        super().__init__()
        self.mode = mode
        self.latent_dim = latent_dim
        self.rank = rank

        init_logit = math.log(init_corr / (1.0 - init_corr))

        if mode == 'scalar':
            self._raw = nn.Parameter(torch.tensor(init_logit))
        elif mode == 'per_dim':
            self._raw = nn.Parameter(torch.full((latent_dim,), init_logit))
        elif mode == 'ar2':
            self._raw1 = nn.Parameter(torch.full((latent_dim,), init_logit))
            self._raw2 = nn.Parameter(torch.zeros(latent_dim))
        elif mode == 'scalar_ar2':
            self._raw1 = nn.Parameter(torch.tensor(init_logit))
            self._raw2 = nn.Parameter(torch.tensor(0.0))
        elif mode == 'vector':
            self._raw_diag = nn.Parameter(torch.full((latent_dim,), init_logit))
            self._U = nn.Parameter(torch.randn(latent_dim, rank) * 0.01)
            self._V = nn.Parameter(torch.randn(latent_dim, rank) * 0.01)
        elif mode == 'vector_ar2':
            self._raw_diag1 = nn.Parameter(torch.full((latent_dim,), init_logit))
            self._U1 = nn.Parameter(torch.randn(latent_dim, rank) * 0.01)
            self._V1 = nn.Parameter(torch.randn(latent_dim, rank) * 0.01)
            self._raw_diag2 = nn.Parameter(torch.zeros(latent_dim))
            self._U2 = nn.Parameter(torch.randn(latent_dim, rank) * 0.01)
            self._V2 = nn.Parameter(torch.randn(latent_dim, rank) * 0.01)
        else:
            raise ValueError(f"Unknown colored_noise mode: {mode!r}")

    def _get_A(self, which=1):
        """Build the transition matrix/vector for lag `which` (1 or 2)."""
        if self.mode == 'scalar':
            return torch.sigmoid(self._raw)
        elif self.mode == 'per_dim':
            return torch.sigmoid(self._raw)
        elif self.mode in ('ar2', 'scalar_ar2'):
            raw = self._raw1 if which == 1 else self._raw2
            return torch.tanh(raw) * 0.5
        elif self.mode == 'vector':
            d = torch.sigmoid(self._raw_diag)
            return d, self._U, self._V
        elif self.mode == 'vector_ar2':
            if which == 1:
                d = torch.tanh(self._raw_diag1) * 0.5
                return d, self._U1, self._V1
            else:
                d = torch.tanh(self._raw_diag2) * 0.3
                return d, self._U2, self._V2

    def _apply_A(self, A, x):
        """Apply transition to state x. Handles scalar/vector/matrix cases."""
        if self.mode in ('scalar', 'per_dim', 'ar2', 'scalar_ar2'):
            return A * x
        else:
            d, U, V = A
            return d * x + (x @ V) @ U.T

    @property
    def effective_noise_corr(self):
        """Return the diagonal correlation coefficient(s) for logging."""
        if self.mode == 'scalar':
            return torch.sigmoid(self._raw)
        elif self.mode == 'per_dim':
            return torch.sigmoid(self._raw).mean()
        elif self.mode == 'ar2':
            return torch.tanh(self._raw1).mean() * 0.5
        elif self.mode == 'scalar_ar2':
            return torch.tanh(self._raw1) * 0.5
        elif self.mode == 'vector':
            return torch.sigmoid(self._raw_diag).mean()
        elif self.mode == 'vector_ar2':
            return torch.tanh(self._raw_diag1).mean() * 0.5

    @property
    def has_lag2(self):
        return self.mode in ('ar2', 'scalar_ar2', 'vector_ar2')

    def init_state(self, batch_size, device, dtype=torch.float32):
        """Return initial (state1, state2) zero tensors."""
        s1 = torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)
        s2 = torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)
        return s1, s2

    def forward(self, white_noise, state1, state2, is_first_step):
        """Filter white noise into colored noise (stateless — state passed in/out).

        Args:
            white_noise: [B, D]
            state1: [B, D] previous output (n_{t-1})
            state2: [B, D] n_{t-2} (used by AR2 modes)
            is_first_step: bool — if True, skip AR and pass through white_noise
        Returns:
            (colored, new_state1, new_state2)
        """
        if is_first_step:
            return white_noise, white_noise, state2

        A1 = self._get_A(1)
        contrib = self._apply_A(A1, state1)

        if self.has_lag2:
            A2 = self._get_A(2)
            contrib = contrib + self._apply_A(A2, state2)

        colored = contrib + white_noise

        new_state2 = state1 if self.has_lag2 else state2
        return colored, colored, new_state2


class OUFlowHead(BaseStochasticHead):
    """OU shell with innovations (NSF, randn, or colored-noise filtered).

        z_t = mu + (z_{t-1} - mu) * decay + innovation * innov_scale

    Three parameterization modes:
      - Fixed (default): decay = exp(-theta*dt), innov_scale = sqrt(dt).
      - Log-learnable theta/dt: same formula, theta and/or dt learned in log-space.
      - Direct dynamics (learnable_dynamics=True): decay = sigmoid(raw),
        innov_scale = softplus(raw). Decoupled, no gradient dead zones.

    Optional colored noise: set colored_noise to one of
      'scalar', 'scalar_ar2', 'per_dim', 'ar2', 'vector', 'vector_ar2'.
    """

    def __init__(self, latent_dim, noise_rank, dt, theta=None, sigma_layout='cholesky',
                 learnable_theta=False, learnable_dt=False, learnable_dynamics=False,
                 init_decay=0.5, init_innov_scale=0.3):
        super().__init__(latent_dim, noise_rank, dt, sigma_layout=sigma_layout)
        self._learnable_dynamics = learnable_dynamics

        if learnable_dynamics:
            self._raw_decay = nn.Parameter(
                torch.tensor(math.log(init_decay / (1.0 - init_decay))))
            self._raw_innov_scale = nn.Parameter(
                torch.tensor(_inverse_softplus(init_innov_scale)))
            self._log_theta = None
            self._log_dt = None
            self.theta = theta if theta is not None else (1.0 / dt)
        else:
            self._raw_decay = None
            self._raw_innov_scale = None
            theta_val = theta if theta is not None else (1.0 / dt)
            if learnable_theta:
                self._log_theta = nn.Parameter(torch.tensor(float(theta_val)).log())
            else:
                self._log_theta = None
                self.theta = theta_val
            if learnable_dt:
                self._log_dt = nn.Parameter(torch.tensor(float(dt)).log())
            else:
                self._log_dt = None

    @property
    def effective_theta(self):
        if self._log_theta is not None:
            return self._log_theta.exp()
        return self.theta

    @property
    def effective_dt(self):
        if self._log_dt is not None:
            return self._log_dt.exp()
        return self.dt

    @property
    def effective_decay(self):
        if self._raw_decay is not None:
            return torch.sigmoid(self._raw_decay)
        return _safe_exp_neg(self.effective_theta * self.effective_dt)

    @property
    def effective_innov_scale(self):
        if self._raw_innov_scale is not None:
            return F.softplus(self._raw_innov_scale)
        dt = self.effective_dt
        if isinstance(dt, torch.Tensor):
            return dt.sqrt()
        return dt ** 0.5

    def step(self, transition_params, z_prev=None, noise_state=None):
        if transition_params.mixture_logits is not None:
            raise ValueError("OUFlowHead does not consume mixture logits.")
        mu = transition_params.mu
        innovation = transition_params.sigma_flat
        if innovation.shape != mu.shape:
            raise ValueError(
                f"OUFlowHead expects innovation shape == mu shape; "
                f"got innovation={tuple(innovation.shape)}, mu={tuple(mu.shape)}. "
                f"Check that transition_head_type='nsf'."
            )
        orig = mu.dtype
        mu = mu.float()
        innovation = innovation.float()

        decay = self.effective_decay
        innov_scale = self.effective_innov_scale
        if z_prev is not None:
            z_f = z_prev.float()
            z_new = mu + (z_f - mu) * decay + innovation * innov_scale
        else:
            z_new = mu + innovation * innov_scale
        return z_new.to(orig), *self._default_stats(mu.device), noise_state


class OURandnHead(BaseStochasticHead):
    """OU process with self-generated white or colored noise.

        z_t = mu + (z_{t-1} - mu) * decay + noise * innov_scale

    Uses mu from the transition head; ignores sigma_flat entirely.
    Generates randn internally, optionally filtered through ColoredNoiseFilter.

    Pair with transition_head_type='mu_only'.
    """

    def __init__(self, latent_dim, noise_rank, dt, theta=None, sigma_layout='cholesky',
                 learnable_dynamics=False, init_decay=0.5, init_innov_scale=0.3,
                 colored_noise=None, colored_noise_rank=4, init_noise_corr=0.5,
                 per_dim_dynamics=False, freeze_decay=False, species_of_dim=None,
                 freeze_innov_scale=False):
        super().__init__(latent_dim, noise_rank, dt, sigma_layout=sigma_layout)
        self._learnable_dynamics = learnable_dynamics
        self.per_dim_dynamics = per_dim_dynamics
        # species_of_dim: for each latent dim, the species of the particle that owns it.
        # One decay/innov_scale per SPECIES then: per_dim_dynamics gives every latent dim
        # its own pair, which breaks exchangeability inside a species (piezotrek), and the
        # scalar pair makes a fast and a slow protein share one noise scale -- the encoder
        # can tell them apart only through mu, which the noise swamps (gm1010: the two
        # proteins' latents differ by 5% of their size). Initialized flat, like per_dim.
        idx = None
        if species_of_dim is not None and len(set(species_of_dim)) > 1:
            if per_dim_dynamics:
                raise ValueError("per_dim_dynamics and per_species_dynamics are exclusive")
            if len(species_of_dim) != latent_dim:
                raise ValueError(f"species_of_dim has {len(species_of_dim)} entries for latent_dim {latent_dim}")
            idx = torch.as_tensor(species_of_dim, dtype=torch.long)
        self.register_buffer('_dyn_index', idx, persistent=False)

        # per_dim_dynamics: one decay/innov_scale PER latent dim instead of one
        # scalar for all of them. For a heterogeneous system (gm1010: M2 fast,
        # GIRK2 slow) a single shared pair forces both proteins onto the same
        # memory timescale and the same diffusion magnitude -- mu can differ per
        # protein, the dynamics cannot. Initialized flat, so step 0 is identical
        # to the scalar model; the dims only diverge if training pushes them.
        d0 = math.log(init_decay / (1.0 - init_decay))
        s0 = _inverse_softplus(init_innov_scale)
        shape = (latent_dim,) if per_dim_dynamics else ()
        if self._dyn_index is not None:
            shape = (int(self._dyn_index.max()) + 1,)
        self._raw_decay = nn.Parameter(torch.full(shape, d0))
        self._raw_innov_scale = nn.Parameter(torch.full(shape, s0))
        # freeze_decay pins decay at init_decay. On gm1010 the seeds split into dead /
        # white / persistent latents within 3 epochs and that split was the seed spread.
        if freeze_decay:
            self._raw_decay.requires_grad_(False)
        # freeze_innov_scale pins the latent noise at init_innov_scale. With the species kept
        # apart on gm1010 it climbs from 5 to 47 over 68 epochs and the drift never forms.
        if freeze_innov_scale:
            self._raw_innov_scale.requires_grad_(False)

        if colored_noise:
            self.noise_filter = ColoredNoiseFilter(
                latent_dim, mode=colored_noise, rank=colored_noise_rank,
                init_corr=init_noise_corr)
        else:
            self.noise_filter = None

    def _per_dim(self, v):
        return v if self._dyn_index is None else v[self._dyn_index]

    @property
    def effective_decay(self):
        return self._per_dim(torch.sigmoid(self._raw_decay))

    @property
    def effective_innov_scale(self):
        return self._per_dim(F.softplus(self._raw_innov_scale))

    def init_noise_state(self, batch_size, device, dtype=torch.float32):
        if self.noise_filter is not None:
            return self.noise_filter.init_state(batch_size, device, dtype)
        return None

    def step(self, transition_params, z_prev=None, noise_state=None):
        mu = transition_params.mu
        orig = mu.dtype
        mu = mu.float()

        noise = torch.randn_like(mu)
        if self.noise_filter is not None and noise_state is not None:
            ns1, ns2 = noise_state
            noise, ns1, ns2 = self.noise_filter(
                noise, ns1, ns2, is_first_step=(z_prev is None))
            noise_state = (ns1, ns2)

        decay = self.effective_decay
        innov_scale = self.effective_innov_scale
        if z_prev is not None:
            z_f = z_prev.float()
            z_new = mu + (z_f - mu) * decay + noise * innov_scale
        else:
            z_new = mu + noise * innov_scale
        return z_new.to(orig), *self._default_stats(mu.device), noise_state


# =========================================================================
#  Factory
# =========================================================================

def species_ids(species_groups, n_particles):
    """species_groups (a partition of range(n_particles)) -> the species of each particle."""
    ids = [None] * n_particles
    for s, g in enumerate(species_groups or []):
        for p in g:
            if not 0 <= p < n_particles or ids[p] is not None:
                raise ValueError(f"species_groups must partition range({n_particles}); got {species_groups}")
            ids[p] = s
    if None in ids:
        raise ValueError(f"species_groups must partition range({n_particles}); got {species_groups}")
    return ids


def _species_of_dim(config, latent_dim):
    """per_species_dynamics: the latent is one slice per particle, so each latent dim
    belongs to a particle and through it to a species. None when the flag is off."""
    if not config.get('per_species_dynamics', False):
        return None
    n = config.get('n_particles', 7)
    if latent_dim % n:
        raise ValueError(f"per_species_dynamics needs latent_dim ({latent_dim}) divisible by n_particles ({n})")
    ids = species_ids(config.get('species_groups'), n)
    return [ids[d // (latent_dim // n)] for d in range(latent_dim)]


def get_stochastic_head(config):
    lt = config.get('latent_type', 'brownian')
    ld = config.get('latent_dim', 21)
    nr = config.get('noise_rank', 16)
    dt = config.get('dt', 0.0001)
    theta = config.get('ou_theta', 1.0 / dt)
    sl = config.get('sigma_layout', 'cholesky')
    learn_theta = config.get('learnable_theta', False)
    learn_dt = config.get('learnable_dt', False)
    dt_init = config.get('ou_dt_init', dt)
    if lt == 'brownian':
        return BrownianHead(ld, nr, dt, sigma_layout=sl)
    if lt == 'ou':
        return OUHead(ld, nr, dt_init, theta=theta, sigma_layout=sl,
                      learnable_theta=learn_theta, learnable_dt=learn_dt)
    if lt == 'ou_gmm':
        return OUGMMHead(ld, nr, dt_init, theta=theta, sigma_layout=sl,
                         learnable_theta=learn_theta, learnable_dt=learn_dt)
    if lt == 'ou_nsf':
        learn_dyn = config.get('learnable_dynamics', False)
        return OUFlowHead(ld, nr, dt_init, theta=theta,
                          learnable_theta=learn_theta, learnable_dt=learn_dt,
                          learnable_dynamics=learn_dyn,
                          init_decay=config.get('init_decay', 0.5),
                          init_innov_scale=config.get('init_innov_scale', 0.3))
    if lt == 'ou_randn':
        return OURandnHead(ld, nr, dt_init, theta=theta,
                           learnable_dynamics=True,
                           init_decay=config.get('init_decay', 0.5),
                           init_innov_scale=config.get('init_innov_scale', 0.3),
                           colored_noise=config.get('colored_noise', None),
                           colored_noise_rank=config.get('colored_noise_rank', 4),
                           init_noise_corr=config.get('init_noise_corr', 0.5),
                           per_dim_dynamics=config.get('per_dim_dynamics', False),
                           freeze_decay=config.get('freeze_decay', False),
                           freeze_innov_scale=config.get('freeze_innov_scale', False),
                           species_of_dim=_species_of_dim(config, ld))
    raise ValueError(
        f"Unknown latent_type {lt!r}. Use 'brownian', 'ou', 'ou_gmm', 'ou_nsf', or 'ou_randn'."
    )
