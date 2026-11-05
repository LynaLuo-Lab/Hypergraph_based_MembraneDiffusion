"""Transition heads — map encoder hidden state to latent-SDE parameters.

Each head produces a LatentTransitionParams(mu, sigma_flat, mixture_logits=None)
consumed by nbcore.stochastic_heads.

sigma_flat semantics depend on which head produces it:
  - MLPGaussTransitionHead:
      Packed full latent_dim x latent_dim Cholesky factor.
      Layout: first latent_dim entries are diagonal (softplus + eps downstream),
              remaining latent_dim*(latent_dim-1)//2 entries are strict lower triangle.
      Consumed by BrownianHead / OUHead / OUGMMHead via _full_cholesky_noise.
  - NSFTransitionHead:
      Already-sampled innovation vector from a conditional NSF, shape [B, latent_dim].
      Consumed by OUFlowHead directly (no further noise sampling).
"""
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P


# =========================================================================
#  Transition params container
# =========================================================================

class LatentTransitionParams(NamedTuple):
    mu: torch.Tensor
    sigma_flat: torch.Tensor
    mixture_logits: Optional[torch.Tensor] = None


def squeeze_transition_params(params: LatentTransitionParams, dim: int) -> LatentTransitionParams:
    mixture_logits = params.mixture_logits
    if mixture_logits is not None:
        mixture_logits = mixture_logits.squeeze(dim)
    return LatentTransitionParams(
        mu=params.mu.squeeze(dim),
        sigma_flat=params.sigma_flat.squeeze(dim),
        mixture_logits=mixture_logits,
    )


def cholesky_flat_dim(latent_dim: int) -> int:
    """Number of parameters needed to parameterize a full latent_dim x latent_dim
    lower-triangular Cholesky factor (diagonal + strict lower-triangle)."""
    return latent_dim * (latent_dim + 1) // 2


def lowrank_diag_flat_dim(latent_dim: int, noise_rank: int) -> int:
    """Number of parameters for the legacy low-rank+diagonal sigma layout:
    diagonal D (latent_dim) + low-rank factor L of shape [latent_dim, noise_rank]."""
    return (noise_rank + 1) * latent_dim


# =========================================================================
#  MLP-Gaussian head (full Cholesky or legacy low-rank+diag; supports GMM)
# =========================================================================

class MLPGaussTransitionHead(nn.Module):
    """MLP mu/sigma heads producing noise params.

    sigma_layout:
      'cholesky'      — packed full latent_dim x latent_dim lower-triangular
                        Cholesky factor (current). sigma_dim = D*(D+1)/2.
      'lowrank_diag'  — diagonal D (latent_dim) plus low-rank factor L of shape
                        [latent_dim, noise_rank]. sigma_dim = (noise_rank+1)*latent_dim.
                        Used by pre-Apr-20 checkpoints (and Apr 21+ checkpoints
                        trained on the un-merged lowrank+diag branch).

    Under GMM mode (latent_type='ou_gmm'), outputs per-component mu/sigma/logits.
    """

    def __init__(self, input_dim, latent_dim, noise_rank=None,
                 latent_type='brownian', gmm_components=4,
                 sigma_layout='cholesky'):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_type = latent_type
        self.gmm_components = gmm_components
        self.sigma_layout = sigma_layout
        if sigma_layout == 'cholesky':
            self.sigma_dim = cholesky_flat_dim(latent_dim)
            self.noise_rank = None
        elif sigma_layout == 'lowrank_diag':
            if noise_rank is None:
                raise ValueError("sigma_layout='lowrank_diag' requires noise_rank.")
            self.sigma_dim = lowrank_diag_flat_dim(latent_dim, noise_rank)
            self.noise_rank = noise_rank
        else:
            raise ValueError(f"Unknown sigma_layout={sigma_layout!r}")
        self.use_mixture = latent_type == 'ou_gmm'
        hidden_dim = max(input_dim // 2, latent_dim)

        mu_out = latent_dim * gmm_components if self.use_mixture else latent_dim
        sigma_out = self.sigma_dim * gmm_components if self.use_mixture else self.sigma_dim

        self.mu_head = nn.Sequential(
            P.spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.SiLU(),
            nn.Linear(hidden_dim, mu_out),
        )
        self.sigma_head = nn.Sequential(
            P.spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.SiLU(),
            nn.Linear(hidden_dim, sigma_out),
        )
        self.logits_head = None
        if self.use_mixture:
            self.logits_head = nn.Sequential(
                P.spectral_norm(nn.Linear(input_dim, hidden_dim)),
                nn.SiLU(),
                nn.Linear(hidden_dim, gmm_components),
            )

    def forward(self, hidden_x):
        prefix = hidden_x.shape[:-1]
        mu = self.mu_head(hidden_x)
        sigma_flat = self.sigma_head(hidden_x)

        if not self.use_mixture:
            return LatentTransitionParams(mu=mu, sigma_flat=sigma_flat)

        mu = mu.reshape(*prefix, self.gmm_components, self.latent_dim)
        sigma_flat = sigma_flat.reshape(*prefix, self.gmm_components, self.sigma_dim)
        mixture_logits = self.logits_head(hidden_x)
        return LatentTransitionParams(
            mu=mu,
            sigma_flat=sigma_flat,
            mixture_logits=mixture_logits,
        )


# =========================================================================
#  Conditional NSF head (hand-rolled, compile-native)
# =========================================================================

class NSFTransitionHead(nn.Module):
    """Conditional Neural Spline Flow for per-step innovation sampling.

    Produces (mu, innovation_sample) where innovation_sample is drawn from
    a conditional coupling-based RQ-NSF (nbcore.nsf_flow) conditioned
    on the encoder hidden.

    Consumed by OUFlowHead:
        z_t = z_{t-1} + theta * (mu - z_{t-1}) * dt + sqrt(dt) * innovation

    Earlier iterations of this head wrapped zuko.flows.NSF; zuko's Distribution
    patterns are untraceable by torch._dynamo and forced the whole flow
    sampling path into eager mode, producing a ~100x wallclock regression
    inside the AR loop. The hand-rolled implementation in nsf_flow is pure
    torch tensor ops and compiles cleanly along with the rest of the model.

    Config keys (all optional):
      nsf_bins:                 4     (K bins per spline)
      nsf_transforms:           2     (number of coupling layers)
      nsf_conditioner_hidden:   128   (MLP width)
      nsf_tail_bound:           3.0   (spline operates on [-B,B]; linear tails outside)
    """

    def __init__(self, input_dim, latent_dim, config):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self._randn_innovations = config.get('randn_innovations', False)

        # MLP head producing mu from hidden
        mlp_hidden = max(input_dim // 2, latent_dim)
        self.mu_head = nn.Sequential(
            P.spectral_norm(nn.Linear(input_dim, mlp_hidden)),
            nn.SiLU(),
            nn.Linear(mlp_hidden, latent_dim),
        )

        if not self._randn_innovations:
            from nbcore.nsf_flow import ConditionalCouplingNSF
            hidden = config.get('nsf_conditioner_hidden', 128)
            bins = config.get('nsf_bins', 4)
            transforms = config.get('nsf_transforms', 2)
            tail_bound = config.get('nsf_tail_bound', 3.0)
            self.flow = ConditionalCouplingNSF(
                latent_dim=latent_dim,
                context_dim=input_dim,
                transforms=transforms,
                bins=bins,
                conditioner_hidden=hidden,
                tail_bound=tail_bound,
            )
        else:
            self.flow = None

    def forward(self, hidden_x):
        prefix = hidden_x.shape[:-1]
        mu = self.mu_head(hidden_x)

        if self._randn_innovations:
            innovation = torch.randn(*prefix, self.latent_dim,
                                     device=hidden_x.device, dtype=hidden_x.dtype)
        else:
            flat_hidden = hidden_x.reshape(-1, self.input_dim)
            innovation = self.flow.rsample(flat_hidden)
            innovation = innovation.reshape(*prefix, self.latent_dim)

        return LatentTransitionParams(mu=mu, sigma_flat=innovation)


class MuOnlyTransitionHead(nn.Module):
    """Produces only mu — for use with OURandnHead which generates its own noise."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        mlp_hidden = max(input_dim // 2, latent_dim)
        self.mu_head = nn.Sequential(
            P.spectral_norm(nn.Linear(input_dim, mlp_hidden)),
            nn.SiLU(),
            nn.Linear(mlp_hidden, latent_dim),
        )

    def forward(self, hidden_x):
        mu = self.mu_head(hidden_x)
        return LatentTransitionParams(mu=mu, sigma_flat=torch.zeros_like(mu))


# =========================================================================
#  Factory
# =========================================================================

def build_transition_head(kind, input_dim, latent_dim, noise_rank, config):
    if kind == 'mlp_gauss':
        return MLPGaussTransitionHead(
            input_dim=input_dim,
            latent_dim=latent_dim,
            noise_rank=noise_rank,
            latent_type=config.get('latent_type', 'brownian'),
            gmm_components=config.get('gmm_components', 4),
            sigma_layout=config.get('sigma_layout', 'cholesky'),
        )
    if kind == 'nsf':
        return NSFTransitionHead(input_dim=input_dim, latent_dim=latent_dim, config=config)
    if kind == 'mu_only':
        return MuOnlyTransitionHead(input_dim=input_dim, latent_dim=latent_dim)
    if kind == 'kan':
        raise NotImplementedError("KAN transition head not yet implemented.")
    raise ValueError(f"Unknown transition_head_type={kind!r}")
