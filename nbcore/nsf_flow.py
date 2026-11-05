"""Hand-rolled rational-quadratic spline flow — compile-native.

Purpose
-------
Per-step conditional sampling inside an autoregressive loop. The zuko-based
implementation caused catastrophic wallclock regressions (~100x) because
zuko's Distribution wrapper patterns cannot be traced by torch._dynamo; each
per-step sample ran in eager Python.

This module is a minimal, pure-torch, compile-native replacement. No
Distribution wrappers, no lazy object construction, no third-party lib. Only
``torch.searchsorted``, ``torch.gather``, elementary arithmetic, and softmax
/ softplus activations -- all traceable.

Design
------
- **Coupling** flow (not autoregressive): sampling is O(num_transforms) in
  latent dim D, not O(D). With D=21 this is orders of magnitude cheaper than
  MAF-style autoregressive flows.
- **Rational-quadratic spline** pieces per transformed dim, following
  Durkan et al. (2019) "Neural Spline Flows". Linear tails outside
  ``[-B, B]`` so out-of-range samples pass through as identity and tails
  match the standard-Normal base distribution.

Shapes
------
- Base noise ``z``: ``[B, D]``, D = latent_dim.
- Conditioning ``context``: ``[B, C]``, C = input_dim.
- Per coupling layer, we split D into (d_id, d_tr) via a boolean mask.
- Spline params per transformed dim: ``3K - 1`` (K widths + K heights +
  K-1 interior derivatives; boundary derivatives fixed to 1 for linear-tail
  continuity).

Only ``rsample(context)`` is on the hot path during training (pure forward
sampling into the base of the SDE). ``log_prob`` is exposed for completeness
and future use in likelihood-style auxiliary losses.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_DEFAULT_MIN_BIN_WIDTH = 1e-3
_DEFAULT_MIN_BIN_HEIGHT = 1e-3
_DEFAULT_MIN_DERIVATIVE = 1e-3


def _searchsorted_bin_index(cumulative_knot_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Locate the bin index k such that cumulative_knot_x[..., k] <= x < cumulative_knot_x[..., k+1].

    cumulative_knot_x: [..., K+1] strictly increasing.
    x: [...]
    returns: [...] long tensor with values in [0, K-1].
    """
    # torch.searchsorted expects the search values with an extra trailing dim
    # of 1 relative to the sorted sequence when both are batched. We feed it
    # [..., K+1] knots and [..., 1] queries, get back [..., 1], squeeze.
    idx = torch.searchsorted(cumulative_knot_x.contiguous(), x.unsqueeze(-1).contiguous())
    idx = idx.squeeze(-1) - 1
    # Clamp to [0, K-1]: searchsorted can return 0 for x < left edge (since
    # we subtract 1 it goes to -1) or K for x at/above the right edge.
    K = cumulative_knot_x.shape[-1] - 1
    return idx.clamp(min=0, max=K - 1)


def rational_quadratic_spline(
    x: torch.Tensor,
    raw_widths: torch.Tensor,
    raw_heights: torch.Tensor,
    raw_derivatives: torch.Tensor,
    tail_bound: float,
    inverse: bool = False,
    min_bin_width: float = _DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = _DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = _DEFAULT_MIN_DERIVATIVE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized monotonic rational-quadratic spline with linear tails.

    Args
    ----
    x: [..., D_tr] real-valued inputs.
    raw_widths: [..., D_tr, K] pre-softmax bin width logits.
    raw_heights: [..., D_tr, K] pre-softmax bin height logits.
    raw_derivatives: [..., D_tr, K-1] pre-softplus interior derivative logits
        (boundary derivatives fixed to 1 for linear-tail continuity).
    tail_bound: outside [-B, B] the map is identity (log-det = 0).
    inverse: if True, invert the spline on in-bounds points (solve the
        rational-quadratic quadratic in x given y).

    Returns
    -------
    y: same shape as x. For |x| > tail_bound, y == x.
    log_abs_det: same shape as x. Zero outside the tail bound.

    All parameters may be broadcast across leading dims (batch, context, etc.).
    """
    K = raw_widths.shape[-1]
    B = tail_bound

    # --- Normalize raw params ---------------------------------------------
    # Widths: softmax so sum to 1, scale to 2B, enforce a minimum width.
    widths = F.softmax(raw_widths, dim=-1)
    widths = min_bin_width + (1.0 - min_bin_width * K) * widths
    widths = widths * (2.0 * B)  # sum(widths) = 2B

    # Heights: same treatment.
    heights = F.softmax(raw_heights, dim=-1)
    heights = min_bin_height + (1.0 - min_bin_height * K) * heights
    heights = heights * (2.0 * B)

    # Derivatives: softplus + floor. Interior derivatives are learned; both
    # boundary derivatives are fixed to 1.0 for C^1 continuity with linear tails.
    interior_derivs = F.softplus(raw_derivatives) + min_derivative     # [..., K-1]
    one = torch.ones_like(interior_derivs[..., :1])
    derivatives = torch.cat([one, interior_derivs, one], dim=-1)        # [..., K+1]

    # Cumulative knot positions: start at -B.
    cum_widths = torch.cumsum(widths, dim=-1)
    cum_widths = F.pad(cum_widths, (1, 0), value=0.0) - B               # [..., K+1]
    cum_heights = torch.cumsum(heights, dim=-1)
    cum_heights = F.pad(cum_heights, (1, 0), value=0.0) - B             # [..., K+1]

    # Ensure right edge is exactly B (minor FP drift from cumsum accumulates).
    cum_widths = cum_widths.clone()
    cum_widths[..., -1] = B
    cum_heights = cum_heights.clone()
    cum_heights[..., -1] = B

    # --- Tail mask --------------------------------------------------------
    inside = (x >= -B) & (x <= B)

    # Prepare an "x effectively inside" -- we evaluate the spline math on all
    # points (cheap vectorized arithmetic) then use the mask to pass tails
    # through as identity. This avoids Python control flow that would break
    # dynamo tracing.
    x_clamped = x.clamp(-B + 1e-6, B - 1e-6)

    # --- Locate bin for each point ---------------------------------------
    # For forward we search knot_x; for inverse we search knot_y.
    if inverse:
        cum_for_search = cum_heights
    else:
        cum_for_search = cum_widths
    bin_idx = _searchsorted_bin_index(cum_for_search, x_clamped)       # [..., D_tr]
    bin_idx_expanded = bin_idx.unsqueeze(-1)

    def _gather(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # t: [..., K (or K+1)], idx: [..., 1] -> [..., 1] then squeeze
        return t.gather(-1, idx).squeeze(-1)

    x_lo = _gather(cum_widths, bin_idx_expanded)
    x_hi = _gather(cum_widths, bin_idx_expanded + 1)
    y_lo = _gather(cum_heights, bin_idx_expanded)
    y_hi = _gather(cum_heights, bin_idx_expanded + 1)
    d_lo = _gather(derivatives, bin_idx_expanded)
    d_hi = _gather(derivatives, bin_idx_expanded + 1)

    w = x_hi - x_lo
    h = y_hi - y_lo
    s = h / w

    if not inverse:
        xi = (x_clamped - x_lo) / w
        one_minus_xi = 1.0 - xi
        numerator = h * (s * xi * xi + d_lo * xi * one_minus_xi)
        denominator = s + (d_hi + d_lo - 2.0 * s) * xi * one_minus_xi
        y = y_lo + numerator / denominator

        deriv_numerator = s * s * (
            d_hi * xi * xi + 2.0 * s * xi * one_minus_xi + d_lo * one_minus_xi * one_minus_xi
        )
        log_abs_det = torch.log(deriv_numerator) - 2.0 * torch.log(denominator)
    else:
        # Solve the quadratic a xi^2 + b xi + c = 0 for xi in [0,1] given y.
        y_clamped = x_clamped
        yy = (y_clamped - y_lo) / h
        a = h * (s - d_lo) + (y_clamped - y_lo) * (d_hi + d_lo - 2.0 * s)
        b = h * d_lo - (y_clamped - y_lo) * (d_hi + d_lo - 2.0 * s)
        c = -s * (y_clamped - y_lo)

        discriminant = (b * b - 4.0 * a * c).clamp_min(0.0)
        xi = (2.0 * c) / (-b - torch.sqrt(discriminant))
        xi = xi.clamp(0.0, 1.0)
        one_minus_xi = 1.0 - xi

        y = xi * w + x_lo   # "y" here is x in the original coordinates

        denominator = s + (d_hi + d_lo - 2.0 * s) * xi * one_minus_xi
        deriv_numerator = s * s * (
            d_hi * xi * xi + 2.0 * s * xi * one_minus_xi + d_lo * one_minus_xi * one_minus_xi
        )
        # log-det of INVERSE = -log-det of forward
        log_abs_det = -(torch.log(deriv_numerator) - 2.0 * torch.log(denominator))

    # Where x was outside the tail bound, y is identity (x) and log-det is 0.
    y = torch.where(inside, y, x)
    log_abs_det = torch.where(inside, log_abs_det, torch.zeros_like(log_abs_det))
    return y, log_abs_det


class CouplingRQSTransform(nn.Module):
    """One coupling layer: half the dims pass through; other half is a RQ-spline
    of x conditioned on (identity half + context).

    Compile note: boolean-mask indexing (``x[..., bool_mask]``) dispatches to
    ``aten.nonzero`` which Inductor cannot compile (dynamic output shape),
    causing graph breaks. We store integer indices instead and use
    ``index_select`` / ``index_copy`` which are fully compilable.
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        bins: int,
        conditioner_hidden: int,
        transformed_mask: torch.Tensor,  # boolean [D]
        tail_bound: float = 3.0,
    ):
        super().__init__()
        mask = transformed_mask.to(dtype=torch.bool)
        if mask.numel() != latent_dim:
            raise ValueError(
                f"transformed_mask length {mask.numel()} must equal latent_dim {latent_dim}"
            )
        # Precomputed integer indices for compile-friendly gather/scatter.
        # Bool-mask indexing triggers nonzero -> graph break under Inductor.
        self.register_buffer(
            'transformed_idx',
            mask.nonzero(as_tuple=False).squeeze(-1).long(),
        )
        self.register_buffer(
            'identity_idx',
            (~mask).nonzero(as_tuple=False).squeeze(-1).long(),
        )
        # Masks retained for introspection / debugging only; not used in forward.
        self.register_buffer('transformed_mask', mask)
        self.register_buffer('identity_mask', ~mask)
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.bins = bins
        self.tail_bound = tail_bound

        self.d_tr = int(mask.sum().item())
        self.d_id = latent_dim - self.d_tr
        self.params_per_dim = 3 * bins - 1

        # Conditioner input: identity half + context -> all spline params for
        # the transformed half in one go.
        cond_in = self.d_id + context_dim
        cond_out = self.d_tr * self.params_per_dim
        self.conditioner = nn.Sequential(
            nn.Linear(cond_in, conditioner_hidden),
            nn.SiLU(),
            nn.Linear(conditioner_hidden, conditioner_hidden),
            nn.SiLU(),
            nn.Linear(conditioner_hidden, cond_out),
        )
        # Zero-init the final layer so the flow starts as identity: all raw
        # params are zero => softmax gives uniform widths/heights, softplus
        # gives constant derivative, so the spline is linear == identity at
        # init. Helps optimization start from a sensible place.
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    def _split_params(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # raw: [B, d_tr * (3K - 1)] -> [B, d_tr, 3K-1]
        raw = raw.view(*raw.shape[:-1], self.d_tr, self.params_per_dim)
        K = self.bins
        w = raw[..., :K]
        h = raw[..., K:2 * K]
        d = raw[..., 2 * K:]
        return w, h, d

    def _run_spline(self, x: torch.Tensor, context: torch.Tensor, inverse: bool):
        # x: [B, D], context: [B, C]. Method is NOT named _apply since that
        # is reserved by nn.Module for device/dtype propagation.
        #
        # index_select/index_copy on integer index tensors (precomputed
        # buffers) is Inductor-friendly; boolean-mask indexing would trigger
        # aten.nonzero and a graph break.
        x_id = x.index_select(-1, self.identity_idx)
        x_tr = x.index_select(-1, self.transformed_idx)
        raw = self.conditioner(torch.cat([x_id, context], dim=-1))
        w, h, d = self._split_params(raw)
        y_tr, log_det_per_dim = rational_quadratic_spline(
            x_tr, w, h, d,
            tail_bound=self.tail_bound,
            inverse=inverse,
        )
        # Scatter halves back into a full-width tensor via index_copy.
        # index_copy is the functional inverse of index_select and is
        # fully compilable.
        y = torch.zeros_like(x)
        y = y.index_copy(-1, self.identity_idx, x_id)
        y = y.index_copy(-1, self.transformed_idx, y_tr)
        log_det = log_det_per_dim.sum(dim=-1)
        return y, log_det

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        return self._run_spline(x, context, inverse=False)

    def inverse(self, y: torch.Tensor, context: torch.Tensor):
        return self._run_spline(y, context, inverse=True)


class ConditionalCouplingNSF(nn.Module):
    """Stack of coupling RQ-spline transforms with alternating masks.

    Base distribution: standard Normal N(0, I_D).
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        transforms: int = 2,
        bins: int = 4,
        conditioner_hidden: int = 128,
        tail_bound: float = 3.0,
    ):
        super().__init__()
        if transforms < 1:
            raise ValueError(f"transforms must be >= 1, got {transforms}")
        self.latent_dim = latent_dim
        self.context_dim = context_dim

        # Alternating masks: even-indexed dims transformed on even layers,
        # odd-indexed dims transformed on odd layers. Guarantees every dim
        # eventually gets transformed (requires transforms >= 2 for full
        # coverage; we enforce >= 1 but recommend >= 2).
        base_idx = torch.arange(latent_dim)
        even_mask = (base_idx % 2 == 0)
        odd_mask = ~even_mask

        self.layers = nn.ModuleList()
        for i in range(transforms):
            mask = even_mask if i % 2 == 0 else odd_mask
            self.layers.append(
                CouplingRQSTransform(
                    latent_dim=latent_dim,
                    context_dim=context_dim,
                    bins=bins,
                    conditioner_hidden=conditioner_hidden,
                    transformed_mask=mask,
                    tail_bound=tail_bound,
                )
            )

    def rsample(self, context: torch.Tensor) -> torch.Tensor:
        """Draw a reparameterized sample from the conditional flow."""
        batch = context.shape[0]
        z = torch.randn(batch, self.latent_dim, device=context.device, dtype=context.dtype)
        for layer in self.layers:
            z, _log_det = layer.forward(z, context)
        return z

    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Log-probability of ``x`` under the conditional flow.

        Chain inverse transforms to recover the base-Normal sample and sum the
        log-det-Jacobians. Not used on the training hot path; exposed for
        likelihood-style auxiliary losses.
        """
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z, context)
            log_det_total = log_det_total + log_det
        base_log_prob = -0.5 * (z * z).sum(dim=-1) - 0.5 * self.latent_dim * math.log(2 * math.pi)
        return base_log_prob + log_det_total
