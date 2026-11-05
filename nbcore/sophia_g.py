"""Triton-accelerated Sophia-G optimizer.

Sophia-G uses a diagonal Hessian estimate (EMA of squared gradients) to
adaptively clip per-parameter updates.  High-curvature parameters (e.g.
attention weights) automatically get smaller effective step sizes; flat
parameters (e.g. MLP weights) get larger ones.

The Triton kernels fuse the hessian update and the momentum+clip step into
single GPU passes, making it competitive with fused AdamW.

Reference: Liu et al., "Sophia: A Scalable Stochastic Second-Order
Optimizer for Language Model Pre-training", 2023.
"""

import torch
from torch.optim.optimizer import Optimizer

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _update_hessian_kernel(
    hessian_ptr,
    grad_ptr,
    beta2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    hessian = tl.load(hessian_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)

    hessian_fp32 = hessian.to(tl.float32)
    grad_fp32 = grad.to(tl.float32)

    new_hessian = beta2 * hessian_fp32 + (1.0 - beta2) * grad_fp32 * grad_fp32

    tl.store(hessian_ptr + offsets, new_hessian.to(hessian.dtype), mask=mask)


@triton.jit
def _step_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    hessian_ptr,
    lr,
    beta1,
    rho,
    bs,
    weight_decay,
    eps,
    n_elements,
    p_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    hessian = tl.load(hessian_ptr + offsets, mask=mask)

    p_fp32 = p.to(tl.float32)
    grad_fp32 = grad.to(tl.float32)
    exp_avg_fp32 = exp_avg.to(tl.float32)
    hessian_fp32 = hessian.to(tl.float32)

    new_exp_avg = beta1 * exp_avg_fp32 + (1.0 - beta1) * grad_fp32

    p_decayed = p_fp32 * (1.0 - lr * weight_decay)

    denominator = tl.maximum(rho * bs * hessian_fp32, eps)
    ratio = tl.abs(new_exp_avg) / denominator
    clamped_ratio = tl.minimum(ratio, 1.0)

    sign_new_exp_avg = tl.where(
        new_exp_avg > 0, 1.0,
        tl.where(new_exp_avg < 0, -1.0, 0.0),
    )
    update = lr * sign_new_exp_avg * clamped_ratio
    new_p = p_decayed - update

    tl.store(p_ptr + offsets, new_p.to(p_dtype), mask=mask)
    tl.store(exp_avg_ptr + offsets, new_exp_avg.to(exp_avg.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


class SophiaG(Optimizer):
    """Sophia-G with fused Triton kernels.

    Args:
        params: Iterable of parameters or param-group dicts.
        lr: Learning rate (default: 1e-4).
        betas: EMA coefficients for momentum and hessian (default: (0.965, 0.99)).
        rho: Scales the Hessian in the clip denominator:
            update = lr * sign(m) * min(|m| / max(rho*bs*h, eps), 1)
            so LARGER rho = larger denominator = SMALLER, more
            curvature-scaled updates. (This read 'smaller = more
            conservative' until 2026-09-16, which is backwards.)
            MEASURED on the real membrane model at eff bs 64, as the
            fraction of coordinates pinned at the clip cap -- where the
            update degenerates to a fixed lr*sign(m) step with no
            curvature information at all:
                rho 0.0159 -> 56.3%   (what the sweeps ship)
                rho 0.04   -> 48.7%   (this default)
                rho 0.10   -> 40.9%
                rho 0.30   -> 31.0%
            (default: 0.04).
        weight_decay: Decoupled weight decay (default: 0.1).
        bs: Effective batch size (batch_size * accumulate_grad_batches).
        eps: Epsilon for numerical stability (default: 1e-15).
        max_consecutive_skips / max_skips: a step whose gradients are not all
            finite is skipped with state untouched; this many in a row, or in
            total, raises instead, so a run that keeps producing them dies loudly
            rather than limping on untrained (defaults 3 / 10).
    """

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.965, 0.99),
        rho=0.04,
        weight_decay=0.1,
        *,
        bs: int,
        eps: float = 1e-15,
        max_consecutive_skips: int = 3,
        max_skips: int = 10,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if rho < 0.0:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if bs <= 0:
            raise ValueError(f"Invalid batch size (bs): {bs}")

        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay,
                        eps=eps, bs=bs)
        super().__init__(params, defaults)
        # Steps refused because a gradient was non-finite. One NaN/inf gradient
        # written into exp_avg/hessian never decays out: sign(NaN) is 0 in
        # _step_kernel, so every later update is weight decay only and training
        # freezes without an error (all six gm1010 S-proj runs, 2026-09-24).
        self.skipped_steps = 0
        self.consecutive_skips = 0
        self.max_consecutive_skips = max_consecutive_skips
        self.max_skips = max_skips

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grads = [p.grad for g in self.param_groups for p in g['params'] if p.grad is not None]
        if grads and not torch.stack([torch.isfinite(g).all() for g in grads]).all():
            self.skipped_steps += 1
            self.consecutive_skips += 1
            print(f"SophiaG: skipped a step with non-finite gradients "
                  f"({self.skipped_steps} so far, {self.consecutive_skips} in a row)", flush=True)
            if (self.consecutive_skips >= self.max_consecutive_skips
                    or self.skipped_steps >= self.max_skips):
                raise RuntimeError(
                    f"SophiaG: {self.skipped_steps} steps with non-finite gradients "
                    f"({self.consecutive_skips} consecutive); training has diverged")
            return loss
        self.consecutive_skips = 0

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            rho = group['rho']
            eps = group['eps']
            bs = float(group['bs'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('SophiaG does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1

                n_elements = p.numel()
                grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

                # Update hessian estimate: h = β₂·h + (1-β₂)·g²
                _update_hessian_kernel[grid](
                    state['hessian'], grad, beta2, n_elements,
                    BLOCK_SIZE=1024,
                )

                # Momentum + hessian-clipped step
                p_dtype_tl = _DTYPE_MAP.get(p.dtype, tl.float32)
                _step_kernel[grid](
                    p, grad, state['exp_avg'], state['hessian'],
                    lr, beta1, rho, bs, weight_decay, eps,
                    n_elements,
                    p_dtype=p_dtype_tl,
                    BLOCK_SIZE=1024,
                )

        return loss
