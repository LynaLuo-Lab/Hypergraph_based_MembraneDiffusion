"""
VACF losses.

Two complementary forms are provided:

  compute_vacf_quantile_loss  — matches the *distribution* of per-pair velocity
    cosine similarities (existing). Does not directly constrain the per-lag
    mean VACF curve.

  compute_vacf_mse_loss       — matches the per-lag *mean* normalized VACF
    curve  <v(0)·v(L)> / <|v|^2>  via squared error against the GT curve.
    This is the form whose definition matches the lab-convention diagnostic
    used in scripts/inference_visualizer.py and archive/scripts/diffusion/ablation_rounds_2026-05/analysis/diagnose_ckpt_samples.py.

  compute_vacf_spatial_l1_loss  — L1 between raw per-particle per-timestep
    VACF dot products (no sorting, no averaging). Preserves spatial and
    temporal structure so gradient flows to individual (particle, time) pairs.

All are particle-blind (pooled across particles, time origins, batch).
"""

import torch


def _vacf_from_velocity(v, corr_lag):
    """Compute VACF cosine similarities from precomputed velocity tensor.

    Args:
        v: [B, T_v, 7, 2] velocity vectors
        corr_lag: lag between velocity pairs

    Returns:
        [N] flat tensor of cosine similarities, or None if not enough timesteps.
    """
    if v.size(1) <= corr_lag:
        return None

    v1 = v[:, :-corr_lag]
    v2 = v[:, corr_lag:]

    dot = (v1 * v2).sum(dim=-1)
    norm1 = torch.sqrt((v1 ** 2).sum(dim=-1) + 1e-8)
    norm2 = torch.sqrt((v2 ** 2).sum(dim=-1) + 1e-8)
    cos_sim = dot / (norm1 * norm2)

    return cos_sim.reshape(-1)


def compute_vacf_quantile_loss(pred_xy, target_xy, quantile_loss_fn,
                               velocity_lags=None, corr_lags=None):
    """
    Quantile-match the distribution of VACF cosine similarities between
    pred and GT across multiple (velocity_lag, corr_lag) combinations.

    Args:
        pred_xy:   [B, T, 7, 2] unwrapped XY positions (predicted)
        target_xy: [B, T, 7, 2] unwrapped XY positions (ground truth)
        quantile_loss_fn: function(pred_flat, target_flat) -> scalar
                          (reuse compute_quantile_loss_batched or similar)
        velocity_lags: list of lags for defining velocities (default [1, 5])
        corr_lags: list of correlation lags (default [1, 5, 10, 25])

    Returns:
        Scalar loss (mean over all lag combinations that had enough data).
    """
    if velocity_lags is None:
        velocity_lags = [1, 5, 10, 25, 50, 100, 200]
    if corr_lags is None:
        corr_lags = [1, 5, 10, 25, 50, 100, 200, 300, 500, 1000, 2000, 3000]

    device = pred_xy.device
    loss = torch.tensor(0.0, device=device)
    count = 0

    # Precompute velocity tensors for each v_lag (avoids redundant recomputation)
    pred_vels = {vl: pred_xy[:, vl:] - pred_xy[:, :-vl]
                 for vl in velocity_lags if pred_xy.size(1) > vl}
    gt_vels = {vl: target_xy[:, vl:] - target_xy[:, :-vl]
               for vl in velocity_lags if target_xy.size(1) > vl}

    for v_lag in velocity_lags:
        if v_lag not in pred_vels or v_lag not in gt_vels:
            continue
        for c_lag in corr_lags:
            pred_vacf = _vacf_from_velocity(pred_vels[v_lag], c_lag)
            gt_vacf = _vacf_from_velocity(gt_vels[v_lag], c_lag)

            if pred_vacf is None or gt_vacf is None:
                continue
            if pred_vacf.numel() < 2 or gt_vacf.numel() < 2:
                continue

            p_sorted, _ = torch.sort(pred_vacf)
            t_sorted, _ = torch.sort(gt_vacf)

            if p_sorted.shape[0] != t_sorted.shape[0]:
                p_sorted = torch.nn.functional.interpolate(
                    p_sorted.view(1, 1, -1),
                    size=t_sorted.shape[0],
                    mode='linear',
                    align_corners=False,
                ).view(-1)

            loss = loss + (p_sorted - t_sorted).abs().mean()
            count += 1

    return loss / max(count, 1)


def compute_latent_corr_loss(z_seq, target_curve=None, lags=None,
                             kernel='tanh', scale=0.05):
    """Direct latent-velocity autocorrelation loss.

    Computes per-lag autocorrelation of dz_t = z_t - z_{t-1} pooled over
    batch x time-origin, then applies a kernel against a target.

    z_seq:        [B, T, latent_dim]  — stacked latent states from the rollout
    target_curve: 1D iterable of target autocorrelation values, one per lag
                  (typically the GT position-VACF curve at the same lags).
                  If None, defaults to GT position-VACF approximated for the
                  7-piezo system: [0.224, 0.129, 0.053, 0.023, 0.005].
    lags:         list of correlation lags (default [1, 2, 5, 10, 25])
    kernel:       'tanh'        — two-piece kernel:
                                    (1 - tanh(corr/scale))
                                    + tanh(relu(corr - target)/scale)
                                  High at corr<=0, low for 0<corr<target,
                                  ramps up past target. Anti-target shape:
                                  punishes decorr + overshoot, no opinion in
                                  the good zone.
                  'mse_target'  — (corr - target)^2 per lag, mean.
                                  Standard target-matching MSE.
    scale:        kernel width (tanh) — natural unit is 'a fraction of the
                  smallest meaningful target value'. 0.05 means the loss
                  saturates within ~0.05 of zero in the good direction.

    Returns: scalar loss summed/meaned over lags with enough horizon.
    """
    import torch.nn.functional as F  # local import: lazy
    if lags is None:
        lags = [1, 2, 5, 10, 25]
    if target_curve is None:
        # GT position-VACF approximation for 7piezo at lags [1,2,5,10,25]
        default = {1: 0.224, 2: 0.129, 5: 0.053, 10: 0.023, 25: 0.005}
        target_curve = [default.get(L, 0.0) for L in lags]

    device = z_seq.device
    # Latent velocity: dz_t = z_{t+1} - z_t  (shape [B, T-1, latent_dim])
    dz = z_seq[:, 1:] - z_seq[:, :-1]
    if dz.size(1) < 2:
        return torch.tensor(0.0, device=device)

    # Pooled |dz|^2 normalizer (mirrors the position-VACF formulation)
    dz_norm_sq = (dz ** 2).sum(dim=-1).mean().clamp_min(1e-8)

    losses = []
    for L, tgt in zip(lags, target_curve):
        if dz.size(1) <= L:
            continue
        dot = (dz[:, :-L] * dz[:, L:]).sum(dim=-1).mean()
        corr = dot / dz_norm_sq
        tgt_t = torch.as_tensor(tgt, dtype=corr.dtype, device=device)

        if kernel == 'tanh':
            below = 1.0 - torch.tanh(corr / scale)
            over = torch.tanh(F.relu(corr - tgt_t) / scale)
            losses.append(below + over)
        elif kernel == 'mse_target':
            losses.append((corr - tgt_t) ** 2)
        else:
            raise ValueError(f"Unknown kernel={kernel!r}; expected 'tanh' or 'mse_target'.")

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def compute_vacf_mse_loss(pred_xy, target_xy, lags=None, velocity_lag=1):
    """Match the per-lag *mean* normalized VACF curve between pred and GT.

    For each correlation lag L in `lags`, compute
        vacf(L) = <v(0) · v(L)> / <|v|^2>
    where v(t) = xy(t+velocity_lag) - xy(t), pooled over batch × time-origin × particle.
    Then return MSE between the pred and GT curve.

    Definition matches archive/scripts/diffusion/ablation_rounds_2026-05/analysis/diagnose_ckpt_samples.py:vacf_curve so the loss
    directly targets the diagnostic metric.

    Args:
        pred_xy:   [B, T, 7, 2] unwrapped XY positions (predicted).
        target_xy: [B, T, 7, 2] unwrapped XY positions (ground truth).
        lags:      correlation lags to match. Default [1, 5, 10, 25, 50, 100, 200].
        velocity_lag: step size used to define velocity (default 1).

    Returns:
        Scalar MSE summed over lags (only those with enough horizon for both).
    """
    if lags is None:
        lags = [1, 5, 10, 25, 50, 100, 200]

    device = pred_xy.device
    # velocity
    if pred_xy.size(1) <= velocity_lag or target_xy.size(1) <= velocity_lag:
        return torch.tensor(0.0, device=device)
    pv = pred_xy[:, velocity_lag:] - pred_xy[:, :-velocity_lag]
    tv = target_xy[:, velocity_lag:] - target_xy[:, :-velocity_lag]

    # <|v|^2> pooled
    p_v2_mean = (pv ** 2).sum(dim=-1).mean().clamp_min(1e-8)
    t_v2_mean = (tv ** 2).sum(dim=-1).mean().clamp_min(1e-8)

    loss = torch.tensor(0.0, device=device)
    count = 0
    for L in lags:
        if pv.size(1) <= L:
            continue
        p_dot = (pv[:, :pv.size(1) - L] * pv[:, L:]).sum(dim=-1).mean()
        t_dot = (tv[:, :tv.size(1) - L] * tv[:, L:]).sum(dim=-1).mean()
        p_vacf = p_dot / p_v2_mean
        t_vacf = t_dot / t_v2_mean
        loss = loss + (p_vacf - t_vacf) ** 2
        count += 1
    return loss / max(count, 1)


def compute_vacf_spatial_l1_loss(pred_xy, target_xy, lags=None, velocity_lag=1):
    """L1 between raw per-(particle, time) VACF dot products, no sorting.

    For each lag L, computes v(t)·v(t+L) for every (batch, time, particle)
    and takes L1 between pred and target. Gradient flows to individual
    spatiotemporal positions.

    Args:
        pred_xy:   [B, T, 7, 2] unwrapped XY positions (predicted).
        target_xy: [B, T, 7, 2] unwrapped XY positions (ground truth).
        lags:      correlation lags. Default [1, 5, 10, 25, 50, 100].
        velocity_lag: step size for velocity definition (default 1).
    """
    if lags is None:
        lags = [1, 5, 10, 25, 50, 100]

    device = pred_xy.device
    if pred_xy.size(1) <= velocity_lag or target_xy.size(1) <= velocity_lag:
        return torch.tensor(0.0, device=device)

    pv = pred_xy[:, velocity_lag:] - pred_xy[:, :-velocity_lag]  # [B, T-1, 7, 2]
    tv = target_xy[:, velocity_lag:] - target_xy[:, :-velocity_lag]

    loss = torch.tensor(0.0, device=device)
    count = 0
    for L in lags:
        if pv.size(1) <= L:
            continue
        p_dot = (pv[:, :-L] * pv[:, L:]).sum(dim=-1)  # [B, T-1-L, 7]
        t_dot = (tv[:, :-L] * tv[:, L:]).sum(dim=-1)
        loss = loss + (p_dot - t_dot).abs().mean()
        count += 1
    return loss / max(count, 1)
