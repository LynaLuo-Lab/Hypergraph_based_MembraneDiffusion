"""Shared physics-metric core for both the in-training stability callback and the
offline resample eval, so the training OBJECTIVE is literally the eval.

Design principles (per user, 2026-06-03):
  - ABSTRACT / BEHAVIORAL only. Every metric is an aggregate statistic over the
    whole rollout (per-lag MSD/VACF/NGP, pooled van-Hove distribution, scalar
    correlations, mean magnitudes) — never a timestep-by-timestep match. The
    model must reproduce the behavior the two GT trajectories SHARE, not match
    either path.
  - NOISE-FLOOR REFERENCED. For each metric M the tolerance is how much two real
    realizations differ: floor_M = reldist(M(traj1), M(traj2)). The per-metric
    penalty is the relu excess above that floor:
        penalty_M = max(0, reldist(M(model), M(gt)) / floor_M - 1)
    -> 0 once the model's statistic sits within the traj1-traj2 spread.
  - EQUAL WEIGHT over the 10 metrics (objective = mean of penalties).

Metrics (10): MSD, NGP, vanHove, VACF, cohesion, step, alpha   (7; pair_mean
dropped as unreliable) + z_step, z_vacf, z_membrane            (3 z-specific).

All compute on GPU (torch); positions are wrapped [N, T, 7, 3]. 512x40k fits in
~10 GB so nothing is chunked.
"""
import torch

BOX = 856.196
MSD_LAGS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000]
VACF_LAGS = [1, 5, 10, 25, 50, 100, 250, 500]
ZVACF_LAGS = [1, 2, 3, 5, 10, 25, 50, 100]      # fine at short lags for the caging dip
COH_LAGS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000]
VH_CAP = 120_000
_TRI = torch.triu_indices(7, 7, offset=1)        # 21 particle pairs


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _unwrap_xy(pos_w, box):
    """pos_w [N,T,7,3] wrapped -> unwrapped XY [N,T,7,2] (min-image cumsum)."""
    xy = pos_w[..., :2]
    disp = xy[:, 1:] - xy[:, :-1]
    disp = disp - box * torch.round(disp / box)
    uw = torch.empty_like(xy)
    uw[:, 0] = xy[:, 0]
    uw[:, 1:] = xy[:, :1] + torch.cumsum(disp, dim=1)
    return uw


def _w1(a, b):
    """1-D Wasserstein-1 between two sample sets, relative to gt rms (torch)."""
    n = min(a.numel(), b.numel(), VH_CAP)
    qa = torch.quantile(a, torch.linspace(0, 1, n, device=a.device))
    qb = torch.quantile(b, torch.linspace(0, 1, n, device=b.device))
    return float((qa - qb).abs().mean() / (b.pow(2).mean().sqrt() + 1e-6))


def _pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = (x.pow(2).sum().sqrt() * y.pow(2).sum().sqrt())
    return float((x * y).sum() / (d + 1e-12))


# --------------------------------------------------------------------------- #
#  metric computation: positions -> aggregate behavioral statistics
# --------------------------------------------------------------------------- #
def compute_metrics(pos_w, box=BOX):
    """pos_w [N,T,7,3] wrapped (torch, any device) -> dict of aggregate stats.

    Stats are raw behavioral quantities; distances() converts pred/gt pairs into
    relative distances. Everything kept on pos_w.device.
    """
    dev = pos_w.device
    n_p = pos_w.shape[2]                               # particle count (7 for piezo, 2 for GM1010)
    tri = _TRI if n_p == 7 else torch.triu_indices(n_p, n_p, offset=1)
    uw = _unwrap_xy(pos_w, box)                       # [N,T,n_p,2]
    v = uw[:, 1:] - uw[:, :-1]                         # XY velocity [N,T-1,7,2]
    sp2 = v.pow(2).sum(-1)                             # XY speed^2 [N,T-1,7]
    out = {}

    # --- TAMSD per lag + alpha (XY) ---
    tamsd, vh = {}, {}
    for L in MSD_LAGS:
        if L >= uw.shape[1]:
            continue
        d = uw[:, L:] - uw[:, :-L]
        r2 = d.pow(2).sum(-1)                          # [N,T-L,7]
        tamsd[L] = float(r2.mean())
        rmag = r2.sqrt().flatten()
        if rmag.numel() > VH_CAP:
            rmag = rmag[torch.randint(0, rmag.numel(), (VH_CAP,), device=dev)]
        vh[L] = rmag
    out['tamsd'] = tamsd
    out['vh'] = vh
    lags_t = torch.tensor([float(L) for L in tamsd], device=dev)
    msd_t = torch.tensor([max(tamsd[L], 1e-9) for L in tamsd], device=dev)
    A = torch.stack([torch.log(lags_t), torch.ones_like(lags_t)], 1)
    out['alpha'] = float(torch.linalg.lstsq(A, torch.log(msd_t).unsqueeze(1)).solution[0, 0])

    # --- NGP per lag (XY, 2D) ---
    ngp = {}
    for L in MSD_LAGS:
        if L >= uw.shape[1]:
            continue
        r2 = (uw[:, L:] - uw[:, :-L]).pow(2).sum(-1)
        m2 = r2.mean(); m4 = r2.pow(2).mean()
        ngp[L] = float(m4 / (2.0 * m2.pow(2) + 1e-12) - 1.0)
    out['ngp'] = ngp

    # --- VACF per lag (XY) ---
    v2 = float(sp2.mean())
    vacf = {}
    for L in VACF_LAGS:
        if v.shape[1] <= L:
            continue
        dot = (v[:, :-L] * v[:, L:]).sum(-1)
        vacf[L] = float(dot.mean() / (v2 + 1e-12))
    out['vacf'] = vacf

    # --- step (XY mean single-step displacement) ---
    out['step'] = float(sp2.sqrt().mean())

    # --- cohesion: RMS pairwise-distance change over lag (wrapped 3D, min-image) ---
    coh = {}
    coh_acc = {L: [0.0, 0] for L in COH_LAGS}
    for a, b in zip(tri[0].tolist(), tri[1].tolist()):
        pdiff = pos_w[:, :, a, :] - pos_w[:, :, b, :]
        pdiff = pdiff - box * torch.round(pdiff / box)
        pdist = pdiff.pow(2).sum(-1).sqrt()           # [N,T]
        for L in COH_LAGS:
            if L >= pdist.shape[1]:
                continue
            dd = pdist[:, L:] - pdist[:, :-L]
            coh_acc[L][0] += float(dd.pow(2).sum()); coh_acc[L][1] += dd.numel()
    for L in COH_LAGS:
        if coh_acc[L][1]:
            coh[L] = (coh_acc[L][0] / coh_acc[L][1]) ** 0.5
    out['coh'] = coh

    # --- z metrics (all aggregate/behavioral) ---
    z = pos_w[..., 2]                                  # [N,T,7]
    dz = z[:, 1:] - z[:, :-1]
    dz = dz - box * torch.round(dz / box)             # z single-step disp [N,T-1,7]
    out['z_step'] = float(dz.abs().mean())            # mean |z-displacement|
    # z-VACF per lag (within-particle z velocity autocorrelation)
    z_v2 = float(dz.pow(2).mean())
    zvacf = {}
    for L in ZVACF_LAGS:
        if dz.shape[1] <= L:
            continue
        zvacf[L] = float((dz[:, :-L] * dz[:, L:]).mean() / (z_v2 + 1e-12))
    out['z_vacf'] = zvacf
    # z_membrane: pooled within-pair temporal corr(xy_dist, |z_i-z_j|) over time
    # (the dynamic surface-following statistic; one behavioral scalar). Vectorized
    # over ICs per pair -> only a 21-pair python loop, no per-IC loop.
    xy = pos_w[..., :2]
    corrs = []
    for a, b in zip(tri[0].tolist(), tri[1].tolist()):
        d = xy[:, :, a, :] - xy[:, :, b, :]
        d = d - box * torch.round(d / box)
        rxy = d.pow(2).sum(-1).sqrt()                 # [N,T]
        zdiff = (z[:, :, a] - z[:, :, b]).abs()       # [N,T]
        rc = rxy - rxy.mean(1, keepdim=True)
        zc = zdiff - zdiff.mean(1, keepdim=True)
        num = (rc * zc).sum(1)                         # [N] per-IC covariance
        den = rc.pow(2).sum(1).sqrt() * zc.pow(2).sum(1).sqrt()
        valid = den > 1e-9
        if valid.any():
            corrs.append(num[valid] / den[valid])
    out['z_membrane'] = float(torch.cat(corrs).mean()) if corrs else 0.0
    return out


# --------------------------------------------------------------------------- #
#  distances: model stats vs a GT reference -> per-metric relative distance
# --------------------------------------------------------------------------- #
def distances(pred, gt):
    """Per-metric RELATIVE distance between two stat-dicts (from compute_metrics).
    All abstract: per-lag aggregates, pooled van-Hove W1, scalar correlations."""
    logd = lambda a, b: abs(torch.log(torch.tensor(max(a, 1e-9) / max(b, 1e-9))).item())
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-9)
    d = {}
    d['MSD'] = sum(logd(pred['tamsd'][L], gt['tamsd'][L]) for L in gt['tamsd']) / len(gt['tamsd'])
    d['NGP'] = sum(abs(pred['ngp'][L] - gt['ngp'][L]) for L in gt['ngp']) / len(gt['ngp'])
    d['vanHove'] = sum(_w1(pred['vh'][L], gt['vh'][L]) for L in gt['vh']) / len(gt['vh'])
    d['VACF'] = sum(abs(pred['vacf'][L] - gt['vacf'][L]) for L in gt['vacf']) / len(gt['vacf'])
    d['cohesion'] = sum(logd(pred['coh'][L], gt['coh'][L]) for L in gt['coh']) / len(gt['coh'])
    d['alpha'] = rel(pred['alpha'], gt['alpha'])
    d['step'] = rel(pred['step'], gt['step'])
    d['z_step'] = rel(pred['z_step'], gt['z_step'])
    d['z_vacf'] = sum(abs(pred['z_vacf'][L] - gt['z_vacf'][L]) for L in gt['z_vacf']) / len(gt['z_vacf'])
    d['z_membrane'] = abs(pred['z_membrane'] - gt['z_membrane'])
    return d


METRICS = ['MSD', 'NGP', 'vanHove', 'VACF', 'cohesion', 'step', 'alpha',
           'z_step', 'z_vacf', 'z_membrane']


def merge_gt(s1, s2):
    """Shared-behavior reference = average/pool of the two GT trajectories' stats.

    Used as the comparison target so the model is scored against the behavior BOTH
    trajectories share (not either path). Being anywhere within the traj1-traj2
    spread then lands within ~1 floor of this midpoint -> penalty 0. This is what
    makes a noisy metric (z_membrane) a soft-but-discriminating constraint rather
    than a brittle single-trajectory match.
    """
    m = {}
    for k in ['tamsd', 'ngp', 'vacf', 'coh', 'z_vacf']:
        keys = set(s1[k]) & set(s2[k])
        m[k] = {L: 0.5 * (s1[k][L] + s2[k][L]) for L in keys}
    for k in ['alpha', 'step', 'z_step', 'z_membrane']:
        m[k] = 0.5 * (s1[k] + s2[k])
    m['vh'] = {L: torch.cat([s1['vh'][L], s2['vh'][L]])
               for L in (set(s1['vh']) & set(s2['vh']))}
    return m


def noise_floor(stats1, stats2):
    """floor_M = relative distance between the two GT trajectories' stats."""
    return distances(stats1, stats2)


def objective(model_stats, gt_stats, floor, eps=1e-4, metrics=None):
    """Equal-weighted mean over `metrics` of max(0, reldist(model,gt)/floor - 1)
    (the relu excess above the two-trajectory noise floor; 0 once the model is
    within the spread). `metrics=None` -> all 10 METRICS (default, z-inclusive).
    Pass a subset (e.g. the 6 lat6 metrics) for a lateral-only objective; each
    listed metric is weighted equally (1/len). `dist`/`pens` are still returned
    for every requested metric so per-metric logging is unaffected."""
    use = METRICS if metrics is None else list(metrics)
    dist = distances(model_stats, gt_stats)
    pens = {m: max(0.0, dist[m] / max(floor[m], eps) - 1.0) for m in use}
    return float(sum(pens.values()) / len(use)), dist, pens
