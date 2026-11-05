import math
import json
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os

from data.paths import resolve, resolve_all
from membrane import physics_metrics as pm


# ------------------------------------------------------------------
# Batched free functions
# ------------------------------------------------------------------

def calculate_wasserstein_rel(pred_dist, target_dist):
    """Wasserstein-1 between sorted 1D distributions.  [B, N] → [B]."""
    p_sorted, _ = torch.sort(pred_dist, dim=-1)
    t_sorted, _ = torch.sort(target_dist, dim=-1)
    if p_sorted.size(-1) != t_sorted.size(-1):
        t_sorted = F.interpolate(
            t_sorted.unsqueeze(1), size=p_sorted.size(-1), mode='linear'
        ).squeeze(1)
    return (p_sorted - t_sorted).abs().mean(dim=-1)


def compute_diffusion_exponent(tamsd_dict):
    """Fit log(MSD) vs log(lag) → alpha.  {lag: [B]} → [B]."""
    lags = sorted(tamsd_dict.keys())
    if len(lags) < 2:
        first = next(iter(tamsd_dict.values()))
        return torch.ones(first.size(0), dtype=torch.float64, device=first.device)

    first = next(iter(tamsd_dict.values()))
    device, B = first.device, first.size(0)

    log_lags = torch.tensor([math.log(l) for l in lags], dtype=torch.float64, device=device)
    log_msd = torch.stack([tamsd_dict[lag] for lag in lags]).clamp_min(1e-8).double().log()

    if not torch.isfinite(log_msd).all() or not torch.isfinite(log_lags).all():
        return torch.ones(B, dtype=torch.float64, device=device)

    A = torch.stack([log_lags, torch.ones_like(log_lags)], dim=1)  # [n_lags, 2]
    result = torch.linalg.lstsq(A, log_msd)
    alpha = result.solution[0]  # [B]
    return torch.where(torch.isfinite(alpha), alpha, torch.ones_like(alpha))


# ------------------------------------------------------------------
# Callback
# ------------------------------------------------------------------

LAGS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000]
VACF_LAGS = [1, 5, 10, 25, 50, 100, 250, 500]  # lab-convention; covers the τ where GT decays
MSD_RATIO_LAG = 500  # canonical "long-time MSD" reference lag


class LongHorizonStabilityCallback(L.Callback):
    # Cache parsed CSVs across trials in the same process to avoid re-parsing
    # a 12k-row CSV every trial.
    _val_data_cache: dict = {}

    def __init__(self, check_interval=1, n_samples=512, chunk_size=1000,
                 csv_path='data/7piezo_traj2.csv', rollout_steps=40000,
                 calibration_path='data/shared/calibration/stab_calibration.json',
                 gt_paths=('data/7piezo_traj1.csv', 'data/7piezo_traj2.csv'),
                 ic_seed=1234, objective_metrics=None):
        super().__init__()
        self.check_interval = check_interval
        self.n_samples = n_samples
        self.chunk_size = chunk_size
        self.csv_path = resolve(csv_path)
        self.rollout_steps = rollout_steps
        self.calibration_path = resolve(calibration_path)
        # Metric subset for stab/reported_objective + stab/stability_objective.
        # None -> all 10 METRICS (z-inclusive, default). A subset (e.g. the 6
        # lat6 metrics) gives a lateral-only objective, each weighted equally.
        self.objective_metrics = objective_metrics
        self.val_data = None
        # --- physics objective (behavioral, both-trajectory; == resample eval) ---
        self.gt_paths = resolve_all(gt_paths)
        self.ic_seed = ic_seed
        self._gt_ready = False
        self.merged_gt = None          # shared-behavior reference (avg of traj1/2)
        self.floor = None              # per-metric traj1-vs-traj2 noise floor
        self.ic_pool = None            # fixed 512 ICs sampled from both trajectories
        # Weights calibrated from round-1 + round-2 ablation comparison
        # (2026-05-24). Prioritizes metrics that actually discriminate between
        # good and bad physics: α, MSD magnitude, pairwise structure.
        # VACF is near-constant across all current models (all produce ≈0),
        # so it's downweighted to avoid injecting noise into the objective.
        self.weights = {
            'tamsd': 3.0,      # relative alpha error — top discriminator
            'msd_log': 2.0,    # |log(MSD@500_pred/GT)| — long-horizon magnitude
            'pair': 2.0,       # pairwise distance distribution — structural
            'ngp': 1.0,        # non-Gaussian parameter — displacement shape
            'van_hove': 0.5,   # displacement PDF Wasserstein — partially redundant w/ MSD+NGP
            'vel': 0.3,        # step-size magnitude — short-term sanity
            'vacf': 0.5,       # VACF curve MSE — currently noise, track for future
        }
        self._calibration_loaded = False

    def setup(self, trainer, pl_module, stage=None):
        self.n_particles = getattr(pl_module, 'n_particles', 7)
        self.n_coords = self.n_particles * 3
        self.n_pairs = self.n_particles * (self.n_particles - 1) // 2

        single_path = getattr(trainer.datamodule, 'data_path', '')
        if single_path and os.path.exists(single_path):
            csv_path = single_path
        else:
            dataset_paths = getattr(trainer.datamodule, 'data_paths', [])
            csv_path = dataset_paths[0] if dataset_paths else self.csv_path

        if os.path.exists(csv_path):
            cached = LongHorizonStabilityCallback._val_data_cache.get(csv_path)
            if cached is None:
                raw_data = pd.read_csv(csv_path, sep=None, engine='python', header=None) \
                    .apply(pd.to_numeric, errors='coerce').dropna().values
                raw_tensor = torch.tensor(raw_data).float()
                cached = raw_tensor[:, 1:]
                LongHorizonStabilityCallback._val_data_cache[csv_path] = cached
            self.val_data = cached
        else:
            print(f"Warning: CSV not found at {csv_path}. Stability callback will be skipped.")

        # Calibration JSON: variance-normalized per-metric weights derived
        # offline from a reference set (see scripts/diffusion/train/calibrate_stab_weights.py).
        # When present, every term contributes ~1 unit of variance across the
        # observed model spectrum, so the composite is interpretable
        # ("0 ≈ GT, larger = farther from GT in normalized units").
        if not self._calibration_loaded and os.path.exists(self.calibration_path):
            try:
                with open(self.calibration_path) as f:
                    payload = json.load(f)
                if 'weights' in payload:
                    self.weights.update({k: float(v) for k, v in payload['weights'].items()})
                    print(f"[stab callback] loaded variance-normalized weights "
                          f"from {self.calibration_path}: {self.weights}")
            except Exception as e:
                print(f"[stab callback] WARNING: failed to read {self.calibration_path}: {e}")
            self._calibration_loaded = True

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _nan_penalty(x, penalty=10.0):
        """Replace NaN/Inf with penalty value.  [B] → [B]."""
        return torch.where(torch.isfinite(x), x, torch.full_like(x, penalty))

    @staticmethod
    def calc_relative_error(pred, gt, epsilon=1e-6):
        return torch.abs(pred - gt) / (torch.abs(gt) + epsilon)

    # ------------------------------------------------------------------
    # XY helpers (PBC-aware, batched)
    # ------------------------------------------------------------------
    def _unwrap_xy(self, positions, box_size):
        """[B, T, n_coords] → [B, T, n_particles, 2] unwrapped XY."""
        pos_nx3 = positions[:, :, :self.n_coords].reshape(
            positions.size(0), positions.size(1), self.n_particles, 3)
        xy = pos_nx3[:, :, :, :2]
        single_disp = xy[:, 1:] - xy[:, :-1]
        single_disp = single_disp - box_size * torch.round(single_disp / box_size)
        unwrapped = torch.zeros_like(xy)
        unwrapped[:, 0] = xy[:, 0]
        unwrapped[:, 1:] = xy[:, 0:1] + torch.cumsum(single_disp, dim=1)
        return unwrapped

    def _xy_displacements(self, positions, lag, box_size, unwrapped=None):
        """[B, T, 21] → [B, T-lag, 7, 2] unwrapped XY displacements."""
        if unwrapped is None:
            unwrapped = self._unwrap_xy(positions, box_size)
        return unwrapped[:, lag:] - unwrapped[:, :-lag]

    # ------------------------------------------------------------------
    # Metric 1: TAMSD + diffusion exponent (global, XY)
    # ------------------------------------------------------------------
    def compute_tamsd(self, positions, lags, box_size, unwrapped=None):
        """Returns {lag: [B] tensor}."""
        tamsd = {}
        for lag in lags:
            disp = self._xy_displacements(positions, lag, box_size, unwrapped=unwrapped)
            if disp.size(1) == 0:
                continue
            r_sq = (disp ** 2).sum(dim=-1)  # [B, T-lag, 7]
            tamsd[lag] = r_sq.mean(dim=-1).mean(dim=-1)  # [B]
        return tamsd

    def compute_tamsd_error(self, pred_pos, gt_pos, box_size,
                            pred_unwrapped=None, gt_unwrapped=None):
        pred_tamsd = self.compute_tamsd(pred_pos, LAGS, box_size, unwrapped=pred_unwrapped)
        gt_tamsd = self.compute_tamsd(gt_pos, LAGS, box_size, unwrapped=gt_unwrapped)
        alpha_pred = compute_diffusion_exponent(pred_tamsd)
        alpha_gt = compute_diffusion_exponent(gt_tamsd)
        alpha_err = (alpha_pred - alpha_gt).abs() / (alpha_gt.abs() + 1e-6)
        return alpha_err, alpha_pred, alpha_gt, pred_tamsd, gt_tamsd

    # ------------------------------------------------------------------
    # Metric 2: Van Hove self-correlation (global, XY)
    # ------------------------------------------------------------------
    def compute_van_hove_error(self, pred_pos, gt_pos, box_size, lag,
                               pred_unwrapped=None, gt_unwrapped=None):
        pred_disp = self._xy_displacements(pred_pos, lag, box_size, unwrapped=pred_unwrapped)
        gt_disp = self._xy_displacements(gt_pos, lag, box_size, unwrapped=gt_unwrapped)
        if pred_disp.size(1) == 0 or gt_disp.size(1) == 0:
            return torch.zeros(pred_pos.size(0), device=pred_pos.device)
        pred_r = torch.sqrt((pred_disp ** 2).sum(dim=-1) + 1e-6).reshape(pred_pos.size(0), -1)
        gt_r = torch.sqrt((gt_disp ** 2).sum(dim=-1) + 1e-6).reshape(gt_pos.size(0), -1)
        raw_w1 = calculate_wasserstein_rel(pred_r, gt_r)
        gt_rms = torch.sqrt((gt_r ** 2).mean() + 1e-6)
        return raw_w1 / gt_rms

    def compute_van_hove_multi_lag(self, pred_pos, gt_pos, box_size,
                                   pred_unwrapped=None, gt_unwrapped=None):
        per_lag = {}
        errs = []
        for lag in LAGS:
            err = self.compute_van_hove_error(
                pred_pos, gt_pos, box_size, lag,
                pred_unwrapped=pred_unwrapped, gt_unwrapped=gt_unwrapped)
            per_lag[lag] = err
            errs.append(err)
        return torch.stack(errs).mean(dim=0), per_lag

    # ------------------------------------------------------------------
    # Metric 3: Non-Gaussian Parameter (global, XY)
    # ------------------------------------------------------------------
    def compute_ngp(self, positions, lag, box_size, unwrapped=None):
        """alpha_2(tau) for d=2.  Returns [B]."""
        disp = self._xy_displacements(positions, lag, box_size, unwrapped=unwrapped)
        if disp.size(1) == 0:
            return torch.zeros(positions.size(0), dtype=torch.float64, device=positions.device)
        r_sq = (disp.double() ** 2).sum(dim=-1).reshape(disp.size(0), -1)  # [B, T*7]
        r4_mean = (r_sq ** 2).mean(dim=-1)
        r2_mean = r_sq.mean(dim=-1)
        d = 2
        return (r4_mean / ((1 + 2.0 / d) * r2_mean ** 2 + 1e-6)) - 1.0

    def compute_ngp_error(self, pred_pos, gt_pos, box_size,
                          pred_unwrapped=None, gt_unwrapped=None):
        per_lag = {}
        errs = []
        for lag in LAGS:
            pred_ngp = self.compute_ngp(pred_pos, lag, box_size, unwrapped=pred_unwrapped)
            gt_ngp = self.compute_ngp(gt_pos, lag, box_size, unwrapped=gt_unwrapped)
            abs_err = (pred_ngp - gt_ngp).abs()
            capped_err = torch.tanh(abs_err)
            per_lag[lag] = {'pred': pred_ngp, 'gt': gt_ngp,
                           'abs_err': abs_err, 'capped_err': capped_err}
            errs.append(capped_err)
        return torch.stack(errs).mean(dim=0), per_lag

    # ------------------------------------------------------------------
    # Metric 4: Pairwise distance distribution — g(r) proxy (chunked, 3D)
    # ------------------------------------------------------------------
    def compute_pairwise_distribution_error(self, pred_pos, gt_pos, box_size):
        """Chunked Wasserstein on full pairwise distance distribution.
        Particle-agnostic: treats all 7 particles as identical.
        pred_pos: [B, T, 21] (wrapped),  gt_pos: [1, T, 21] (wrapped)
        Returns: [B]
        """
        triu = torch.triu_indices(self.n_particles, self.n_particles, offset=1)
        chunk = self.chunk_size
        T = pred_pos.size(1)
        n_chunks = max(T // chunk, 1)

        B = pred_pos.size(0)
        chunk_errs = []
        for i in range(n_chunks):
            s, e = i * chunk, min((i + 1) * chunk, T)
            pred_xyz = pred_pos[:, s:e, :self.n_coords].reshape(B, e - s, self.n_particles, 3)
            gt_xyz = gt_pos[:, s:e, :self.n_coords].reshape(1, e - s, self.n_particles, 3)
            # PBC minimum-image pairwise diffs
            pred_diff = pred_xyz.unsqueeze(3) - pred_xyz.unsqueeze(2)
            pred_diff = pred_diff - box_size * torch.round(pred_diff / box_size)
            pred_dists = torch.sqrt((pred_diff ** 2).sum(dim=-1) + 1e-6)
            gt_diff = gt_xyz.unsqueeze(3) - gt_xyz.unsqueeze(2)
            gt_diff = gt_diff - box_size * torch.round(gt_diff / box_size)
            gt_dists = torch.sqrt((gt_diff ** 2).sum(dim=-1) + 1e-6)
            # Upper triangle → flatten
            pred_flat = pred_dists[:, :, triu[0], triu[1]].reshape(B, -1)
            gt_flat = gt_dists[:, :, triu[0], triu[1]].reshape(1, -1)
            gt_rms = torch.sqrt((gt_flat ** 2).mean() + 1e-6)
            chunk_errs.append(calculate_wasserstein_rel(pred_flat, gt_flat) / gt_rms)
        return torch.stack(chunk_errs).mean(dim=0)  # [B]

    # ------------------------------------------------------------------
    # Metric 5: Velocity errors (global, XY)
    # ------------------------------------------------------------------
    def compute_velocity_errors(self, pred_pos, gt_pos, box_size,
                                pred_unwrapped=None, gt_unwrapped=None):
        pred_diff = self._xy_displacements(pred_pos, 1, box_size, unwrapped=pred_unwrapped)
        gt_diff = self._xy_displacements(gt_pos, 1, box_size, unwrapped=gt_unwrapped)
        if pred_diff.size(1) == 0 or gt_diff.size(1) == 0:
            z = torch.zeros(pred_pos.size(0), device=pred_pos.device)
            return z, z
        pred_vel_x = pred_diff[..., 0].abs().mean(dim=(1, 2))
        pred_vel_y = pred_diff[..., 1].abs().mean(dim=(1, 2))
        gt_vel_x = gt_diff[..., 0].abs().mean(dim=(1, 2))
        gt_vel_y = gt_diff[..., 1].abs().mean(dim=(1, 2))
        return self.calc_relative_error(pred_vel_x, gt_vel_x), \
               self.calc_relative_error(pred_vel_y, gt_vel_y)

    # ------------------------------------------------------------------
    # Metric 6 (NEW): Lag-wise mean-VACF curve MSE (lab-convention).
    # VACF(L) = <v(0)·v(L)> / <|v|^2> with v = unwrapped XY position diff at
    # lag=1. Matches the definition used by archive/scripts/diffusion/ablation_rounds_2026-05/analysis/diagnose_ckpt_samples.py
    # and membrane/vacf_loss.compute_vacf_mse_loss.
    # ------------------------------------------------------------------
    def compute_vacf_curve_error(self, pred_pos, gt_pos, box_size,
                                  pred_unwrapped=None, gt_unwrapped=None,
                                  lags=None):
        if lags is None:
            lags = VACF_LAGS
        pred_uw = pred_unwrapped if pred_unwrapped is not None else self._unwrap_xy(pred_pos, box_size)
        gt_uw = gt_unwrapped if gt_unwrapped is not None else self._unwrap_xy(gt_pos, box_size)
        # velocity (lag=1)
        pv = pred_uw[:, 1:] - pred_uw[:, :-1]
        gv = gt_uw[:, 1:] - gt_uw[:, :-1]
        if pv.size(1) < 2 or gv.size(1) < 2:
            return torch.zeros(pred_pos.size(0), device=pred_pos.device)
        p_v2 = (pv ** 2).sum(dim=-1).mean(dim=(1, 2)).clamp_min(1e-8)  # [B]
        g_v2 = (gv ** 2).sum(dim=-1).mean(dim=(1, 2)).clamp_min(1e-8)  # [1]
        sq_err = torch.zeros(pred_pos.size(0), device=pred_pos.device)
        count = 0
        for L_ in lags:
            if pv.size(1) <= L_ or gv.size(1) <= L_:
                continue
            p_dot = (pv[:, :pv.size(1) - L_] * pv[:, L_:]).sum(dim=-1).mean(dim=(1, 2))  # [B]
            g_dot = (gv[:, :gv.size(1) - L_] * gv[:, L_:]).sum(dim=-1).mean(dim=(1, 2))  # [1]
            p_vacf = p_dot / p_v2
            g_vacf = g_dot / g_v2
            sq_err = sq_err + (p_vacf - g_vacf) ** 2
            count += 1
        return sq_err / max(count, 1)

    # ------------------------------------------------------------------
    # Metric 7 (NEW): symmetric long-time MSD log-ratio.
    # |log(MSD@500_pred / MSD@500_gt)|. Equal penalty for under- and
    # over-dispersion; complements alpha_err (shape) with magnitude.
    # ------------------------------------------------------------------
    def compute_msd_log_ratio_error(self, pred_tamsd, gt_tamsd, lag=MSD_RATIO_LAG):
        if lag not in pred_tamsd or lag not in gt_tamsd:
            return None
        p = pred_tamsd[lag].clamp_min(1e-8)
        g = gt_tamsd[lag].clamp_min(1e-8)
        return (p.log() - g.log()).abs()

    # ------------------------------------------------------------------
    # Teleport penalty (diagnostic only, not in objective)
    # ------------------------------------------------------------------
    def compute_teleport_penalty(self, pred_pos, box_size, pred_unwrapped=None):
        pred_diff = self._xy_displacements(pred_pos, 1, box_size, unwrapped=pred_unwrapped)
        if pred_diff.size(1) == 0:
            return torch.zeros(pred_pos.size(0), device=pred_pos.device)
        step_disp = torch.sqrt((pred_diff ** 2).sum(dim=-1) + 1e-6)
        max_disp = step_disp.reshape(pred_pos.size(0), -1).max(dim=-1).values
        thresh = 0.5 * box_size
        return F.relu(max_disp - thresh)

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, pl_module, initial_x, gt, box_size, rollout_steps):
        """Single continuous rollout + all metrics.

        initial_x: [B, 1, 21]
        gt:        [1, T_gt, 21]   — used only for comparison metrics
        rollout_steps: int         — pred trajectory length (independent of T_gt)

        When rollout_steps > T_gt, pred-vs-gt comparison metrics use the
        common prefix (T_gt). When rollout_steps < T_gt, gt is sliced.

        Returns dict of [B] tensors.
        """
        T_gt = gt.size(1)
        trajectory, unwrapped = pl_module.predict_future(
            initial_x, steps=rollout_steps, return_unwrapped=True)

        # Align pred to gt's length for comparison metrics.
        T_compare = min(T_gt, rollout_steps)
        traj_c = trajectory[:, :T_compare, :]
        unw_c = unwrapped[:, :T_compare, :]
        gt_c = gt[:, :T_compare, :]

        pred_xy = self._unwrap_xy(unw_c, box_size)
        gt_xy = self._unwrap_xy(gt_c, box_size)

        tamsd_err, alpha_pred, alpha_gt, tamsd_pred, tamsd_gt = \
            self.compute_tamsd_error(unw_c, gt_c, box_size,
                                     pred_unwrapped=pred_xy, gt_unwrapped=gt_xy)
        van_hove_err, van_hove_per_lag = \
            self.compute_van_hove_multi_lag(unw_c, gt_c, box_size,
                                            pred_unwrapped=pred_xy, gt_unwrapped=gt_xy)
        ngp_err, ngp_per_lag = \
            self.compute_ngp_error(unw_c, gt_c, box_size,
                                    pred_unwrapped=pred_xy, gt_unwrapped=gt_xy)
        pairwise_dist_err = self.compute_pairwise_distribution_error(
            traj_c, gt_c, box_size)
        vel_err_x, vel_err_y = \
            self.compute_velocity_errors(unw_c, gt_c, box_size,
                                          pred_unwrapped=pred_xy, gt_unwrapped=gt_xy)

        # NEW: VACF curve MSE
        vacf_err = self.compute_vacf_curve_error(
            unw_c, gt_c, box_size,
            pred_unwrapped=pred_xy, gt_unwrapped=gt_xy)
        # NEW: MSD@500 log ratio (uses tamsd dicts we already have)
        msd_log_err = self.compute_msd_log_ratio_error(tamsd_pred, tamsd_gt, lag=MSD_RATIO_LAG)
        if msd_log_err is None:  # lag not present in TAMSD set
            msd_log_err = torch.zeros(initial_x.size(0), device=initial_x.device)

        # Teleport / pred-only diagnostic uses FULL pred trajectory (rollout_steps)
        # so long-time runaway is caught even when T_gt is short.
        teleport = self.compute_teleport_penalty(unwrapped, box_size,
                                                  pred_unwrapped=self._unwrap_xy(unwrapped, box_size))

        return {
            'tamsd_err': tamsd_err,          # NOTE: actually relative alpha error (legacy name)
            'alpha_pred': alpha_pred,
            'alpha_gt': alpha_gt,
            'tamsd_pred': tamsd_pred,
            'tamsd_gt': tamsd_gt,
            'van_hove_err': van_hove_err,
            'van_hove_per_lag': van_hove_per_lag,
            'ngp_err': ngp_err,
            'ngp_per_lag': ngp_per_lag,
            'pairwise_dist_err': pairwise_dist_err,
            'vel_err_x': vel_err_x,
            'vel_err_y': vel_err_y,
            'vacf_err': vacf_err,
            'msd_log_err': msd_log_err,
            'teleport': teleport,
            'rollout_steps_actual': rollout_steps,
            'compare_steps_actual': T_compare,
        }

    # ------------------------------------------------------------------
    # Both-trajectory behavioral physics objective (== the resample eval)
    # ------------------------------------------------------------------
    def _ensure_gt(self, device, box, z=True):
        """Load both GT trajectories ONCE; precompute the merged-behavior
        reference, the per-metric noise floor, and the fixed 512-IC pool.
        z=False (a model without protein height): the reference is MD with every
        height at 0, since cohesion is a 3D pair distance here, and the objective
        is the metrics that are not about z."""
        if self._gt_ready:
            return
        self._gt_ready = True
        if not z:
            bad = [m for m in (self.objective_metrics or ()) if m.startswith('z_')]
            if bad:
                raise ValueError(f'objective_metrics {bad} score protein heights, and predict_z=False has none')
            if self.objective_metrics is None:
                self.objective_metrics = [m for m in pm.METRICS if not m.startswith('z_')]
        if not all(os.path.exists(p) for p in self.gt_paths):
            print(f"Stability callback: GT files {self.gt_paths} not all present "
                  f"-> physics objective DISABLED.")
            return

        def _load(p):
            key = ('phys_gt', p)
            cached = LongHorizonStabilityCallback._val_data_cache.get(key)
            if cached is None:
                raw = pd.read_csv(p, sep=None, engine='python', header=None) \
                    .apply(pd.to_numeric, errors='coerce').dropna().values
                cached = torch.tensor(raw[:, 1:], dtype=torch.float32)
                LongHorizonStabilityCallback._val_data_cache[key] = cached
            return cached.to(device)                            # [T, 21]

        t1, t2 = _load(self.gt_paths[0]), _load(self.gt_paths[1])
        if not z:
            keep = torch.ones(t1.shape[1], dtype=t1.dtype, device=t1.device)
            keep[2::3] = 0
            t1, t2 = t1 * keep, t2 * keep
        if {t1.shape[1], t2.shape[1]} != {3 * self.n_particles}:
            # a reshape to n_particles would otherwise read another system's columns without complaint
            raise ValueError(f'reference trajectories {self.gt_paths[:2]} are {t1.shape[1]} and {t2.shape[1]} wide, '
                             f'n_particles={self.n_particles} needs {3 * self.n_particles}')
        s1 = pm.compute_metrics(t1.reshape(1, -1, self.n_particles, 3), box)
        s2 = pm.compute_metrics(t2.reshape(1, -1, self.n_particles, 3), box)
        self.merged_gt = pm.merge_gt(s1, s2)                    # shared behavior
        self.floor = pm.noise_floor(s1, s2)                     # traj1-vs-traj2 spread
        # Fixed 512 random ICs pooled from BOTH trajectories (seeded -> stable
        # epoch-to-epoch and comparable across arms / sweeps).
        pool = torch.cat([t1, t2], dim=0)                       # [T1+T2, 21]
        g = torch.Generator().manual_seed(self.ic_seed)
        idx = torch.randperm(pool.shape[0], generator=g)[:self.n_samples]
        self.ic_pool = pool[idx].contiguous()                   # [n_samples, 21] (on device)

    # ------------------------------------------------------------------
    # Main callback entry point
    # ------------------------------------------------------------------
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.check_interval != 0:
            return
        box = float(getattr(pl_module, 'box_size', pm.BOX))
        self._ensure_gt(pl_module.device, box, z=getattr(pl_module, 'predict_z', True))
        if self.merged_gt is None:
            return

        torch.cuda.empty_cache()
        pl_module.eval()
        diverged = False
        with torch.inference_mode():
            x0 = torch.remainder(self.ic_pool.to(pl_module.device), box).unsqueeze(1)  # [B,1,21]
            sw = pl_module.predict_future(x0, steps=self.rollout_steps,
                                          return_unwrapped=False)
            if not torch.isfinite(sw).all():
                diverged = True
            else:
                pos = sw.reshape(self.n_samples, self.rollout_steps, self.n_particles, 3)
                model_stats = pm.compute_metrics(pos, box)
                obj, dist, pens = pm.objective(model_stats, self.merged_gt, self.floor,
                                               metrics=self.objective_metrics)
        pl_module.train()

        if diverged or obj != obj:        # NaN / divergence -> large penalty
            pl_module.log_dict({"stab/reported_objective": 1e6,
                                "stab/stability_objective": 1e6},
                               prog_bar=False, logger=True)
            return

        metrics = {
            "stab/reported_objective": obj,    # monitored: ckpt + ReduceLROnPlateau + Optuna
            "stab/stability_objective": obj,   # monitored: ModelCheckpoint + NaNDetection
        }
        for mtr in pm.METRICS:
            metrics[f"stab/penalty_{mtr}"] = pens.get(mtr, float('nan'))
            metrics[f"stab/dist_{mtr}"] = dist.get(mtr, float('nan'))
            metrics[f"stab/floor_{mtr}"] = self.floor.get(mtr, float('nan'))
        for k in ('step', 'z_step', 'z_membrane', 'alpha'):
            metrics[f"stab_model/{k}"] = float(model_stats[k])
        pl_module.log_dict(metrics, prog_bar=False, logger=True)
        # one line per check in the run log: with W&B off nothing else records the epoch curve
        print(f"stab epoch {trainer.current_epoch}: objective {obj:.4f}  "
              + " ".join(f"{m} {pens[m]:.3f}" for m in pm.METRICS if m in pens), flush=True)
