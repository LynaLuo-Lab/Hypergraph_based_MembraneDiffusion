import argparse
import json
import sys
import warnings
import torch
import torch._logging
import lightning as L
import os

warnings.filterwarnings("ignore", message=".*Online softmax is disabled.*")
warnings.filterwarnings("ignore", message=".*isinstance.*LeafSpec.*is deprecated.*")
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import WandbLogger
from lightning import seed_everything
from data.DataModule import BrownianDataModule
from data.paths import resolve_config
from nbcore.config.validate import validate_config
from membrane.LongHorizonStabilityCallback import LongHorizonStabilityCallback
from membrane.FitObjectiveCallback import FitObjectiveCallback
from nbcore.NaNDetectionCallback import NaNDetectionCallback
from nbcore.OptunaReportCallback import OptunaReportCallback
from membrane.model_registry import (MODEL_ARCH_DEFAULTS, arch_defaults,
                                              get_model_class)

try:
    from optuna.integration import PyTorchLightningPruningCallback
except ImportError:
    PyTorchLightningPruningCallback = None


# =========================================================================
#  Callbacks
# =========================================================================

def epoch_checkpoint(config, dirpath):
    """Per-epoch tracker: keeps every checkpoint and (with WandbLogger's log_model='all')
    uploads each to W&B as an artifact at save time. ckpt_every_epoch=False keeps one
    rolling checkpoint instead -- not zero: at an epoch end Lightning writes last.ckpt only
    if this tracker saved something in that epoch (with save_top_k=0 it appears once, when
    training ends), and last.ckpt is what a preempted run resumes from."""
    return ModelCheckpoint(
        every_n_epochs=1,
        dirpath=dirpath,
        save_top_k=-1 if config.get('ckpt_every_epoch', True) else 1,
        save_last=True,
        filename='epoch-{epoch:03d}',
    )


def checkpoint_callbacks(config, dirpath):
    """The best-only trackers, then the per-epoch one. The order is load-bearing: last.ckpt
    is written by the per-epoch tracker and carries the state of the trackers that ran
    before it, so a resumed run only keeps a single best checkpoint if they come first."""
    cbs = [ModelCheckpoint(monitor="stab/stability_objective", dirpath=dirpath,
                           save_top_k=1, mode='min', save_last=False, filename='best-{epoch:03d}')]
    if config.get('fit_objective', False):
        cbs.append(ModelCheckpoint(monitor="fit/objective", dirpath=dirpath,
                                   save_top_k=1, mode='min', save_last=False, filename='fit-{epoch:03d}'))
    cbs.append(epoch_checkpoint(config, dirpath))
    return cbs


class CurriculumLearningCallback(Callback):
    """Linear ramp of training sequence length."""
    def __init__(self, start_len=50, end_len=500, warmup_epochs=25):
        super().__init__()
        self.start_len = start_len
        self.end_len = end_len
        self.warmup_epochs = warmup_epochs

    def _len_for(self, epoch):
        if epoch >= self.warmup_epochs:
            return self.end_len
        progress = epoch / self.warmup_epochs
        return int(self.start_len + (self.end_len - self.start_len) * progress)

    def _apply(self, trainer, pl_module, epoch, log=True):
        if not hasattr(trainer.datamodule, 'set_train_seq_length'):
            return
        current_len = self._len_for(epoch)
        trainer.datamodule.set_train_seq_length(current_len)
        if log:
            pl_module.log("train/seq_length", float(current_len), prog_bar=True)

    def on_train_start(self, trainer, pl_module):
        # Requeue safety: the DataModule is rebuilt on resume and re-inits to
        # seq_len_start (DataModule.__init__), while this ramp only fired at epoch
        # END -- so a job resumed at epoch N trained its first epoch at the start
        # length and silently corrected only at that epoch's end. Under a 1-day
        # wall that is one wrong epoch per requeue. Re-derive from current_epoch,
        # which Lightning restores with ckpt_path=.
        self._apply(trainer, pl_module, trainer.current_epoch, log=False)

    def on_train_epoch_end(self, trainer, pl_module):
        self._apply(trainer, pl_module, trainer.current_epoch)


# =========================================================================
#  Configuration: base defaults + per-model architecture defaults
# =========================================================================

# These are architecture-invariant defaults shared by every model type.
BASE_CONFIG = {
    'project_name': 'Multilag_OUGMM_chol_vs_diag',
    'wandb_group': '',  # set per-run via --wandb-group (e.g. cholesky/diagonal)
    # 'all' uploads every ModelCheckpoint save to W&B (per-epoch + best).
    # Set to True for best-only, False to disable. See WandbLogger(log_model=...).
    'wandb_log_model': 'all',
    'n_particles': 7,
    'dataset_path': 'data/7piezo_traj2.csv',
    'dataset_paths': [
        'data/7piezo_traj1.csv',
        'data/7piezo_traj2.csv',
    ],  # fallback raw CSVs; used only when augmented_dataset_paths is empty
    'augmented_dataset_paths': [
        'data/shared/7piezo/augmented/7piezo_traj1_aug40320.npy',
        'data/shared/7piezo/augmented/7piezo_traj2_aug40320.npy',
    ],  # preferred: pre-generated S7 x D4 augmentations (see archive/scripts/shared/data_prep/generate_augmented_data.py)
    'checkpoint_path': '',
    'multi_system': False,
    'system_configs': [],
    'model_type': 'multilag',
    'decoder_type': 'mlp',
    # decoder_noise_type: instantaneous (per-step) noise covariance —
    # 'cholesky' (per-particle 3x3 full cov, xyz correlated) | 'diagonal'
    # (per-axis independent) | 'cholesky_full' (full 21x21, cross-particle).
    # See membrane/decoders.py. Override per-run via --decoder-noise-type.
    'decoder_noise_type': 'cholesky',
    # decoder_ar1: orthogonal TEMPORAL AR(1) coupling on the output noise
    # (eps_t = A.eps_{t-1} + L.eta_t), global + contractive. 'none' = white |
    # 'diag' (per-particle per-axis A) | 'block' (per-particle 3x3 A) |
    # 'full' (DxD A, cross-particle). Override via --decoder-ar1.
    'decoder_ar1': 'none',
    'transition_head_type': 'mlp_gauss',  # OU-GMM uses standard Gaussian head
    'seed': 42,

    # --- Training schedule ---
    'n_samples': 512,
    # samples_per_epoch=960 / batch_size=80 = 12 batches per epoch. Over 50
    # epochs that's 600 grad steps, matching the 1cpzhztr/566-step target.
    'samples_per_epoch': 960,
    # Default LR mirrors gayso5zk's Sophia-calibrated value. Per-run jobs in
    # the chol_vs_diag sweep override with --learning-rate from one lr_find
    # per noise_type (see archive/slurm/diffusion/2026-05-19_chol-vs-diag/submit_lr_find_chol_vs_diag.slurm).
    'learning_rate': 6.993848069961716e-05,
    'input_noise_sigma': 0.0,
    'seq_len_start': 650,
    'seq_len_end': 5000,
    # Ramp seq_len over 20 epochs -- gentler than bumping seq by a jump every
    # epoch; gives the LSTM states time to adapt.
    'seq_len_warmup': 20,
    'num_workers': 16,

    # --- Fixed training params ---
    # Cluster A4500 21.2 GB bf16-mixed probe: bs=80 peaks at 17.95 GB (3.25 GB
    # headroom); bs=88 peaks at 19.71 GB (1.49 GB, too tight for long runs);
    # bs=96 OOMs. See slurm/shared/submit_memory_probe.slurm + d263c2e probe log.
    'batch_size': 80,
    'n_rollout_samples': 1,
    # 50 epochs * 12 batches/epoch = 600 gradient steps.
    'max_epochs': 50,
    'latent_type': "ou_gmm",
    'gmm_components': 3,  # gayso5zk
    # --- NSF transition head (unused under transition_head_type='mlp_gauss';
    # kept so switching back to NSF is a one-key change) ---
    'nsf_bins': 4,
    'nsf_transforms': 2,
    'nsf_conditioner_hidden': 128,
    'nsf_tail_bound': 3.0,
    'dt': 0.0001,
    'box_size': 856.1959882301632,

    # --- Loss weights (architecture-invariant) ---
    # Calibrated from run ew1iihb8 first training batch (step 1, epoch 0).
    # w_i = 5 / raw_i so each weighted term equals 5 at init, total loss ~= 35.
    # Raw initial values: disp=9.282, ang=0.430, pair=0.0074, zq=0.0466,
    # vacf=0.146, coh=0.0062, zdisp=4.754.
    'w_displacement': 0.538691,
    'w_angle': 11.619345,
    'w_pairwise': 678.834138,
    'w_z_quantile': 0.0,
    'w_vacf': 34.159186,
    'w_cohesion': 802.474596,
    'w_z_displacement': 1.051773,
    # NGP / displacement-kurtosis matching loss (added 2026-06-13). Off by
    # default; targets the 4th-moment ratio of XY displacement (the persistent
    # NGP gap). Anchor set by the lat6_kurtosis GP sweep.
    'w_kurtosis': 0.0,
    'kurtosis_lags': [1, 5, 10, 25, 50, 100, 250],
    # Lag-wise mean-VACF MSE loss on the predicted position-VACF curve.
    # Off by default (introduced 2026-05-21).
    'w_vacf_mse': 0.0,
    # Direct latent-velocity autocorrelation loss on z's (added 2026-05-23).
    # Off by default. Bypasses the decoder chain — applies pressure directly
    # at the SDE latent level. kernel='tanh' is the anti-target shape:
    # punishes corr<=0 + overshoot past target, no opinion in the good zone.
    'w_latent_corr': 0.0,
    'latent_corr_kernel': 'tanh',     # 'tanh' or 'mse_target'
    'latent_corr_lags': None,         # None -> [1, 2, 5, 10, 25]
    'latent_corr_target': None,       # None -> GT 7piezo position-VACF curve
    'latent_corr_scale': 0.05,        # tanh width

    # --- Regularization (architecture-invariant) ---
    'dropout': 0.2516022097745067,
    'gradient_clip_val': 20.0,
    # patience=5 epochs = 40 batches at 8 batches/ep ~ 50 min cluster wall.
    # Previously was 3 with ~360 batches/epoch (proportional to ~10% of run);
    # staying near that same proportion with the new epoch size.
    'scheduler_patience': 5,

    # --- Optimizer ---
    'optimizer_type': 'sophia_g',
    'sophia_rho': 0.015890222820529036,

    # --- Stochastic SDE (shared by all) ---
    'latent_dim': 21,
    'noise_rank': 55,  # gayso5zk; 55 > latent_dim=21 so per-component cov is full-rank capable

    # --- Memory ---
    # gradient_checkpointing=True required for seq=5000 even at k=1: without
    # it the activation tape over 5000 AR steps OOMs every batch size that
    # would otherwise fit. Probe-verified at bs=80 bf16-mixed on A4500 21.2 GB.
    'gradient_checkpointing': True,
    'accumulate_grad_batches': 1,
}

# Per-decoder architecture defaults for the generalized multilag model.
# Each entry contains ONLY the keys specific to that decoder_type.
# Shared keys (encoder sizes, etc.) live in MULTILAG_DEFAULTS below.
MULTILAG_DEFAULTS = {
    'num_heads': 21, 'hidden_dim': 63,
    'n_lag_lstms': 7, 'lag_lstm_width': 64,
    'dec_hidden_dim': 256,
}

DECODER_DEFAULTS = {
    'mlp':     {'dec_mlp_layers': 3},
    'xformer': {'dec_xformer_layers': 2, 'dec_xformer_heads': 8},
    # xformer_particles: the FIXED cross-attention decoder. 'xformer' builds a
    # one-token query against a one-token key/value, so its attention weights are
    # identically 1.0 and the query is provably ignored -- it is an MLP with
    # 526k dead parameters. Use this for new runs; 'xformer' stays only so old
    # checkpoints load.
    'xformer_particles': {'dec_xformer_layers': 2, 'dec_xformer_heads': 8},
    # Per-protein cross-attention. 'xformer' builds ONE query and ONE key token, so
    # its softmax is identically 1.0 and nothing relational happens; this one puts
    # the proteins on the sequence axis so change can attend to position.
    'xformer_tokens': {'dec_xformer_layers': 2, 'dec_xformer_heads': 8,
                       'dec_hyper_tokens': True},
    'kan':     {'kan_grid_size': 5, 'kan_spline_order': 3},
}

# Legacy support: old per-variant keys still resolve to the generalized form.
# Tools (Optuna studies, saved configs) may still reference these names.
LEGACY_MODEL_TYPE_MAP = {
    'multilag_mlp_7x3d':     ('multilag', 'mlp'),
    'lstm_mlp_7x3d':         ('multilag', 'mlp'),
    'lstm_xformer_7x3d':     ('multilag', 'xformer'),
    'lstm_kan_7x3d':         ('multilag', 'kan'),
    'lstm_lstm_7x3d':        ('multilag', 'mlp'),
    'parallel_xformer_7x3d': ('multilag', 'xformer'),
}


def get_default_config(model_type=None, decoder_type=None):
    """Build a complete config dict for the generalized multilag model.

    model_type: 'multilag' (default) or any legacy alias.
    decoder_type: 'mlp' | 'xformer' | 'kan' (defaults to 'mlp').
    Legacy model_type names are translated automatically.
    """
    config = dict(BASE_CONFIG)
    mt = model_type or config['model_type']
    if mt in LEGACY_MODEL_TYPE_MAP:
        canonical, implied_decoder = LEGACY_MODEL_TYPE_MAP[mt]
        mt = canonical
        if decoder_type is None:
            decoder_type = implied_decoder

    # Model types that bring their own architecture keys (the membrane-first
    # hypergraph) take them from the registry and skip the multilag/decoder tables
    # entirely -- they have neither a geo_embed nor a decoder_type.
    arch = arch_defaults(mt)
    if arch:
        config['model_type'] = mt
        config.update(arch)
        return config
    if mt != 'multilag':
        raise KeyError(
            f"Unknown model_type {mt!r}. Supported: 'multilag', "
            f"{sorted(k for k in MODEL_ARCH_DEFAULTS)} "
            f"(or legacy aliases: {sorted(LEGACY_MODEL_TYPE_MAP)})."
        )
    dt = decoder_type or config.get('decoder_type', 'mlp')
    if dt not in DECODER_DEFAULTS:
        raise KeyError(
            f"Unknown decoder_type {dt!r}. Available: {sorted(DECODER_DEFAULTS)}"
        )

    config['model_type'] = mt
    config['decoder_type'] = dt
    config.update(MULTILAG_DEFAULTS)
    config.update(DECODER_DEFAULTS[dt])
    return config


# =========================================================================
#  Training
# =========================================================================

def run_training(config, wandb_run=None, optuna_trial=None):
    resolve_config(config)
    seed_everything(config['seed'], workers=True)
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True
    # Dynamo config attr names vary across torch versions (recompile_limit was
    # renamed / added over releases); set only those present.
    for _attr in ('recompile_limit', 'cache_size_limit', 'accumulated_recompile_limit'):
        if hasattr(torch._dynamo.config, _attr):
            setattr(torch._dynamo.config, _attr, 10000)

    # Emit a warning each time dynamo recompiles a graph, plus the reason.
    # If the per-step function is specializing on step_idx (as int), this will
    # produce one warning per unique step value on the first epoch and
    # nothing thereafter. If it is NOT specializing, you will see one or a
    # handful of compiles total and then silence.
    torch._logging.set_logs(recompiles=True)

    # --- Data ---
    dm = BrownianDataModule(config)
    dm.setup()

    # --- Model class ---
    BrownianModel = get_model_class(config['model_type'])

    # --- Surgical checkpoint resume ---
    ckpt_path = config.get('checkpoint_path', '')
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"--- SURGICAL RESUME: Loading weights from {ckpt_path} ---")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            # Overwrite architecture keys to prevent size mismatches. The
            # architecture-relevant keys live in MULTILAG_DEFAULTS (shared
            # encoder/decoder dims) and DECODER_DEFAULTS[decoder_type] (per
            # decoder variant), plus a few shared latent-space keys.
            dec_type = config.get('decoder_type', 'mlp')
            arch_keys = list(MULTILAG_DEFAULTS.keys())
            arch_keys += list(DECODER_DEFAULTS.get(dec_type, {}).keys())
            arch_keys += list(arch_defaults(config['model_type']).keys())
            arch_keys += ['latent_dim', 'noise_rank', 'gmm_components', 'geo_ic_once',
                          'geo_ic_axes', 'geo_pair_dir_shells']
            for key in arch_keys:
                if key in hparams:
                    print(f"  Overwriting config[{key}] with checkpoint value: {hparams[key]}")
                    config[key] = hparams[key]

    model = BrownianModel(config)

    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- Callbacks ---
    # Checkpoint destination. Trainer(default_root_dir=...) is NOT enough: with a
    # logger attached Lightning derives the path from logger.save_dir, and
    # WandbLogger's defaults to the cwd -- a per-pod emptyDir on the cluster, which
    # dies with the pod. ModelCheckpoint(dirpath=) is the only setting that wins.
    # Ported from train_brownian_p.py (commit d673ded), which had it; this trainer
    # did NOT, while k8s/membrane/2026-09-07_squash-joint/job-membrane-squash-joint.yaml:90 sets default_root_dir and
    # the key validates -- so it was accepted and silently ignored.
    _ckpt_dir = config.get('default_root_dir') or None

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        CurriculumLearningCallback(
            start_len=config.get('seq_len_start', 50),
            end_len=config.get('seq_len_end', 200),
            warmup_epochs=config.get('seq_len_warmup', 50),
        ),
        LongHorizonStabilityCallback(
            check_interval=config.get('stab_check_interval', 1),
            n_samples=config.get('n_samples', 512),
            rollout_steps=config.get('stab_rollout_steps', 40000),
            gt_paths=config.get(
                'stability_gt_paths',
                ('data/7piezo_traj1.csv', 'data/7piezo_traj2.csv'),
            ),
            objective_metrics=config.get('stab_objective_metrics', None),
        ),
        *checkpoint_callbacks(config, _ckpt_dir),
        NaNDetectionCallback(
            patience=3, monitor="stab/stability_objective", optuna_trial=optuna_trial
        ),
        OptunaReportCallback(
            optuna_trial=optuna_trial, monitor="stab/reported_objective"
        ),
    ]
    # Weight-fit objective (clustering + lateral diffusion + membrane depth, membrane/fit_objective.py),
    # scored every epoch; checkpoint_callbacks gives it its own best checkpoint. OFF by default.
    if config.get('fit_objective', False):
        callbacks.append(FitObjectiveCallback(
            gt_paths=config.get('stability_gt_paths',
                                ('data/7piezo_traj1.csv', 'data/7piezo_traj2.csv')),
            npz_paths=config.get('membrane_data_paths'),
            n_rollouts=config.get('fit_objective_rollouts', 32),
            dirpath=_ckpt_dir,
            rc=config.get('fit_rc'),
            core=config.get('fit_core'),
        ))
    # Early stopping on the same objective ModelCheckpoint selects on. ON by default:
    # the curriculum ablation measured both arms overfitting hard past their peak
    # (flat seed 300 went 8.24 -> 17.27 on stab/reported_objective), so the tail of
    # every run is training past the point of use. Set early_stop_patience=0 to
    # disable and reproduce pre-2026-09-20 behaviour exactly.
    # start_epoch guards the curriculum case: during a seq_len ramp the objective is
    # computed on SHORTER rollouts than deployment, so stopping there would kill a run
    # that had never seen a full-length sequence. With warmup=0 this is a no-op.
    _es_patience = int(config.get('early_stop_patience', 10))
    if _es_patience > 0:
        from nbcore.callbacks.early_stop import DelayedEarlyStopping
        callbacks.append(DelayedEarlyStopping(
            monitor='stab/stability_objective', mode='min',
            patience=_es_patience,
            min_delta=float(config.get('early_stop_min_delta', 0.0)),
            start_epoch=int(config.get('early_stop_start_epoch',
                                       config.get('seq_len_warmup', 0))),
            verbose=True,
        ))
    # Weight EMA, OFF by default (ema_decay=0), so every existing config is unchanged.
    # It is an observer: the shadow never feeds back into training, so it cannot move
    # the base model. It writes ema-epoch=NNN.ckpt beside the normal checkpoints, to be
    # scored afterwards through the same eval path -- which makes EMA-vs-live a PAIRED
    # comparison inside one run, free of the 14-40% seed spread that makes unpaired
    # comparisons on this model uninformative.
    _ema_decay = float(config.get('ema_decay', 0.0))
    if _ema_decay > 0:
        from nbcore.callbacks.ema import EMACallback
        callbacks.append(EMACallback(
            decay=config.get('ema_decays') or _ema_decay,
            dirpath=_ckpt_dir,
            start_epoch=int(config.get('ema_start_epoch', 0)),
            every_n_epochs=int(config.get('ema_every_n_epochs', 1)),
        ))

    # --- Logger ---
    wandb_log_model = config.get('wandb_log_model', False)
    wandb_group = config.get('wandb_group') or None  # empty string -> None
    if wandb_run is not None:
        logger = WandbLogger(experiment=wandb_run, log_model=wandb_log_model)
    else:
        run_name = config.get('wandb_run_name') or None
        if run_name is None and wandb_group:
            run_name = f"{wandb_group}_seed{config['seed']}"
        logger = WandbLogger(
            project=config['project_name'],
            config=config,
            log_model=wandb_log_model,
            group=wandb_group,
            name=run_name,
        )

    # --- Trainer ---
    accum_steps = config.get('accumulate_grad_batches', 1)
    trainer = L.Trainer(
        max_epochs=config['max_epochs'],
        accelerator="cuda",
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config.get('gradient_clip_val', 0.3),
        gradient_clip_algorithm="norm",
        precision="bf16-mixed",
        limit_val_batches=0,
        enable_progress_bar=(optuna_trial is None),
        accumulate_grad_batches=accum_steps,
        # samples_per_epoch=128 / batch_size=64 -> 2 batches/epoch, so
        # log_every_n_steps > 1 drops per-step logs entirely.
        log_every_n_steps=1,
    )

    # --- Log model summary ---
    if trainer.logger and hasattr(trainer.logger, 'experiment'):
        exp = trainer.logger.experiment
        exp.summary["model/total_params"] = total_params
        exp.summary["model/trainable_params"] = trainable_params
        exp.summary["model/total_params_M"] = total_params / 1e6
        exp.summary["model/model_type"] = config['model_type']
        exp.summary["data/dataset_path"] = config.get('dataset_path', '')
        exp.summary["data/box_size"] = config.get('box_size', 1.0)

    # --- Fit ---
    try:
        resume_ckpt = config.get('resume_from_checkpoint', None)
        trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"OOM during training — returning inf score")
        if optuna_trial is not None:
            return float('inf')
        raise

    # --- Return score ---
    score = trainer.callback_metrics.get('stab/reported_objective', None)
    if score is None:
        return float('inf')
    return score.item() if hasattr(score, 'item') else float(score)


# =========================================================================
#  LR Finder
# =========================================================================

def run_lr_find(config, num_training=100, min_lr=1e-7, max_lr=1e-1,
                output_dir='lr_find'):
    """Run Lightning's LR finder. Saves plot + JSON; returns the suggested LR.

    Strategy:
      - Fixed short seq (= config['seq_len_start']) for speed.
      - torch.compile disabled to avoid paying compile warmup on every mini-step.
      - No curriculum / stability / checkpoint callbacks — lr_find is a mock
        training phase, not a real run; side-effecting callbacks would corrupt
        state.
      - W&B disabled.
    """
    from lightning.pytorch.tuner import Tuner

    seed_everything(config['seed'], workers=True)
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True

    # Local copy; do not mutate caller's config.
    cfg = dict(config)
    seq_len = int(cfg.get('seq_len_start', 500))
    cfg['seq_len_start'] = seq_len
    cfg['seq_len_end'] = seq_len
    cfg['max_epochs'] = 1
    # Ensure the dataloader can yield num_training batches inside one epoch so
    # lr_find doesn't have to cross epoch boundaries.
    cfg['samples_per_epoch'] = cfg['batch_size'] * num_training

    # Disable torch.compile BEFORE building the model (model ctor applies it).
    torch_compile_orig = torch.compile
    torch.compile = lambda m, **k: m  # noqa: E731

    # Disable W&B for lr_find only; restore after so the subsequent real
    # training run syncs normally. Using os.environ directly (not setdefault)
    # so the 'disabled' value doesn't leak through into run_training.
    wandb_mode_prev = os.environ.get('WANDB_MODE')
    os.environ['WANDB_MODE'] = 'disabled'

    try:
        print(f"[lr_find] seq_len={seq_len}  batch_size={cfg['batch_size']}  "
              f"k={cfg['n_rollout_samples']}  num_training={num_training}  "
              f"range=[{min_lr:.1e}, {max_lr:.1e}]", flush=True)

        dm = BrownianDataModule(cfg)
        dm.setup()
        BrownianModel = get_model_class(cfg['model_type'])
        model = BrownianModel(cfg)

        trainer = L.Trainer(
            max_epochs=1,
            accelerator='cuda',
            logger=False,
            callbacks=[],  # NO callbacks during lr_find
            gradient_clip_val=cfg.get('gradient_clip_val', 0.3),
            gradient_clip_algorithm='norm',
            precision='bf16-mixed',
            limit_val_batches=0,
            enable_progress_bar=True,
            enable_checkpointing=False,
            log_every_n_steps=1,
        )

        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            model, datamodule=dm,
            num_training=num_training,
            min_lr=min_lr,
            max_lr=max_lr,
        )
        suggestion = lr_finder.suggestion()

        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'lr_find.png')
        try:
            fig = lr_finder.plot(suggest=True)
            fig.savefig(plot_path, dpi=120, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f"[lr_find] saved plot -> {plot_path}", flush=True)
        except Exception as e:
            print(f"[lr_find] plot failed: {e}", flush=True)

        data_path = os.path.join(output_dir, 'lr_find_results.json')
        with open(data_path, 'w') as f:
            json.dump({
                'suggestion': float(suggestion) if suggestion is not None else None,
                'lrs': lr_finder.results.get('lr', []),
                'losses': lr_finder.results.get('loss', []),
                'num_training': num_training,
                'min_lr': min_lr,
                'max_lr': max_lr,
                'seq_len': seq_len,
                'batch_size': cfg['batch_size'],
                'n_rollout_samples': cfg['n_rollout_samples'],
                'transition_head_type': cfg.get('transition_head_type'),
                'latent_type': cfg.get('latent_type'),
                'decoder_type': cfg.get('decoder_type'),
            }, f)
        print(f"[lr_find] saved raw results -> {data_path}", flush=True)

        print("=" * 64, flush=True)
        if suggestion is None:
            print(f"[lr_find] NO SUGGESTION — inspect the plot / JSON manually.", flush=True)
        else:
            print(f"[lr_find] SUGGESTION: {suggestion:.3e}", flush=True)
            print(f"[lr_find] (current BASE_CONFIG learning_rate = {config['learning_rate']:.3e})", flush=True)
        print("=" * 64, flush=True)
        return suggestion

    finally:
        torch.compile = torch_compile_orig
        if wandb_mode_prev is None:
            os.environ.pop('WANDB_MODE', None)
        else:
            os.environ['WANDB_MODE'] = wandb_mode_prev


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="",
                        help="Path to .ckpt file for surgical resume")
    parser.add_argument('--resume', type=str, default="",
                        help="Path to .ckpt file for full Lightning resume (optimizer + epoch + LR)")
    parser.add_argument('--model_type', type=str, default=None,
                        help="Model type (overrides default)")
    parser.add_argument('--project_name', type=str, default=None,
                        help="W&B project name (overrides default)")
    parser.add_argument('--max-epochs', type=int, default=None,
                        help="Override max_epochs (useful for short validation runs)")
    parser.add_argument('--samples-per-epoch', type=int, default=None,
                        help="Override samples_per_epoch")
    parser.add_argument('--seq-len-end', type=int, default=None,
                        help="Override seq_len_end (and seq_len_warmup stays proportional)")
    parser.add_argument('--wandb-mode', type=str, default=None,
                        help="Force W&B mode (e.g. 'disabled', 'offline')")
    parser.add_argument('--lr-find', action='store_true',
                        help="Run Lightning LR finder first, replace "
                             "config['learning_rate'] with the suggestion, "
                             "then continue to full training in the same "
                             "invocation. Saves lr_find/lr_find.png + "
                             "lr_find/lr_find_results.json.")
    parser.add_argument('--lr-find-only', action='store_true',
                        help="Run lr_find and exit without training. Use this "
                             "when the lr_find output will feed downstream jobs "
                             "via --learning-rate. Implies --lr-find.")
    parser.add_argument('--lr-find-num-training', type=int, default=100,
                        help="Number of mini-training steps for lr_find (default 100).")
    parser.add_argument('--lr-find-min-lr', type=float, default=1e-7,
                        help="Minimum LR for lr_find sweep (default 1e-7).")
    parser.add_argument('--lr-find-max-lr', type=float, default=1e-1,
                        help="Maximum LR for lr_find sweep (default 1e-1).")
    parser.add_argument('--lr-find-output-dir', type=str, default='lr_find',
                        help="Output directory for lr_find artifacts (default 'lr_find').")
    parser.add_argument('--seed', type=int, default=None,
                        help="Override config['seed']. Used by the chol_vs_diag "
                             "sweep to vary the 5 seeds per noise_type.")
    parser.add_argument('--decoder-noise-type', type=str, default=None,
                        choices=['cholesky', 'diagonal', 'cholesky_full'],
                        help="Override config['decoder_noise_type'] "
                             "(cholesky | diagonal | cholesky_full).")
    parser.add_argument('--decoder-ar1', type=str, default=None,
                        choices=['none', 'diag', 'block', 'full'],
                        help="Override config['decoder_ar1'] — temporal AR(1) output "
                             "noise coupling A (none | diag | block | full).")
    parser.add_argument('--learning-rate', type=float, default=None,
                        help="Override config['learning_rate']. Slurm jobs pass "
                             "the lr_find suggestion read from "
                             "lr_find/<noise>/lr_find_results.json.")
    parser.add_argument('--wandb-group', type=str, default=None,
                        help="Group label for W&B (e.g. 'cholesky' or 'diagonal'). "
                             "Lets the chol_vs_diag sweep show as two side-by-side "
                             "groups in the wandb UI.")
    parser.add_argument('--loss-weights-json', type=str, default=None,
                        help="Path to a JSON file containing {w_displacement, "
                             "w_angle, w_pairwise, w_z_quantile, w_vacf, w_cohesion, "
                             "w_z_displacement}. Overrides the matching keys in "
                             "BASE_CONFIG. Used by the full chol_vs_diag pipeline "
                             "to feed GPSampler best_params into training.")
    parser.add_argument('--config-json', type=str, default=None,
                        help="Full config-override JSON applied over defaults "
                             "(architecture + loss weights + membrane paths). Its "
                             "model_type/decoder_type pick the right defaults first, "
                             "then every key in the JSON wins. Run-specific CLI flags "
                             "below (--seed, --max-epochs, ...) still override it.")
    args = parser.parse_args()

    config = get_default_config(model_type=args.model_type)
    if args.config_json:
        with open(args.config_json) as f:
            _cfg_over = json.load(f)
        # re-derive defaults for the JSON's own decoder so DECODER_DEFAULTS apply,
        # then let the JSON win over them.
        config = get_default_config(model_type=_cfg_over.get('model_type'),
                                    decoder_type=_cfg_over.get('decoder_type'))
        config.update(_cfg_over)
        validate_config(config, source=args.config_json)
        print(f"  config-json: applied {len(_cfg_over)} overrides from {args.config_json}")
    if args.checkpoint:
        config['checkpoint_path'] = args.checkpoint
    if args.resume:
        config['resume_from_checkpoint'] = args.resume
    if args.project_name:
        config['project_name'] = args.project_name
    if args.max_epochs is not None:
        config['max_epochs'] = args.max_epochs
    if args.samples_per_epoch is not None:
        config['samples_per_epoch'] = args.samples_per_epoch
    if args.seq_len_end is not None:
        config['seq_len_end'] = args.seq_len_end
    if args.wandb_mode is not None:
        os.environ['WANDB_MODE'] = args.wandb_mode
    if args.seed is not None:
        config['seed'] = args.seed
    if args.decoder_noise_type is not None:
        config['decoder_noise_type'] = args.decoder_noise_type
    if args.decoder_ar1 is not None:
        config['decoder_ar1'] = args.decoder_ar1
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.wandb_group is not None:
        config['wandb_group'] = args.wandb_group
    if args.loss_weights_json is not None:
        if not os.path.exists(args.loss_weights_json):
            raise FileNotFoundError(
                f"--loss-weights-json {args.loss_weights_json!r} not found."
            )
        with open(args.loss_weights_json) as f:
            lw = json.load(f)
        _W_KEYS = ('w_displacement', 'w_angle', 'w_pairwise', 'w_z_quantile',
                   'w_vacf', 'w_cohesion', 'w_z_displacement')
        missing = [k for k in _W_KEYS if k not in lw]
        if missing:
            raise KeyError(
                f"--loss-weights-json {args.loss_weights_json} missing keys: {missing}"
            )
        for k in _W_KEYS:
            print(f"  loss-weights override: {k} {config.get(k)} -> {lw[k]}")
            config[k] = float(lw[k])

    if args.lr_find or args.lr_find_only:
        suggestion = run_lr_find(
            config,
            num_training=args.lr_find_num_training,
            min_lr=args.lr_find_min_lr,
            max_lr=args.lr_find_max_lr,
            output_dir=args.lr_find_output_dir,
        )
        if args.lr_find_only:
            if suggestion is not None:
                print(f"[lr_find_only] suggestion={float(suggestion):.3e} "
                      f"saved to {args.lr_find_output_dir}/lr_find_results.json",
                      flush=True)
            else:
                print(f"[lr_find_only] no suggestion returned; inspect "
                      f"{args.lr_find_output_dir}/ manually.", flush=True)
            sys.exit(0)
        if suggestion is not None:
            print(f"[lr_find -> train] learning_rate "
                  f"{config['learning_rate']:.3e} -> {float(suggestion):.3e}",
                  flush=True)
            config['learning_rate'] = float(suggestion)
        else:
            print(f"[lr_find -> train] no suggestion returned; keeping "
                  f"learning_rate = {config['learning_rate']:.3e}", flush=True)

    run_training(config)
