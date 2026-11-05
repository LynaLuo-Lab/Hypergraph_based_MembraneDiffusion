"""Learning-rate schedules selected by the ``lr_schedule`` config key.

Until now ``configure_optimizers`` hardcoded ReduceLROnPlateau, so a sweep that set
``lr_schedule`` got the plateau schedule regardless of what it asked for -- every arm
of the 30-cell seed sweep would have trained the same schedule.

``plateau`` reproduces the previous hardcoded behaviour exactly, so runs that do not
set ``lr_schedule`` are unchanged.

The warmup schedules use LambdaLR, which scales each param group's own ``initial_lr``.
That matters because the dynamics group carries ``lr * dynamics_lr_mult``; a scheduler
that set an absolute LR would flatten that ratio.
"""
import math

import torch


SCHEDULES = ('constant', 'plateau', 'cosine_warmup', 'linear_warmup')


def _warmup_lambda(kind, warmup_epochs, total_epochs, min_factor):
    """Multiplicative factor on the base LR, as a function of epoch index."""
    warm = max(0, int(warmup_epochs))
    total = max(int(total_epochs), warm + 1)
    decay_span = max(1, total - warm)

    def fn(epoch):
        if epoch < warm:
            # ramp over the warmup, never starting at exactly 0 (a zero-LR first epoch
            # wastes it and makes the loss curve unreadable)
            return (epoch + 1) / (warm + 1)
        t = min(1.0, (epoch - warm) / decay_span)
        if kind == 'cosine_warmup':
            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * t))
        return min_factor + (1.0 - min_factor) * (1.0 - t)      # linear_warmup

    return fn


def build_scheduler_config(optimizer, hparams, monitor='stab/reported_objective'):
    """Return the dict Lightning expects under 'lr_scheduler'."""
    kind = hparams.get('lr_schedule', 'plateau')
    if kind not in SCHEDULES:
        raise ValueError(
            f"lr_schedule={kind!r} is not one of {SCHEDULES}. "
            f"An unimplemented schedule silently falls back to the default, which is "
            f"how a sweep ends up with three identical arms."
        )

    if kind == 'constant':
        # The only schedule that never lowers the LR. Needed as a control: the other
        # three all decay, so a study over schedules could not otherwise tell "decaying
        # at the wrong moment hurts" from "decaying at all hurts". LambdaLR with a
        # unit factor rather than a no-op keeps every group's initial_lr bookkeeping
        # identical to the warmup branch, so the dynamics group's lr_mult survives.
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _e: 1.0)
        return {"scheduler": sched, "interval": "epoch", "frequency": 1}

    if kind == 'plateau':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=hparams.get('scheduler_patience', 2),
            threshold=hparams.get('scheduler_threshold', 1e-4),
            threshold_mode='rel', min_lr=1e-7)
        return {"scheduler": sched, "monitor": monitor,
                "interval": "epoch", "frequency": 1, "strict": False}

    fn = _warmup_lambda(kind,
                        hparams.get('lr_warmup_epochs', 3),
                        hparams.get('max_epochs', 50),
                        hparams.get('lr_min_factor', 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
    # no 'monitor': LambdaLR is not metric-driven, and passing one makes Lightning
    # wait on a metric that may not be logged on every epoch.
    return {"scheduler": sched, "interval": "epoch", "frequency": 1}
