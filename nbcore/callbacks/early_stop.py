"""Early stopping on the stability objective, delayed past any curriculum warmup.

WHY IT EXISTS. The curriculum ablation (3 seeds, paired) measured that BOTH arms
overfit hard past their peak: curriculum 11.57 -> 14.89 and flat 9.27 -> 16.39 on
stab/reported_objective, with one flat seed going 8.24 -> 17.27. Selection is on
best-checkpoint so that costs no accuracy, but it does mean a large share of every
run is spent training past the point of use.

WHY IT IS DELAYED. Plain EarlyStopping is a trap under a seq_len curriculum: during
the ramp the model is scored on SHORTER rollouts than it will be deployed on, so the
objective can plateau or worsen for reasons that have nothing to do with convergence,
and the run dies before ever seeing a full-length sequence. `start_epoch` suppresses
stopping until the curriculum has finished. With the flat schedule (warmup 0) this
degenerates to ordinary EarlyStopping, which is the intended common case.
"""
from lightning.pytorch.callbacks import EarlyStopping


class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, *args, start_epoch: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = int(start_epoch)

    def _should_skip_check(self, trainer) -> bool:
        if trainer.current_epoch < self.start_epoch:
            return True
        return super()._should_skip_check(trainer)
