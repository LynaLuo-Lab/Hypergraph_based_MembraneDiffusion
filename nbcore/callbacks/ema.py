"""Exponential moving average of the weights, as a pure OBSERVER.

WHY AN OBSERVER. The EMA shadow is never read back into training -- the optimizer
steps the live weights and nothing else. So this cannot change the trajectory or hurt
the base model; it only produces an ADDITIONAL candidate checkpoint to score. An
earlier attempt at EMA in this project was reported to hurt, which is what happens
when the averaged weights are fed back into training or when selection happens on the
EMA alone. Neither is done here: both candidates are kept and scored side by side.

WHY IT MIGHT HELP THE THING WE ACTUALLY CARE ABOUT. Training-seed spread on this
model is 14-40% CV while rollout noise is 0.3-1.3% (eval512, six arms, all
"training dominates"), and within one arm the best seed is 2.7x the worst. Weight
averaging is the standard lever against exactly that, and it costs nothing at train
time.

PICK THE DECAY FROM THE STEP COUNT, NOT FROM HABIT. These runs are SHORT: the
diffusion sweep is 960 samples/epoch at bs 64 for 25 epochs = 375 optimizer steps, the
membrane one 600. An EMA with coefficient d averages over roughly 1/(1-d) steps, so the
usual 0.999 (1000 steps) or 0.9999 (10000) is longer than the entire run -- the shadow
then degenerates to a uniform mean over every step taken, INCLUDING the untrained early
ones, which is a good way to make EMA look harmful. With the debias below, the decay
only starts to bind after n > (10d-1)/(1-d) steps:

    decay 0.9 -> 80 steps    0.95 -> 170    0.98 -> 440    0.999 -> 8990

so at 375 steps anything at or above 0.98 is indistinguishable from a plain running
mean. At this length the meaningful range is 0.8-0.98, i.e. "average the last 5 to 50
steps" -- with 15 steps per epoch, the last 0.3 to 3.3 epochs.

WHY WEIGHTS-ONLY, PER EPOCH, AND SCORED OFFLINE. Scoring the shadow inside the
training loop would double the stability rollout every epoch. Instead this writes the
averaged weights beside the normal checkpoints and the comparison runs afterwards,
through the same eval path the live weights get -- so the EMA-vs-live comparison is
PAIRED within a run and carries none of the seed variance that makes unpaired
comparisons on this model useless.

SPECTRAL-NORM u, v ARE RECOMPUTED AT WRITE TIME. The power-iteration vectors have sign
and rotation freedom, so their average shrinks (|u| ~ 0.8) and sigma = u.Wv under-reads
the top singular value of the averaged weight, inflating W/sigma by up to 1.5x. The
saved copy gets the top singular pair of the averaged weight, signed to match the live u.
"""
import os

import torch
from lightning.pytorch.callbacks import Callback


def _refresh_spectral_uv(sd, live):
    """Set each spectral-norm _u, _v in sd to the top singular pair of its averaged weight."""
    suffix = 'parametrizations.weight.original'
    for k in list(sd):
        if not k.endswith(suffix):
            continue
        base = k[:-len('original')]
        ku, kv = base + '0._u', base + '0._v'
        if ku not in sd or kv not in sd:
            continue
        w = sd[k].float()
        mat = w.flatten(1)                                # dim 0, as torch's _SpectralNorm
        U, _, Vh = torch.linalg.svd(mat, full_matrices=False)
        u, v = U[:, 0], Vh[0]
        lu = live.get(ku)
        if lu is not None and torch.dot(u, lu.detach().float().to(u.device)) < 0:
            u, v = -u, -v
        sd[ku] = u.to(sd[ku].dtype)
        sd[kv] = v.to(sd[kv].dtype)


class EMACallback(Callback):
    """Maintain an EMA of the model weights and checkpoint it each epoch.

    Args:
        decay: EMA coefficient. 0 disables the callback entirely.
        dirpath: where to write ema-epoch=NNN.ckpt. Defaults beside ModelCheckpoint.
        start_epoch: skip the early epochs, whose weights are far from converged and
            would otherwise dominate the average for a long time.
        every_n_epochs: write cadence. The shadow still updates every step.
    """

    def __init__(self, decay=0.999, dirpath=None, start_epoch=0, every_n_epochs=1):
        super().__init__()
        # Several decays at once. Holding k shadows costs k * (weights) of memory and
        # k fused mul_/add_ per step -- nothing beside a forward/backward -- and it
        # removes a real confound: with one decay, "EMA did not help" is indistinguish-
        # able from "that decay was wrong for this run length".
        decays = [decay] if isinstance(decay, (int, float)) else list(decay)
        for d in decays:
            if not 0.0 <= float(d) < 1.0:
                raise ValueError(f"ema_decay must be in [0,1), got {d}")
        self.decays = [float(d) for d in decays if float(d) > 0.0]
        self.decay = self.decays[0] if self.decays else 0.0
        self.dirpath = dirpath
        self.start_epoch = int(start_epoch)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.shadows = {}
        self._n = 0

    @property
    def enabled(self):
        return bool(self.decays)

    @property
    def shadow(self):
        """The first decay's shadow, for single-decay use and the tests."""
        return self.shadows.get(self.decay)

    def _resolve_dir(self, trainer):
        if self.dirpath:
            return self.dirpath
        for cb in trainer.callbacks:                     # sit beside ModelCheckpoint
            dp = getattr(cb, 'dirpath', None)
            if dp:
                return dp
        return trainer.default_root_dir or '.'

    def on_train_start(self, trainer, pl_module):
        if not self.enabled:
            return
        base = {k: v.detach().clone().float()
                for k, v in pl_module.state_dict().items()
                if v.dtype.is_floating_point}
        self.shadows = {d: {k: v.clone() for k, v in base.items()} for d in self.decays}
        self._n = 0

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.enabled or trainer.current_epoch < self.start_epoch:
            return
        if not self.shadows:
            self.on_train_start(trainer, pl_module)
        self._n += 1
        sd = pl_module.state_dict()
        warm = (1.0 + self._n) / (10.0 + self._n)
        for decay, shadow in self.shadows.items():
            # Debiased warmup: a raw EMA started from the init weights carries that
            # init for ~1/(1-decay) steps, which at decay=0.999 is 1000 steps -- most
            # of an epoch here. min() makes the average behave like a running mean
            # until the nominal decay is the tighter constraint.
            d = min(decay, warm)
            for k, sh in shadow.items():
                v = sd.get(k)
                if v is not None:
                    sh.mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.enabled or not self.shadows:
            return
        if trainer.current_epoch < self.start_epoch:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs and \
                trainer.current_epoch != trainer.max_epochs - 1:
            return
        out = self._resolve_dir(trainer)
        os.makedirs(out, exist_ok=True)
        hp = dict(getattr(pl_module, 'hparams', {}) or {})
        live = pl_module.state_dict()
        for decay, shadow in self.shadows.items():
            tag = f"d{decay:.5f}".rstrip('0').rstrip('.').replace('.', '')
            path = os.path.join(
                out, f"ema{tag}-epoch={trainer.current_epoch:03d}.ckpt")
            # Same shape the eval path expects: a 'state_dict' key holding weights. No
            # optimizer state -- these are scored, never resumed, and carrying it would
            # triple the file for nothing.
            saved = {k: v.clone() for k, v in shadow.items()}
            _refresh_spectral_uv(saved, live)
            torch.save({'state_dict': saved,
                        'epoch': trainer.current_epoch,
                        'ema_decay': decay,
                        'hyper_parameters': hp}, path)

    def state_dict(self):
        return {'shadows': self.shadows, 'n': self._n, 'decays': self.decays}

    def load_state_dict(self, sd):
        self.shadows = sd.get('shadows', {})
        self._n = sd.get('n', 0)
        if sd.get('decays'):
            self.decays = sd['decays']
            self.decay = self.decays[0]
