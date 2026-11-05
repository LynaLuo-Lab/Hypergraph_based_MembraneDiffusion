"""Report the stability objective to Optuna each epoch and honor pruner decisions."""

import math
import lightning as L


class OptunaReportCallback(L.Callback):
    def __init__(self, optuna_trial, monitor="stab/reported_objective"):
        super().__init__()
        self.optuna_trial = optuna_trial
        self.monitor = monitor

    def on_train_epoch_end(self, trainer, pl_module):
        if self.optuna_trial is None:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        val = metric.item() if hasattr(metric, "item") else float(metric)
        if math.isnan(val) or math.isinf(val):
            return
        self.optuna_trial.report(val, step=trainer.current_epoch)
        if self.optuna_trial.should_prune():
            import optuna
            raise optuna.exceptions.TrialPruned(
                f"Pruned by MedianPruner at epoch {trainer.current_epoch} "
                f"({self.monitor}={val:.4f})"
            )
