import math
import lightning as L


class NaNDetectionCallback(L.Callback):
    """
    Monitors training loss for NaN values and terminates training early.

    When running under Optuna, raises TrialPruned so the trial is marked as
    pruned (not completed) and doesn't pollute the study with NaN scores.
    Without Optuna, stops training via trainer.should_stop.

    Args:
        patience: Number of NaN batches tolerated before terminating.
                  Default 3 allows for rare transient NaNs from stochastic noise.
        monitor: The metric to check at epoch end (in addition to batch-level checks).
    """

    def __init__(self, patience=3, monitor="train_loss", optuna_trial=None):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.optuna_trial = optuna_trial
        self.nan_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Check every batch for NaN loss."""
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is not None:
            loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
            if math.isnan(loss_val) or math.isinf(loss_val):
                self.nan_count += 1
        elif outputs is None:
            # training_step returned None (NaN skip) — count it as a NaN occurrence
            self.nan_count += 1

        # Also check the model's internal NaN counter (set by training_step)
        model_nan_count = getattr(pl_module, '_nan_count', 0)
        total_nans = max(self.nan_count, model_nan_count)

        if total_nans >= self.patience:
            msg = (f"🚨 NaN detected {total_nans} times (patience={self.patience}). "
                   f"Terminating training at epoch {trainer.current_epoch}, "
                   f"batch {batch_idx}.")
            print(f"\n{'='*60}")
            print(msg)
            print(f"{'='*60}\n")

            if self.optuna_trial is not None:
                import optuna
                raise optuna.exceptions.TrialPruned(msg)

            trainer.should_stop = True

    def on_train_epoch_end(self, trainer, pl_module):
        """Check epoch-level monitored metric for NaN."""
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return

        val = metric.item() if hasattr(metric, 'item') else float(metric)
        if math.isnan(val) or math.isinf(val):
            self.nan_count += 1
            msg = (f"🚨 {self.monitor} is NaN/Inf at epoch {trainer.current_epoch}. "
                   f"NaN count: {self.nan_count}/{self.patience}")
            print(msg)

            if self.nan_count >= self.patience:
                if self.optuna_trial is not None:
                    import optuna
                    raise optuna.exceptions.TrialPruned(
                        f"Pruned: {self.monitor} NaN for {self.nan_count} occurrences")

                trainer.should_stop = True