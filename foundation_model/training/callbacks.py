"""Trainer callbacks for UI-visible progress and validation metrics."""


class FoundationStatusCallback:
    """Small wrapper that creates a Transformers callback lazily."""

    @staticmethod
    def build(status_path, config):
        from transformers import TrainerCallback
        from foundation_model.training.status import write_status

        class _Callback(TrainerCallback):
            def __init__(self):
                self.latest_eval_loss = None
                self.best_checkpoint = None

            def _common(self, state):
                progress = 0.0
                if state.max_steps:
                    progress = min(1.0, max(0.0, float(state.global_step) / float(state.max_steps)))
                return {
                    "state": "running",
                    "progress": progress,
                    "global_step": int(state.global_step),
                    "max_steps": int(state.max_steps or 0),
                    "world_size": config.world_size,
                    "global_batch_size": config.global_batch_size,
                    "base_model": config.base_model,
                    "peft_strategy": config.peft_strategy,
                    "distributed_strategy": config.distributed_strategy,
                    "eval_loss": self.latest_eval_loss,
                    "best_checkpoint": self.best_checkpoint,
                }

            def on_train_begin(self, args, state, control, **kwargs):
                if state.is_world_process_zero:
                    write_status(status_path, **self._common(state))

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not state.is_world_process_zero:
                    return
                logs = dict(logs or {})
                payload = self._common(state)
                payload.update(
                    loss=logs.get("loss"),
                    learning_rate=logs.get("learning_rate"),
                    epoch=logs.get("epoch"),
                )
                write_status(status_path, **payload)

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if not state.is_world_process_zero:
                    return
                metrics = dict(metrics or {})
                value = metrics.get("eval_loss")
                if value is not None:
                    self.latest_eval_loss = float(value)
                self.best_checkpoint = getattr(state, "best_model_checkpoint", None) or self.best_checkpoint
                payload = self._common(state)
                payload["epoch"] = metrics.get("epoch")
                write_status(status_path, **payload)

            def on_save(self, args, state, control, **kwargs):
                if not state.is_world_process_zero:
                    return
                self.best_checkpoint = getattr(state, "best_model_checkpoint", None) or self.best_checkpoint
                write_status(status_path, **self._common(state))

            def on_train_end(self, args, state, control, **kwargs):
                if state.is_world_process_zero:
                    payload = self._common(state)
                    payload.update(state="finalizing", progress=1.0)
                    write_status(status_path, **payload)

        return _Callback()
