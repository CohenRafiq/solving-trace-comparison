from torch import nn

import wandb


class Logger:
    """A simple logger class that wraps around Weights & Biases (wandb)
       for logging metrics and models during training."""

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        save_code: bool = False,
        enabled: bool = True,
        log_every: int = 10,
        should_log_model: bool = False,
        log: str = "gradients",
        log_freq: int = 200,
        config: dict | None = None,
    ):
        self.enabled = enabled
        self.log_every = log_every
        self.should_log_model = should_log_model
        self.log = log
        self.log_freq = log_freq
        self._step = 0

        if self.enabled:
            wandb.init(
                project=project,
                entity=entity,
                name=name,
                tags=tags,
                group=group,
                save_code=save_code,
                config=config,
            )

    @property
    def run_id(self) -> str | None:
        if not self.enabled or wandb.run is None:
            return None
        return wandb.run.id

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        if not self.enabled or self._step % self.log_every != 0:
            self._step += 1
            return
        self._step += 1
        wandb.log(metrics, step=step)

    def log_model(self, model_path: str, name: str = "model"):
        if not self.enabled or not self.should_log_model:
            return
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    def watch_model(self, model: nn.Module, log_freq: int | None = None):
        if self.enabled:
            wandb.watch(model, log=self.log, log_freq=log_freq or self.log_freq)

    def finish(self):
        if self.enabled:
            wandb.finish()
