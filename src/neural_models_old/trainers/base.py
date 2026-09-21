from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        epochs: int,
        logger: Any | None = None,
        scheduler: Any | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 1e-4,
        print_every: int = 1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.epochs = epochs
        self.logger = logger
        self.scheduler = scheduler
        self.patience = early_stopping_patience
        self.min_delta = early_stopping_min_delta
        self.print_every = print_every
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

        if self.logger is not None and hasattr(self.logger, "watch_model"):
            self.logger.watch_model(model)

    def train(self, log_every: int = 10) -> None:
        if self.print_every > 0:
            print("Starting training...")

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                loss = self._train_step(batch)
                epoch_loss += loss.item()

                if batch_idx % log_every == 0 and self.logger:
                    self.logger.log_metrics(
                        {
                            "batch_loss": loss.item(),
                            "epoch": epoch,
                            "batch": batch_idx,
                        }
                    )

            if self.scheduler:
                self.scheduler.step()

            avg_train_loss = epoch_loss / len(self.train_loader)

            train_metrics = self.evaluate(self.train_loader)
            test_metrics = self.evaluate(self.test_loader)

            metrics = {
                "train_loss": train_metrics.get("loss", avg_train_loss),
                "train_accuracy": train_metrics.get("accuracy", 0.0),
                **{f"train_{k}": v for k, v in train_metrics.items() if k != "loss"},
                "test_loss": test_metrics.get("loss", float("inf")),
                "test_accuracy": test_metrics.get("accuracy", 0.0),
                **{f"test_{k}": v for k, v in test_metrics.items() if k != "loss"},
            }

            self._log_epoch(epoch, metrics)
            if self.print_every > 0 and epoch % self.print_every == 0:
                self._print_epoch(epoch, metrics)

            test_loss = metrics["test_loss"]
            test_acc = metrics["test_accuracy"]

            if self.patience is not None:
                if test_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = test_loss
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience or test_acc >= 1.0:
                        if self.print_every > 0:
                            print(f"Early stopping triggered at epoch {epoch}")
                        return

    def evaluate(self, data_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        accumulated_metrics: dict[str, float] = {}
        total_samples = 0
        n_batches = len(data_loader)

        with torch.no_grad():
            for batch in data_loader:
                batch_metrics, batch_size = self._evaluation_step(batch)
                total_samples += batch_size

                for key, value in batch_metrics.items():
                    accumulated_metrics[key] = accumulated_metrics.get(key, 0.0) + value

        return self._finalise_metrics(accumulated_metrics, n_batches, total_samples)

    def test(self) -> dict[str, float]:
        return self.evaluate(self.test_loader)

    @abstractmethod
    def _train_step(self, batch: Any) -> torch.Tensor: ...

    @abstractmethod
    def _evaluation_step(self, batch: Any) -> tuple[dict[str, float], int]: ...

    @abstractmethod
    def _finalise_metrics(
        self, accumulated: dict[str, float], n_batches: int, total_samples: int
    ) -> dict[str, float]: ...

    @abstractmethod
    def _log_epoch(self, epoch: int, metrics: dict[str, float]) -> None: ...

    @abstractmethod
    def _print_epoch(self, epoch: int, metrics: dict[str, float]) -> None: ...
