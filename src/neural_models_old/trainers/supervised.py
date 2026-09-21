from typing import Any

import torch

from neural_models_old.criterion.abstain import AbstainLoss
from neural_models_old.trainers.base import BaseTrainer


class NNGSupervisedTrainer(BaseTrainer):
    def _train_step(self, batch: Any) -> torch.Tensor:
        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if outputs.shape != targets.shape:
            outputs = outputs.reshape(targets.shape[0], -1, *targets.shape[1:])
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def _evaluation_step(self, batch: Any) -> tuple[dict[str, float], int]:
        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        outputs = self.model(inputs)

        if outputs.shape != targets.shape:
            outputs = outputs.reshape(targets.shape[0], -1, *targets.shape[1:])

        # Compute loss
        loss = self.criterion(outputs, targets)

        # Handle abstain predictions if using AbstainLoss
        abstain_rate = 0.0
        correct = 0.0
        per_label_acc = 0.0
        batch_size = targets.size(0)

        if isinstance(self.criterion, AbstainLoss):
            # Get abstain mask
            abstain_mask = self.criterion.compute_abstain_mask(outputs)
            abstain_rate = abstain_mask.float().mean().item()

            # Committed guess among black/white (drop abstain channel), strict puzzle-level accuracy
            predictions = outputs[:, :2].argmax(dim=1).float()
            correct = (predictions == targets).flatten(1).all(dim=1).sum().item()
            per_label_acc = (predictions == targets).float().mean().item()
        else:
            # Standard accuracy computation
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct = (predictions == targets).flatten(1).all(dim=1).sum().item()
            per_label_acc = (predictions == targets).float().mean().item()

        return {
            "correct": correct,
            "per_label_acc": per_label_acc,
            "loss": loss.item(),
            "abstain_rate": abstain_rate,
        }, batch_size

    def _finalise_metrics(
        self, accumulated: dict[str, float], n_batches: int, total_samples: int
    ) -> dict[str, float]:
        return {
            "accuracy": accumulated["correct"] / total_samples if total_samples > 0 else 0.0,
            "loss": accumulated["loss"] / n_batches,
            "per_label_accuracy": accumulated["per_label_acc"] / n_batches,
            "abstain_rate": accumulated["abstain_rate"] / n_batches,
        }

    def _log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        if self.logger:
            log_dict = {
                "train_loss": metrics.get("train_loss", 0.0),
                "train_accuracy": metrics.get("train_accuracy", 0.0),
                "test_loss": metrics.get("test_loss", 0.0),
                "test_accuracy": metrics.get("test_accuracy", 0.0),
                "epoch": epoch,
                "train_per_label_accuracy": metrics.get("train_per_label_accuracy", 0.0),
                "test_per_label_accuracy": metrics.get("test_per_label_accuracy", 0.0),
            }
            if "train_abstain_rate" in metrics:
                log_dict["train_abstain_rate"] = metrics["train_abstain_rate"]
            if "test_abstain_rate" in metrics:
                log_dict["test_abstain_rate"] = metrics["test_abstain_rate"]
            self.logger.log_metrics(log_dict)

    def _print_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        train_acc = metrics.get("train_accuracy", 0.0)
        test_acc = metrics.get("test_accuracy", 0.0)
        msg = (
            f"Epoch {epoch:02d}: "
            f"Train Loss={metrics.get('train_loss', 0.0):.4f}, Train Acc={train_acc:.4f} | "
            f"Test Loss={metrics.get('test_loss', 0.0):.4f}, Test Acc={test_acc:.4f}"
        )
        if "train_abstain_rate" in metrics:
            msg += f" | Abstain: {metrics.get('train_abstain_rate', 0.0):.2%}"
        print(msg)
