from typing import Any

import torch

from neural_models_old.trainers.base import BaseTrainer


class NNGRewardTrainer(BaseTrainer):
    def _train_step(self, batch: Any) -> torch.Tensor:
        inputs = batch[0].to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss, _, _, _ = self.criterion(outputs, inputs)
        loss.backward()
        self.optimizer.step()
        return loss

    def _evaluation_step(self, batch: Any) -> tuple[dict[str, float], int]:
        inputs = batch[0].to(self.device)
        prediction = self.model(inputs)
        loss, row_loss, col_loss, clue_match = self.criterion(prediction, inputs)

        batch_size = inputs.size(0)
        return {
            "loss": loss.item(),
            "row_loss": row_loss.item(),
            "col_loss": col_loss.item(),
            "clue_match_sum": clue_match.sum().item(),
            "correct_sum": (clue_match == 1.0).sum().item(),
        }, batch_size

    def _finalise_metrics(
        self, accumulated: dict[str, float], n_batches: int, total_samples: int
    ) -> dict[str, float]:
        return {
            "test_accuracy": accumulated["correct_sum"] / total_samples
            if total_samples > 0
            else 0.0,
            "test_loss": accumulated["loss"] / n_batches,
            "row_loss": accumulated["row_loss"] / n_batches,
            "col_loss": accumulated["col_loss"] / n_batches,
            "clue_match_percent": accumulated["clue_match_sum"] / total_samples
            if total_samples > 0
            else 0.0,
        }

    def _log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        if self.logger:
            self.logger.log_metrics(
                {
                    "train_loss": metrics.get("train_loss", 0.0),
                    "test_loss": metrics.get("test_loss", 0.0),
                    "test_accuracy": metrics.get("test_accuracy", 0.0),
                    "clue_match_percent": metrics.get("clue_match_percent", 0.0),
                    "row_loss": metrics.get("row_loss", 0.0),
                    "col_loss": metrics.get("col_loss", 0.0),
                    "epoch": epoch,
                }
            )

    def _print_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        test_acc = metrics.get("test_accuracy", 0.0) * 100
        print(
            f"Epoch {epoch:02d} | Train loss: {metrics.get('train_loss', 0.0):.4f} "
            f"| Test loss: {metrics.get('test_loss', 0.0):.4f} | Test accuracy: {test_acc:.3f}% "
            f"| Clue-match: {metrics.get('clue_match_percent', 0.0):.2f}%"
        )
