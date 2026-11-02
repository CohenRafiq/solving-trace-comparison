from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from trace_comparison.training.common import FitResult, PerSample
from trace_comparison.data.data import flatten_traces, select, split_by_puzzle
from trace_comparison.training.task import Task, TaskData



# --------------------------------------------------------------------- model


class MLPClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_classes: int,
        hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.latent_dim = hidden

    def forward(
        self,
        x: torch.Tensor,
        solver_id=None,          # kept for API compatibility
        return_hidden: bool = False,
    ):
        h = self.drop(self.act(self.fc1(self.flatten(x))))
        out = self.fc2(h)
        return (out, [h]) if return_hidden else out

    @torch.no_grad()
    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self(x, return_hidden=True)[1][-1]


# ---------------------------------------------------------------------- task


class ClassifierTask(Task):
    name = "classifier"

    def __init__(self, model: DictConfig, name: str | None = None, channel_first: bool = False):
        super().__init__(model, name)
        self.channel_first = channel_first

    @staticmethod
    def confusion_matrix(out: PerSample, n_classes: int) -> torch.Tensor:
        """``(true, pred)`` counts."""
        return torch.bincount(out["target"] * n_classes + out["pred"], minlength=n_classes**2).view(
            n_classes, n_classes
        )

    def build_data(self, ds, k, val_frac, seed) -> TaskData:
        flat = flatten_traces(ds, k)
        x = flat.traces.permute(0, 1, 3, 2).contiguous() if self.channel_first else flat.traces  # (C, H, W)
        train_idx, val_idx = split_by_puzzle(flat.puzzle_idx, flat.n_puzzles, val_frac, seed)
        data = (x, flat.solver_idx)  # x (M, k, W, H), y (M,)
        return TaskData(
            select(data, train_idx),
            select(data, val_idx),
            {"classes": list(flat.solvers), "n_classes": len(flat.solvers), "in_features": x[0].numel()},
        )

    def model_kwargs(self, data: TaskData) -> dict[str, Any]:
        return {"in_features": data.info["in_features"], "n_classes": data.info["n_classes"]}

    def step(self, model, batch):
        x, y = batch
        logits = model(x)
        return F.cross_entropy(logits, y), {"pred": logits.argmax(1), "target": y}

    def metrics(self, out):
        return {"acc": (out["pred"] == out["target"]).float().mean().item()}

    def describe(self, data: TaskData) -> None:
        n = data.info["n_classes"]
        print(
            f"{len(data.train[0])} train / {len(data.val[0])} val samples, {n} solvers "
            f"(chance {1 / n:.3f}), input {tuple(data.train[0].shape[1:])}"
        )

    def checkpoint_extras(self, result: FitResult, data: TaskData) -> dict[str, Any]:
        out = result.best_val_outputs
        return {
            "classes": data.info["classes"],
            "metrics": {"val_loss": result.best_val_loss, "val_acc": self.metrics(out)["acc"]},
            "val_confusion": self.confusion_matrix(out, data.info["n_classes"]),
        }

    def report(self, result: FitResult, data: TaskData) -> None:
        super().report(result, data)
        print("Best validation results:")
        print(f"  Loss: {result.best_val_loss:.4f}")
        print(f"  Accuracy: {self.metrics(result.best_val_outputs)['acc']:.4f}")
        confusion = self.confusion_matrix(result.best_val_outputs, data.info["n_classes"])
        per_class = (confusion.diag() / confusion.sum(1).clamp(min=1)).tolist()
        print("per-solver val accuracy:")
        for name, a in sorted(zip(data.info["classes"], per_class), key=lambda t: t[1]):
            print(f"  {name:<12} {a:.3f}")