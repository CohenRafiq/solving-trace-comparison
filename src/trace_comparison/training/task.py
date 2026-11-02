"""The :class:`Task` interface: everything that differs between training runs.

The entry point (``python -m trace_comparison.training``) is task-agnostic.  It loads
the dataset, asks the task to build its data and model, runs the shared training loop
(:func:`trace_comparison.training.common.fit`), and saves a checkpoint.  A task is
selected, and its model configured, in yaml (see ``config/task/*.yaml``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from trace_comparison.data.dataset import TraceDataset
from trace_comparison.training.common import (
    Batch,
    FitResult,
    PerSample,
    format_metrics,
)


@dataclass
class TaskData:
    """Train/val tensor tuples plus task-specific metadata (class names, sizes, ...)."""

    train: Batch
    val: Batch
    info: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """Subclass this to add a new kind of model/objective.

    Hydra instantiates it from ``cfg.task`` with ``_recursive_: false``, so ``model`` is
    received as a raw config node and only instantiated once the data (and therefore the
    input/output sizes) is known.
    """

    name: str = "task"

    def __init__(self, model: DictConfig, name: str | None = None) -> None:
        self.model_cfg = model
        if name is not None:
            self.name = name

    # ---- required -----------------------------------------------------------
    @abstractmethod
    def build_data(self, ds: TraceDataset, k: int, val_frac: float, seed: int) -> TaskData: ...

    @abstractmethod
    def model_kwargs(self, data: TaskData) -> dict[str, Any]:
        """Constructor arguments derived from the data (feature sizes, n classes, ...)."""

    @abstractmethod
    def step(self, model: nn.Module, batch: Batch): ...  # -> (loss, per-sample dict)

    @abstractmethod
    def metrics(self, out: PerSample) -> dict[str, float]: ...

    # ---- optional hooks -----------------------------------------------------
    def describe(self, data: TaskData) -> None:
        print(f"{len(data.train[0])} train / {len(data.val[0])} val samples")

    def checkpoint_extras(self, result: FitResult, data: TaskData) -> dict[str, Any]:
        return {}

    def report(self, result: FitResult, data: TaskData) -> None:
        print(f"best val {format_metrics(result.best_val_loss, self.metrics(result.best_val_outputs))}")

    # ---- provided -----------------------------------------------------------
    def build_model(self, data: TaskData) -> tuple[nn.Module, dict[str, Any]]:
        """Instantiate the yaml-configured model; also return what's needed to rebuild it."""
        kwargs = self.model_kwargs(data)
        config = OmegaConf.to_container(self.model_cfg, resolve=True)
        target = config.pop("_target_")
        config.update(kwargs)
        model = instantiate(self.model_cfg, **kwargs)
        return model, {"model_target": target, "model_config": config}