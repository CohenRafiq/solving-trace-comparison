from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseLogger(ABC):
    @abstractmethod
    def setup(self, cfg: Any = None): ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None): ...

    @abstractmethod
    def log_model(self, model_path: str, name: str = "model"): ...

    @abstractmethod
    def watch_model(self, model: torch.nn.Module, log_freq: int = 100): ...

    @abstractmethod
    def finish(self): ...
