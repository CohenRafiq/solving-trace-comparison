from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np
import math
import torch
from torch import nn
from pathlib import Path
from typing import Any, ClassVar
from hydra.utils import get_class

from .base import Block, DataKind

@dataclass
class TransformBlock(Block, ABC):
    input_kind = DataKind.DENSE
    output_kind = DataKind.DENSE

    @abstractmethod
    def _apply(self, x: np.ndarray) -> np.ndarray: ...

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x.reshape(x.shape[0], x.shape[1], -1)
        return self._apply(out).reshape(self.output_shape(x.shape))

@dataclass
class SameShapeBlock(TransformBlock, ABC):
    """
    input_shape: (solvers, n_puzzles, ...)
    output_shape: (solvers, n_puzzles, ...)
    """

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        self.check_shape(3, input_shape, "min")
        return input_shape

@dataclass
class TraceEmbeddingBlock(TransformBlock, ABC):
    """
    input_shape: (solvers, n_puzzles, ...)
    output_shape: (solvers, n_puzzles, emb_dim)
    """
    emb_dim: int = -1

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        self.check_shape(3, input_shape, "min")
        return (input_shape[0], input_shape[1], self.emb_dim)

@dataclass
class ComparisonBlock(Block, ABC):
    """
    input_shape: (solvers, n_puzzles, ...)
    output_shape: (solvers, solvers, ...)
    """

    input_kind = DataKind.DENSE
    output_kind = DataKind.MATRIX

    is_distance: ClassVar[bool] = False

    @property
    def diagonal(self) -> float:
        return 0.0 if self.is_distance else 1.0

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        self.check_shape(2, input_shape, "min")
        return (input_shape[0], input_shape[0])

    @abstractmethod
    def _apply(self, a: np.ndarray, b: np.ndarray) -> float: ...

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(x.shape[0], x.shape[1], -1)
        preprocessed = self._preprocess(x)
        if preprocessed is None:
            return np.full(self.output_shape(x.shape), 1.0 - self.diagonal, dtype=np.float32)
        output = np.full(self.output_shape(x.shape), self.diagonal, dtype=np.float32)
        for i, j in combinations(range(len(x)), 2):
            output[i, j] = output[j, i] = self._apply(preprocessed[i], preprocessed[j])
        return output
    
    def get_context(self) -> dict[str, object]:
        return {"matrix_is_distance": self.is_distance}

@dataclass
class AlignmentBlock(Block, ABC):
    """
    input_shape:  (solvers, n_puzzles, None, width, height)  # None = ragged step axis
    output_shape: (solvers, n_puzzles, k, width, height)
    """

    k: int | None = None

    input_kind = DataKind.JAGGED
    output_kind = DataKind.DENSE

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        self.check_shape(5, input_shape, "exact")
        n_solvers, n_puzzles, _, *remaining_dims = input_shape
        return (n_solvers, n_puzzles, self.k, *remaining_dims)

    def _output_dtype(self, x: np.ndarray) -> np.dtype:
        return next(iter(x.flat)).dtype

    @abstractmethod
    def _apply(self, x: np.ndarray) -> np.ndarray: ...

    def _prepare(self, x: np.ndarray) -> None: ...

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._prepare(x)
        output = np.empty(self.output_shape(self.input_kind.shape(x)), dtype=self._output_dtype(x))
        for idx, trace in np.ndenumerate(x):
            output[idx] = self._apply(trace)
        return output

    def set_context(self, context: dict) -> None:
        if self.k is not None:
            return

        # Will throw error if key not in context
        model_input_dim = context["model_input_dim"]
        grid_numel = math.prod(context["grid_shape"])

        k, remainder = divmod(model_input_dim, grid_numel)
        if remainder:
            raise ValueError(
                f"{type(self).__name__}: cannot infer k, model_input_dim "
                f"({model_input_dim}) is not divisible by grid_numel ({grid_numel})"
            )
        if k < 1:
            raise ValueError(f"{type(self).__name__} requires k >= 1, got {self.k}")
        self.k = k

@dataclass
class MatrixNormBlock(Block, ABC):
    """
    input_shape:  (solvers, solvers)
    output_shape: (solvers, solvers), values in [0, 1]
    """
    input_kind = DataKind.MATRIX
    output_kind = DataKind.MATRIX

    is_distance: bool | None = None  # None -> take from context; explicit value overrides

    def output_shape(self, input_shape):
        self.check_shape(2, input_shape, "exact")
        return input_shape

    def set_context(self, context):
        if self.is_distance is None:
            self.is_distance = context.get("matrix_is_distance")  # may stay None

    def forward(self, x):
        if self.is_distance is None:
            raise ValueError(f"{self.name}: unknown whether input is similarity or distance; "
                             "set is_distance or place a ComparisonBlock upstream")
        out = x.astype(np.float32, copy=True)
        i, j = np.triu_indices_from(x, k=1)
        vals = self._apply(x[i, j])          # -> [0, 1]
        if self.is_distance:
            vals = 1.0 - vals
        out[i, j] = out[j, i] = vals
        np.fill_diagonal(out, 1.0)           # output is always similarity
        return out

@dataclass
class ModelBlock(Block, ABC):

    checkpoint_dir: str
    model_type: str
    device: str = "cpu"
    model_version: str | None = None

    model: nn.Module = field(init=False)
    payload: dict = field(init=False)

    def __post_init__(self):
        self.model, self.payload = self.load_model()

    @property
    def model_path(self) -> str:
        model_dir = Path(self.checkpoint_dir) / self.model_type
        if not model_dir.exists():
            raise ValueError(f"Directory {model_dir} does not exist")
        if self.model_version is not None:
            return model_dir / f"{self.model_version}.pt"
        latest = max((p for p in model_dir.glob("*.pt") if p.stem.isdigit()), default=None)
        if latest is None:
            raise ValueError(f"No checkpoints found in {model_dir}")
        return latest

    def load_model(self) -> tuple[nn.Module, dict[str, Any]]:
        payload = torch.load(self.model_path, map_location=self.device, weights_only=True)
        target = payload.get("model_target", None)
        if target is None:
            raise ValueError("checkpoint has no 'model_target' and no default was given")
        model = get_class(target)(**payload["model_config"])
        model.load_state_dict(payload["model_state_dict"])
        model.to(self.device).eval()
        return model, payload