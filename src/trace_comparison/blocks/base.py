from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from itertools import combinations

import numpy as np


class DataKind(Enum):
    """The array contract a block consumes or produces."""

    JAGGED = "jagged"  # (S, N) object array of float (steps, width, height) cells
    DENSE = "dense"  # floating ndarray with no ragged axis
    MATRIX = "matrix"  # dense square (S, S) solver similarity matrix


def logical_shape(x: np.ndarray, kind: DataKind) -> tuple:
    """`x.shape`, with the ragged step axis of a JAGGED array marked `None`."""
    if kind is not DataKind.JAGGED:
        return x.shape
    return (*x.shape, None, *(x.flat[0].shape[1:] if x.size else ()))


def check_kind(x: object, kind: DataKind, where: str) -> None:
    """Cheap contract check: object dtype <=> JAGGED, and MATRIX must be square."""
    ok = isinstance(x, np.ndarray) and (x.dtype == object) == (kind is DataKind.JAGGED)
    if ok and kind is DataKind.MATRIX:
        assert isinstance(x, np.ndarray)
        ok = x.ndim == 2 and x.shape[0] == x.shape[1]
    if not ok:
        got = f"{getattr(x, 'dtype', type(x).__name__)} {getattr(x, 'shape', '')}"
        raise ValueError(f"{where}: expected {kind.value} array, got {got}")


class Block(ABC):
    input_kind: DataKind = DataKind.DENSE
    output_kind: DataKind = DataKind.DENSE

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]: ...

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    def __call__(self, x: np.ndarray, input_shape: tuple | None = None) -> np.ndarray:
        """Check `x`, run `forward`, check the result against `output_shape`.

        `input_shape` is the pipeline's tracked logical shape (`None` at the
        ragged axis); when omitted it is derived from `x`.
        """
        check_kind(x, self.input_kind, f"{self.name} input")
        expected = self.output_shape(input_shape or logical_shape(x, self.input_kind))
        output = self.forward(x)
        check_kind(output, self.output_kind, f"{self.name} output")
        actual = logical_shape(output, self.output_kind)
        mismatch = any(
            e not in (None, a) for a, e in zip(actual, expected, strict=False)
        )
        if len(actual) != len(expected) or mismatch:
            raise ValueError(f"{self.name}: output shape {actual} != expected {expected}")
        return output

    def apply_combinations(
        self,
        x: np.ndarray,
        fn: Callable[[np.ndarray, np.ndarray], float | np.ndarray],
    ) -> np.ndarray:
        """Fill a symmetric output with `fn(x[i], x[j])` over all pairs (diagonal = 1)."""
        output = np.ones(self.output_shape(x.shape), dtype=np.float32)
        for i, j in combinations(range(len(x)), 2):
            output[i, j] = output[j, i] = fn(x[i], x[j])
        return output
