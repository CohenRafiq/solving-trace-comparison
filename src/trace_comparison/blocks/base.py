from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal
import numpy as np


class DataKind(Enum):
    """The array contract a block consumes or produces."""

    JAGGED = "jagged"    # (S, N) object array of float (steps, width, height) cells
    DENSE = "dense"      # floating ndarray with no ragged axis
    MATRIX = "matrix"    # dense square (S, S) solver similarity matrix

    def check(self, x: object, where: str) -> None:
        """Cheap contract check: object dtype <=> JAGGED, and MATRIX must be square."""
        ok = isinstance(x, np.ndarray) and (x.dtype == object) == (self is DataKind.JAGGED)
        
        if ok and self is DataKind.MATRIX:
            ok = x.ndim == 2 and x.shape[0] == x.shape[1]
        
        if not ok:
            dtype = getattr(x, 'dtype', type(x).__name__)
            shape = getattr(x, 'shape', '')
            raise ValueError(f"{where}: expected {self.value} array, got {dtype} {shape}")


    def shape(self, x: np.ndarray) -> tuple:
        """`x.shape`, with the ragged step axis of a JAGGED array marked `None`."""
        if self is not DataKind.JAGGED:
            return x.shape
        return (*x.shape, None, *(x.flat[0].shape[1:] if x.size else ()))

class Block(ABC):
    input_kind: DataKind
    output_kind: DataKind
    eps: float = 1e-8

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]: ...

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Check `x`, run `forward`, check the result against `output_shape`.

        `input_shape` is the pipeline's tracked logical shape (`None` at the
        ragged axis); when omitted it is derived from `x`.
        """
        self.input_kind.check(x, f"{self.name} input")
        expected = self.output_shape(self.input_kind.shape(x))
        output = self.forward(x)
        self.output_kind.check(output, f"{self.name} output")
        actual = self.output_kind.shape(output)
        mismatch = any(
            e not in (None, a) for a, e in zip(actual, expected, strict=False)
        )
        if len(actual) != len(expected) or mismatch:
            raise ValueError(f"{self.name}: output shape {actual} != expected {expected}")
        return output

    def get_context(self) -> dict[str, object]:
        """Return a dict of context variables to share with other blocks."""
        return {}

    def set_context(self, context: dict[str, object]) -> None:
        """Receive a dict of context variables from other blocks."""
        pass

    def check_shape(self, n: int,
                    input_shape: tuple[int, ...],
                    type: Literal["min", "exact"]
                    ) -> None:
        if len(input_shape) < n and type == "min":
            raise ValueError(f"Input shape must have at least {n} dimensions, got {input_shape}")
        if len(input_shape) != n and type == "exact":
            raise ValueError(f"Input shape must have exactly {n} dimensions, got {input_shape}")