from abc import ABC, abstractmethod

from typing import Callable
import numpy as np
from itertools import combinations

class Block(ABC):

    def apply_combinations(self, x: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        """Apply the block to all combinations of the input array."""
        output = np.ones(self.output_shape(x.shape), dtype=np.float32)
        for i, j in combinations(range(x.shape[0]), 2):
            output[i, j] = fn(x[i], x[j])
            output[j, i] = output[i, j]
        return output

    @abstractmethod
    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        ...

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        ...

    def __call__(self, x: np.ndarray) -> np.ndarray:
        output = self.forward(x)
        if output.shape != self.output_shape(x.shape):
            raise ValueError(
                f"Output shape {output.shape} does not match "
                f"computed output shape {self.output_shape(x.shape)}"
            )
        return output
