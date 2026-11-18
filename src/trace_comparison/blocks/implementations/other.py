import numpy as np
from dataclasses import dataclass
import math

from ..block_types import TransformBlock, ModelBlock

class ClassificationWeights(TransformBlock, ModelBlock):
    ...


class ImitationWeights(TransformBlock, ModelBlock):
    ...


@dataclass
class Centroid(TransformBlock):
    """
    input_shape: (solvers, n_puzzles, *feature_dims)
    output_shape: (solvers, prod(feature_dims))
    """

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        self.check_shape(3, input_shape, "min")
        return (input_shape[0], math.prod(input_shape[2:]))

    def _apply(self, x: np.ndarray) -> np.ndarray:
        return np.mean(x, axis=1)
