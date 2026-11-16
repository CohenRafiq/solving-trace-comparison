import numpy as np
from dataclasses import dataclass

from ..block_types import TransformBlock, ModelBlock

class ClassificationWeights(TransformBlock, ModelBlock):
    ...


class ImitationWeights(TransformBlock, ModelBlock):
    ...


class Centroid(TransformBlock):
    """
    input_shape: (solvers, n_puzzles, emb_dim)
    output_shape: (solvers, emb_dim)
    """

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        self.check_shape(3, input_shape, "min")
        return (input_shape[0], input_shape[1])

    def _apply(self, x: np.ndarray) -> np.ndarray:
        return np.mean(x, axis=1, keepdims=False)
