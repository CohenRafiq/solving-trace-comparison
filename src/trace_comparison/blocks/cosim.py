import numpy as np

from trace_comparison.blocks.base import Block, DataKind


class CosineSimilarity(Block):
    """
    input_shape: (solvers, ...)
    output_shape: (solvers, solvers, ...)

    Computes the cosine similarity between each pair of solvers.
    """

    input_kind = DataKind.DENSE
    output_kind = DataKind.MATRIX

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) < 1:
            raise ValueError(f"Input shape must have at least 1 dimension, got {input_shape}")
        return (input_shape[0], input_shape[0])

    def cosine_sim(self, x1: np.ndarray, x2: np.ndarray) -> float:
        dot_product = np.sum(x1 * x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        if norm_x1 == 0 or norm_x2 == 0:
            return 0.0
        return float(dot_product / (norm_x1 * norm_x2))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.apply_combinations(x, self.cosine_sim)
