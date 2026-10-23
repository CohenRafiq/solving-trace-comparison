import numpy as np

from trace_comparison.blocks.base import Block, DataKind


class RBFCentroidSimilarity(Block):
    """
    input_shape: (solvers, hidden_dim)
    output_shape: (solvers, solvers)

    Computes the centroid of each solver's data using the mean across all puzzles.
    """

    input_kind = DataKind.DENSE
    output_kind = DataKind.DENSE

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) != 2:
            raise ValueError(f"Input shape must have exactly 2 dimensions, got {input_shape}")
        return (input_shape[0], input_shape[0])

    def forward(self, x: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - x[None, :, :]          # (S, S, d)
        d2 = np.sum(diff ** 2, axis=-1)               # squared Euclidean
        scale = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        return np.exp(-d2 / scale)                     # in (0, 1], 1 on diagonal
