import numpy as np

from trace_comparison.blocks.base import Block, DataKind


class Centroid(Block):
    """
    input_shape: (solvers, n_puzzles, hidden_dim)
    output_shape: (solvers, hidden_dim)

    Computes the centroid of each solver's data using the mean across all puzzles.
    """

    input_kind = DataKind.DENSE
    output_kind = DataKind.DENSE

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) != 3:
            raise ValueError(f"Input shape must have exactly 3 dimensions, got {input_shape}")
        return (input_shape[0], input_shape[2])

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.mean(x, axis=1, keepdims=False)