import numpy as np
from trace_comparison.blocks.base import Block

class RMSESimilarity(Block):
    """"
    input_shape: (solvers, ...)
    output_shape: (solvers, solvers, ...)

    Computes the 1 - RMSE between each pair of solvers.
    """

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) < 1:
            raise ValueError(f"Input shape must have at least 1 dimension, got {input_shape}")
        return (input_shape[0], input_shape[0])

    def rmse_sim(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return 1.0 - np.sqrt(np.mean((x1 - x2) ** 2))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.apply_combinations(x, self.rmse_sim)