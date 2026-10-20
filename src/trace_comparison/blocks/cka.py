import numpy as np

from trace_comparison.blocks.base import Block, DataKind


class CKASimilarity(Block):
    """
    input_shape: (solvers, samples, ...)
    output_shape: (solvers, solvers)

    Computes linear CKA between each pair of solvers. Trailing dimensions
    are flattened into a single feature axis, so each solver is treated as a
    (samples, features) matrix.
    """

    input_kind = DataKind.DENSE
    output_kind = DataKind.MATRIX

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) < 2:
            raise ValueError(
                f"Input shape must have at least 2 dimensions (solvers, samples, ...), "
                f"got {input_shape}"
            )
        return (input_shape[0], input_shape[0])

    def cka_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        # a, b are already centered, (samples, features)
        cross = np.linalg.norm(a.T @ b, ord="fro") ** 2
        norm_a = np.linalg.norm(a.T @ a, ord="fro")
        norm_b = np.linalg.norm(b.T @ b, ord="fro")
        return cross / (norm_a * norm_b + self.eps)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Flatten to (solvers, samples, features) and center once, not per pair.
        x = x.reshape(x.shape[0], x.shape[1], -1).astype(np.float64, copy=False)
        x = x - x.mean(axis=1, keepdims=True)
        return self.apply_combinations(x, self.cka_sim)