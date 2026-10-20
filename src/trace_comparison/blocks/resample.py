import numpy as np

from trace_comparison.blocks.base import Block, DataKind


class Resample(Block):
    """
    input_shape:  (solvers, n_puzzles, None, width, height)  # None = ragged step axis
    output_shape: (solvers, n_puzzles, k, width, height)

    Resamples each solver/puzzle trace to a fixed number of steps (k)
    using nearest-neighbour (duplicate) index selection.
    """

    input_kind = DataKind.JAGGED
    output_kind = DataKind.DENSE

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) < 4:
            raise ValueError(f"Input shape must be at least 4D, got {input_shape}")
        n_solvers, n_puzzles, _, *remaining_dims = input_shape
        return (n_solvers, n_puzzles, self.k, *remaining_dims)

    def forward(self, x: np.ndarray) -> np.ndarray:
        n_solvers, n_puzzles = x.shape
        first = next(iter(x.flat), None)
        if first is None:
            raise ValueError("Resample cannot densify an empty dataset")
        output = np.empty((n_solvers, n_puzzles, self.k, *first.shape[1:]), dtype=first.dtype)
        for solver_idx, solver_traces in enumerate(x):
            for puzzle_idx, trace in enumerate(solver_traces):
                if trace.shape[0] == 0:
                    raise ValueError(f"Trace ({solver_idx}, {puzzle_idx}) has no steps to resample")
                resample_indices = np.linspace(0, trace.shape[0] - 1, self.k).round().astype(int)
                output[solver_idx, puzzle_idx] = trace[resample_indices]
        return output
