from trace_comparison.blocks.base import Block
import numpy as np


class Resample(Block):
    """
    input_shape:  (solvers, n_puzzles, steps_jagged, ...)
    output_shape: (solvers, n_puzzles, k, ...)

    Resamples each puzzle's trace to a fixed number of steps (k)
    using nearest-neighbour (duplicate) index selection.
    """

    def __init__(self, k: int) -> None:
        self.k = k

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        n_solvers, n_puzzles, _, *remaining_dims = input_shape
        return (n_solvers, n_puzzles, self.k, *remaining_dims)

    def forward(self, x: np.ndarray) -> np.ndarray:
        n_solvers, n_puzzles = x.shape[:2]
        output = np.empty(
            (n_solvers, n_puzzles, self.k) + x.shape[3:], dtype=np.float32
        )
        for solver_idx, solver_traces in enumerate(x):
            for puzzle_idx, trace in enumerate(solver_traces):
                num_steps = trace.shape[0]
                resample_indices = (
                    np.linspace(0, num_steps - 1, self.k).round().astype(int)
                )
                output[solver_idx, puzzle_idx] = trace[resample_indices]
        return output