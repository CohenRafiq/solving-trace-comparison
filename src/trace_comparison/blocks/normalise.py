import numpy as np
from scipy.stats import rankdata

from trace_comparison.blocks.base import Block, DataKind


class Normaliser(Block):
    """
    input_shape: (solvers, n_puzzles, ...)
    output_shape: (solvers, n_puzzles, ...)

    Normalises each grid independently (over all its trailing dims), before any
    pairwise metric is computed. Because the transform depends only on the grid
    itself, identical grids stay identical and remain perfectly similar.

    modes:
        "zscore": mean 0, std 1 per grid. Removes mean and std confounds exactly.
        "rank":   rank-transform per grid, scaled to [0, 1]. Removes mean, std and
                  the whole value distribution (skew, ceiling effects), and stays
                  bounded. Ties get their average rank.
        "minmax": rescale each grid to [0, 1]. Removes range, but not distribution shape.

    Constant grids (zero spread) have no information to normalise; they map to all zeros
    (zscore / minmax) or all 0.5 (rank).
    """

    input_kind = DataKind.DENSE
    output_kind = DataKind.DENSE

    _MODES = ("zscore", "rank", "minmax")

    def __init__(self, mode: str = "zscore", eps: float = 1e-12) -> None:
        if mode not in self._MODES:
            raise ValueError(f"Unknown mode {mode!r}, expected one of {self._MODES}")
        self.mode = mode
        self.eps = eps

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) < 2:
            raise ValueError(f"Input shape must have at least 2 dimensions, got {input_shape}")
        return input_shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        flat = x.reshape(x.shape[0] * x.shape[1], -1).astype(np.float64)  # (N, d)

        if self.mode == "zscore":
            centred = flat - flat.mean(axis=1, keepdims=True)
            std = flat.std(axis=1, keepdims=True)
            out = np.where(std > self.eps, centred / np.maximum(std, self.eps), 0.0)

        elif self.mode == "minmax":
            lo = flat.min(axis=1, keepdims=True)
            rng = flat.max(axis=1, keepdims=True) - lo
            out = np.where(rng > self.eps, (flat - lo) / np.maximum(rng, self.eps), 0.0)

        else:  # rank
            d = flat.shape[1]
            ranks = rankdata(flat, axis=1)  # 1..d, ties averaged
            out = (ranks - 1) / max(d - 1, 1)  # scale to [0, 1]
            constant = np.ptp(flat, axis=1) <= self.eps
            out[constant] = 0.5

        return out.reshape(x.shape)
