from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ...types import State
from ..utils.const import UNKNOWN, empty_int8
from ..utils.line_checking import line_marginal

from __future__ import annotations

@dataclass(frozen=True)
class ACState(State):
    """
    grid_population: (k, H, W) int8 of UNKNOWN / EMPTY / FILLED.
    clue_blocks: (row blocks, col blocks), each a tuple of block tuples.
    """

    pop_of_grids: np.ndarray = field(default_factory=empty_int8)
    clue_blocks: tuple[tuple[tuple[int, ...], ...], ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.pop_of_grids.flags.writeable = False

    def _init_params(self, padded_clues: np.ndarray, grid_shape: tuple[int, int]) -> dict:
        return {
            "pop_of_grids": np.full((self.k, *grid_shape), UNKNOWN, dtype=np.int8),
            "clue_blocks": self._clues_to_tuples(padded_clues),
        }

class ACFuseState(ACState):
    """Predict probability of unknown cells."""

    def _fuse_log_odds(self, row_probs: np.ndarray, col_probs: np.ndarray) -> np.ndarray:
        both_filled = row_probs * col_probs
        both_empty = (1.0 - row_probs) * (1.0 - col_probs)
        denom = both_filled + both_empty
        mean = np.clip((row_probs + col_probs) / 2.0, 0.0, 1.0)
        safe_denom = np.where(denom > 0, denom, 1.0)
        return np.where(denom > 0, both_filled / safe_denom, mean)

    @property
    def grids(self) -> np.ndarray:
        k, h, w = self.pop_of_grids.shape
        row_probs = np.empty((k, h, w))
        col_probs = np.empty((k, h, w))
        for pop_idx in range(k):
            for row in range(h):
                known = tuple(self.pop_of_grids[pop_idx, row, :].tolist())
                row_probs[pop_idx, row, :] = line_marginal(
                    self.clue_blocks[0][row], known)
            for col in range(w):
                known = tuple(self.pop_of_grids[pop_idx, :, col].tolist())
                col_probs[pop_idx, :, col] = line_marginal(
                    self.clue_blocks[1][col], known)

        return self._fuse_log_odds(row_probs, col_probs)
