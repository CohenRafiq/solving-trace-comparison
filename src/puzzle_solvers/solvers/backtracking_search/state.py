from __future__ import annotations

from dataclasses import dataclass, field, replace
import numpy as np

from utils.grid import clues_to_tuples
from ...types import State
from ..utils.const import UNKNOWN, EMPTY, FILLED, empty_int8


@dataclass(frozen=True)
class Decision:
    """A placed cell. `flipped` means the alternative is already in use, so nothing is left to try."""
    row: int
    col: int
    value: int
    flipped: bool = False

    def flip(self) -> Decision:
        other = FILLED if self.value == EMPTY else EMPTY
        return replace(self, value=other, flipped=True)


@dataclass(frozen=True)
class BSState(State):
    """
    grid:   (k, H, W) int8 of UNKNOWN / EMPTY / FILLED.
    blocks: (row blocks, col blocks), each a tuple of block tuples.
    stack:  per trajectory, the decisions made so far, oldest first.
    """
    pop_of_grids: np.ndarray = field(default_factory=empty_int8)
    clue_blocks: tuple[tuple[tuple[int, ...], ...], ...] = ()
    stack: tuple[tuple[Decision, ...], ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.pop_of_grids.flags.writeable = False

    def _init_params(self, padded_clues: np.ndarray, grid_shape: tuple[int, int]) -> dict:
        return {
            "pop_of_grids": np.full((self.k, *grid_shape), UNKNOWN, dtype=np.int8),
            "clue_blocks": clues_to_tuples(padded_clues),
            "stack": tuple(() for _ in range(self.k)),
        }