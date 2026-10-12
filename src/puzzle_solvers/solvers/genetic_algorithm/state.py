from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from utils.grid import clues_to_tuples
from ...types import State
from ..utils.const import empty_bool


@dataclass(frozen=True)
class GAState(State):
    """
    population: (k, H, W) bool of candidate solutions.
    blocks:     clues as block tuples.
    """

    pop_of_grids: np.ndarray = field(default_factory=empty_bool)
    clue_blocks: tuple[tuple[tuple[int, ...], ...], ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.pop_of_grids.flags.writeable = False

    def _init_params(self, padded_clues: np.ndarray, grid_shape: tuple[int, int]) -> dict:
        if self.k < 2:
            raise ValueError("GAState requires k >= 2 for a population.")
        density = padded_clues.sum() / (2 * grid_shape[0] * grid_shape[1])
        return {
            "pop_of_grids": self.rng.random((self.k, *grid_shape)) < density,
            "clue_blocks": clues_to_tuples(padded_clues),
        }
        
