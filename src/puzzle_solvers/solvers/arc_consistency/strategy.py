from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

import numpy as np

from ...types import Strategy
from ..utils.const import EMPTY, FILLED
from ..utils.line_checking import find_forced_cells_from_line, get_line
from .state import ACState


class _ACBaseStrategy(Strategy, ABC):

    @abstractmethod
    def _step_member(self, rng: np.random.Generator, grid: np.ndarray, blocks: tuple) -> bool:
        ...

    def __call__(self, state: ACState) -> ACState:
        grid = state.pop_of_grids.copy()

        moved = [self._step_member(state.rng, grid[pop_idx], state.clue_blocks) for pop_idx in range(state.k)]
        if not any(moved):
            return state

        return replace(state, pop_of_grids=grid, step=state.step + 1)



class ACParallelStrategy(_ACBaseStrategy):
    """One move: deduce from every row and column using the grid as it was, then apply all."""

    def _step_member(self, rng, grid, blocks) -> bool:
        h, w = grid.shape
        new_fill = np.zeros(grid.shape, dtype=bool)
        new_empty = np.zeros(grid.shape, dtype=bool)

        for axis, n in ((0, h), (1, w)):
            for index in range(n):
                target_line = tuple(get_line(grid, axis, index).tolist())
                fill, empty = find_forced_cells_from_line(
                    blocks[axis][index], target_line)
                fill_view = get_line(new_fill, axis, index)
                empty_view = get_line(new_empty, axis, index)
                fill_view |= fill
                empty_view |= empty

        if (new_fill & new_empty).any() or not (new_fill | new_empty).any():
            return False

        grid[new_fill] = FILLED
        grid[new_empty] = EMPTY
        return True


class ACPerLineStrategy(_ACBaseStrategy):
    """One move: apply the first line, in random order, that lets any cell be set."""

    def _step_member(self, rng, grid, blocks) -> bool:
        h, w = grid.shape
        lines = [(0, i) for i in range(h)] + [(1, j) for j in range(w)]
        for n in rng.permutation(len(lines)):
            axis, index = lines[n]
            target_line = tuple(get_line(grid, axis, index).tolist())
            fill, empty = find_forced_cells_from_line(
                blocks[axis][index], target_line)
            if not (fill | empty).any():
                continue
            line = get_line(grid, axis, index)
            line[fill] = FILLED
            line[empty] = EMPTY
            return True
        return False
