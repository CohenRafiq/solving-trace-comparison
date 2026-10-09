from __future__ import annotations

from dataclasses import replace
import numpy as np

from ...types import Strategy
from ..utils.const import UNKNOWN, EMPTY, FILLED
from ..utils.line_checking import consistent_at_cell
from .state import BSState, Decision


class BSStrategy(Strategy):
    """Commit one more cell per trajectory, backtracking as far as needed to do it."""

    def __call__(self, state: BSState) -> BSState:
        grid = state.pop_of_grids.copy()
        stacks = []
        for i in range(state.k):
            stack = list(state.stack[i])
            if np.any(grid[i] == UNKNOWN):
                self._commit_one(grid[i], stack, state.clue_blocks, state.rng)
            stacks.append(tuple(stack))

        return replace(state, grid=grid, stack=tuple(stacks), step=state.step + 1)

    def _commit_one(self, grid: np.ndarray, stack: list[Decision], blocks: tuple, rng: np.random.Generator) -> None:
        d = self._pick(grid, rng)
        while True:
            grid[d.row, d.col] = d.value
            if consistent_at_cell(grid, d.row, d.col, blocks):
                stack.append(d)
                return

            if not d.flipped:               # try the other value for this cell
                d = d.flip()
                continue

            grid[d.row, d.col] = UNKNOWN    # both values failed: unwind
            while stack and stack[-1].flipped:
                undone = stack.pop()
                grid[undone.row, undone.col] = UNKNOWN
            if not stack:
                raise ValueError("Search exhausted: puzzle is unsatisfiable.")
            d = stack.pop().flip()          # flip the latest decision with an untried value

    @staticmethod
    def _pick(grid: np.ndarray, rng: np.random.Generator) -> Decision:
        rows, cols = np.nonzero(grid == UNKNOWN)
        i = rng.integers(len(rows))
        return Decision(int(rows[i]), int(cols[i]), int(rng.choice([EMPTY, FILLED])))