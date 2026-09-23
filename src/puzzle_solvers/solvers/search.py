import random
from functools import cache, lru_cache

import numpy as np

from puzzle_solvers.solvers.base import BaseSolver
from utils.grid import ternarise


class BacktrackingSearch(BaseSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    @lru_cache(maxsize=200_000)
    def _line_feasible(vals: tuple, target: tuple) -> bool:
        """Cached across ALL calls, not just within one _check_line invocation."""
        n = len(vals)

        @cache
        def dp(i: int, j: int, run_len: int) -> bool:
            if i == n:
                if run_len > 0:
                    return j == len(target) - 1 and run_len == target[j]
                return j == len(target)

            v = vals[i]

            if v in (0, -1):
                if run_len > 0:
                    if j < len(target) and run_len == target[j]:
                        if dp(i + 1, j + 1, 0):
                            return True
                else:
                    if dp(i + 1, j, 0):
                        return True

            if v in (1, -1):
                if j < len(target) and run_len + 1 <= target[j]:
                    if dp(i + 1, j, run_len + 1):
                        return True

            return False

        return dp(0, 0, 0)

    def _check_line(self, line: np.ndarray, clue: np.ndarray, epsilon: float = 1e-2) -> bool:
        vals = tuple(line)
        target = tuple(int(x) for x in np.asarray(clue).flatten() if int(x) > 0)
        return self._line_feasible(vals, target)

    def _check_consistency(self, grid: np.ndarray, row: int, col: int) -> bool:
        if not self._check_line(grid[row, :], self._clues[0][row]):
            return False
        if not self._check_line(grid[:, col], self._clues[1][col]):
            return False
        return True

    def _reset_state(self, clues: np.ndarray, prev: np.ndarray) -> None:
        """Persistent per-solve state: a single incremental backtracking trajectory."""
        self._clues = np.asarray(clues)
        self._grid = ternarise(np.asarray(prev), epsilon=0)
        self._stack: list[list[int | bool]] = []  # [row, col, next_value_index, committed]
        self._values: dict[tuple[int, int], list[int]] = {}  # (row, col) -> candidates

    def _start_new_cell(self) -> None:
        unknown = np.argwhere(self._grid == -1)
        if len(unknown) == 0:
            return
        r, c = unknown[random.randrange(len(unknown))]
        r, c = int(r), int(c)
        values = [0, 1]
        random.shuffle(values)
        self._stack.append([r, c, 0, False])
        self._values[(r, c)] = values

    def _advance(self, num_moves: int) -> None:
        """Commit `num_moves` cell assignments, one incremental DFS step each."""
        grid = self._grid
        backtrack = False
        for _ in range(num_moves):
            while True:
                if np.all(grid != -1):
                    return
                if not self._stack:
                    self._start_new_cell()
                    backtrack = False
                    continue
                r, c, idx, committed = self._stack[-1]
                values = self._values[(r, c)]
                if backtrack and committed:
                    stack_top = self._stack[-1]
                    stack_top[2] = idx + 1
                    stack_top[3] = False
                    backtrack = False
                    grid[r, c] = -1
                    continue
                if committed:
                    self._start_new_cell()
                    continue
                if idx >= len(values):
                    grid[r, c] = -1
                    self._stack.pop()
                    del self._values[(r, c)]
                    backtrack = True
                    continue
                value = values[idx]
                grid[r, c] = value
                if self._check_consistency(grid, r, c):
                    self._stack[-1][3] = True
                    break
                grid[r, c] = -1
                self._stack[-1][2] = idx + 1

    def _step(
        self, clues: np.ndarray, prev: np.ndarray, num_steps: int, sampling_ratio: float
    ) -> np.ndarray:
        if not hasattr(self, "_grid"):
            self._reset_state(clues, prev)
        self._advance(max(1, num_steps))
        out = np.where(self._grid == -1, 0.5, self._grid.astype(float))
        return np.array([out])
