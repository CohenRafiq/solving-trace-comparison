import time
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from utils.grid import blank_grid, grid_shape_from_clues, normalise_clues


class StoppingReason(Enum):
    NOT_STOPPED = 0
    SOLVED = 1
    NO_CHANGE = 2
    TIMEOUT = 3


class BaseSolver(ABC):
    default_max_time: float = 3.0

    def __init__(self, **kwargs):
        super().__init__()

    def keep_solving(
        self, prev: np.ndarray, nxt: np.ndarray, correct: np.ndarray, eps: float = 1e-3
    ) -> StoppingReason:
        pred_prev = np.round(prev).flatten()
        pred_next = np.round(nxt).flatten()
        if np.array_equal(pred_next, np.asarray(correct).flatten()):
            return StoppingReason.SOLVED
        if np.isclose(prev, nxt, atol=eps).all() and np.array_equal(pred_prev, pred_next):
            return StoppingReason.NO_CHANGE
        return StoppingReason.NOT_STOPPED

    def solve(
        self,
        clues: np.ndarray,
        solution: np.ndarray,
        max_time: float | None = None,
        sampling_ratio: float = 1.0,
    ) -> tuple[np.ndarray, int, StoppingReason]:
        if max_time is None:
            max_time = self.default_max_time
        clues_arr = np.asarray(clues)
        grid_shape = grid_shape_from_clues(clues_arr)
        clues_arr = normalise_clues(clues_arr)
        prev = blank_grid(*grid_shape)

        self._reset_state(clues_arr, prev)

        steps = []
        stopping_reason = StoppingReason.TIMEOUT
        deadline = time.monotonic() + max_time
        while time.monotonic() < deadline:
            traj = self._step(clues_arr, prev, num_steps=1, sampling_ratio=sampling_ratio)
            if len(traj) == 0:
                break
            nxt = traj[-1]
            steps.append(nxt)
            stopping = self.keep_solving(prev, nxt, solution)
            if stopping != StoppingReason.NOT_STOPPED:
                stopping_reason = stopping
                break
            prev = nxt
        return np.array(steps), len(steps), stopping_reason

    def _reset_state(self, clues: np.ndarray, prev: np.ndarray) -> None:
        del clues, prev

    @abstractmethod
    def _step(
        self, clues: np.ndarray, prev: np.ndarray, sampling_ratio: float
    ) -> np.ndarray:
        pass
