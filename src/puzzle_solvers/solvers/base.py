import time
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from utils.grid import normalise_clues


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
        self, next: np.ndarray, prev_steps: list[np.ndarray], correct: np.ndarray, eps: float = 5e-3
    ) -> StoppingReason:
        pred_next = next.copy()
        pred_next[next == 0.5] = -1
        pred_next = np.round(pred_next).flatten()
        if np.array_equal(pred_next, np.asarray(correct).flatten()):
            return StoppingReason.SOLVED
        for prev in prev_steps:
            if np.isclose(prev, next, atol=eps).all() and np.array_equal(np.round(prev).flatten(), np.round(next).flatten()):
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
        clues_arr = normalise_clues(clues_arr)

        steps = []
        stopping_reason = StoppingReason.TIMEOUT
        deadline = time.monotonic() + max_time
        prev = self._first_step(clues_arr, solution.shape, sampling_ratio=sampling_ratio)
        while time.monotonic() < deadline:
            next = self._step(clues_arr, prev, sampling_ratio=sampling_ratio)
            stopping = self.keep_solving(next, steps, solution)
            steps.append(next)
            if stopping != StoppingReason.NOT_STOPPED:
                stopping_reason = stopping
                break
            prev = next
        return np.array(steps), len(steps), stopping_reason


    @abstractmethod
    def _step(
        self, clues: np.ndarray, prev: np.ndarray, sampling_ratio: float = 1.0
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _first_step(
        self, clues: np.ndarray, grid_shape: tuple[int, int], sampling_ratio: float = 1.0
    ) -> np.ndarray:
        pass
