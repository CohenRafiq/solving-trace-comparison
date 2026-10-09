from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
import numpy as np
from typing import Literal, Self

import torch

from utils.grid import unpad_clue

class StoppingReason(Enum):
    NOT_STOPPED = 0
    SOLVED = 1
    NO_CHANGE = 2
    TIMEOUT = 3

class Strategy(ABC):
    """Takes one move on the whole population state."""
    
    @abstractmethod
    def __call__(self, state: State) -> State:
        """Take one move in the solving process."""
        ...

@lru_cache(maxsize=10_000)
def unpad_single_clue(clue: tuple[float, ...]) -> tuple[int, ...]:
    return unpad_clue(list(clue))

@dataclass(frozen=True)
class State(ABC):
    """Everything the solver needs to store between moves"""
    pop_of_grids: np.ndarray
    k: int = 1
    seed: int | None = None
    step: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(
            self.seed + self.step if 
            self.seed is not None else None
            )

    def _axis_to_tuples(self, clues: np.ndarray, axis: int) -> tuple[tuple[int, ...], ...]:
        length = clues.shape[axis]
        return tuple(
            unpad_single_clue(tuple(clues[axis, i]))
            for i in range(length)
        )

    def _clues_to_tuples(self, clues: np.ndarray) -> tuple[tuple[int, ...], ...]:
        rows = self._axis_to_tuples(clues, axis=0)
        cols = self._axis_to_tuples(clues, axis=1)
        return (rows, cols)

    @abstractmethod
    def _init_params(self, clues: np.ndarray, grid_shape: tuple[int, int]) -> dict:
        """Initialise the parameters of the state."""
        ...

    def initialise(self, clues: np.ndarray, grid_shape: tuple[int, int]) -> Self:
        """Build the starting state. Its grid is not recorded as a step."""
        params = self._init_params(clues, grid_shape)
        return replace(self, step=0, **params)

    @property
    def grids(self) -> np.ndarray:
        if isinstance(self.pop_of_grids, np.ndarray):
            return self.pop_of_grids.astype(np.float64)
        if isinstance(self.pop_of_grids, torch.Tensor):
            return self.pop_of_grids.detach().cpu().numpy().astype(np.float64)
        raise TypeError(f"Unsupported type for pop_of_grids: {type(self.pop_of_grids)}")

@dataclass(frozen=True)
class Aggregator(ABC):
    """(k, H, W) predictions -> (H, W)."""

    # replace: swap -1 with 0.5 before averaging, 
    # skip: ignore -1 values when averaging
    unknown_handling: Literal["replace", "skip"] = "replace"

    @abstractmethod
    def __call__(self, predictions: np.ndarray, clues: np.ndarray) -> np.ndarray:
        ...