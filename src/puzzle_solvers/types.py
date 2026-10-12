from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Literal, Self

import numpy as np
import torch


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

@dataclass(frozen=True)
class State(ABC):
    """Everything the solver needs to store between moves"""

    pop_of_grids: np.ndarray
    k: int = 1
    seed: int | None = None
    step: int = 0
    rng: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rng",
            np.random.default_rng(
                self.seed + self.step if self.seed is not None else None
            ),
        )

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
        """The readout: what a trace records, as an (H, W) grid per member."""
        return self.pop_of_grids.astype(np.float64)

    @property
    def raw_grids(self) -> np.ndarray:
        pop = self.pop_of_grids
        if isinstance(pop, np.ndarray):
            return pop
        if isinstance(pop, torch.Tensor):
            return pop.detach().cpu().numpy()
        raise TypeError(f"Unsupported type for pop_of_grids: {type(pop)}")

@dataclass(frozen=True)
class Aggregator(ABC):
    """(k, H, W) predictions -> (H, W)."""

    # replace: swap -1 with 0.5 before averaging
    # skip: ignore -1 values when averaging
    unknown_handling: Literal["replace", "skip"] = "replace"

    @abstractmethod
    def __call__(self, predictions: np.ndarray, clues: np.ndarray) -> np.ndarray:
        ...
