import time
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from puzzle_solvers.types import Aggregator, State, StoppingReason, Strategy


@dataclass(frozen=True)
class Solver:
    """A solver is a combination of a strategy, state, and aggregator."""

    name: str
    strategy: Strategy
    state: State
    aggregator: Aggregator
    seed: int | None = None
    stopping_condition: Literal["rounded", "close"] = "close"
    solved_threshold: float = 0.2
    stopping_threshold: float = 1e-3

    def _is_solved(self, curr_pred: np.ndarray, solution: np.ndarray) -> bool:
        match self.stopping_condition:
            case "rounded":
                return bool(np.array_equal(np.round(curr_pred).flatten(), solution.flatten()))
            case "close":
                return bool(np.allclose(curr_pred, solution, atol=self.solved_threshold))
            case _:
                raise ValueError(f"Unknown stopping_condition: {self.stopping_condition}")

    def _has_stalled(self, prev_raw: np.ndarray, curr_raw: np.ndarray) -> bool:
        return bool(np.isclose(prev_raw, curr_raw, atol=self.stopping_threshold).all())

    def solve(
        self, clues: np.ndarray, solution: np.ndarray, max_time: float
    ) -> tuple[np.ndarray, int, StoppingReason]:
        """Solve the puzzle with the given clues and solution within the time limit."""
        flat_clues = np.asarray(clues, dtype=np.float32).reshape(1, -1)
        state = self.state.initialise(clues, solution.shape)
        solving_traces: list[np.ndarray] = []
        prev_raw = None
        finish_time = time.monotonic() + max_time
        while time.monotonic() < finish_time:
            state = self.strategy(state)
            curr_pred = self.aggregator(state.grids, flat_clues)
            curr_raw = state.raw_grids
            solving_traces.append(curr_pred)
            if len(solving_traces) > 1:
                if self._is_solved(curr_pred, solution):
                    return np.array(solving_traces), len(solving_traces), StoppingReason.SOLVED
                if prev_raw is not None and self._has_stalled(prev_raw, curr_raw):
                    return np.array(solving_traces), len(solving_traces), StoppingReason.NO_CHANGE
            prev_raw = curr_raw
        return np.array(solving_traces), len(solving_traces), StoppingReason.TIMEOUT

    def with_seed(self, seed: int) -> "Solver":
        """A copy whose state draws from `seed` instead of the configured one."""
        return replace(self, state=replace(self.state, seed=seed))

    def describe(self) -> dict:
        return {
            "name": self.name,
            "strategy": type(self.strategy).__name__,
            "state": type(self.state).__name__,
            "aggregator": type(self.aggregator).__name__,
            "seed": self.seed,
            "stopping_condition": self.stopping_condition,
            "solved_threshold": self.solved_threshold,
            "stopping_threshold": self.stopping_threshold,
        }


SOLVER_KEYS = ("name", "strategy", "state", "aggregator")


def build_solver(cfg: DictConfig) -> Solver:
    missing = [k for k in SOLVER_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"solver config is missing {missing}; got keys {list(cfg)}")

    return Solver(
        name=cfg.name,
        strategy=instantiate(cfg.strategy),
        state=instantiate(cfg.state),
        aggregator=instantiate(cfg.aggregator),
        seed=cfg.get("seed"),
        stopping_condition=cfg.get("stopping_condition", "close"),
        solved_threshold=cfg.get("solved_threshold", 0.2),
        stopping_threshold=cfg.get("stopping_threshold", 1e-3),
    )
