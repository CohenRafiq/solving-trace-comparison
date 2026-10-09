from dataclasses import dataclass
import time 
from typing import Literal
import numpy as np

from puzzle_solvers.types import (
    StoppingReason, Strategy, State, Aggregator
)


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

    def _early_stop(self, curr_pred: np.ndarray, prev_pred: np.ndarray, solution: np.ndarray) -> StoppingReason:
        prev_r = np.round(prev_pred).flatten()
        curr_r = np.round(curr_pred).flatten()
        match self.stopping_condition:
            case "rounded":
                if np.array_equal(curr_r, solution.flatten()):
                    return StoppingReason.SOLVED
            case "close":
                if np.allclose(curr_pred, solution, atol=self.solved_threshold):
                    return StoppingReason.SOLVED
            case _:
                raise ValueError(f"Unknown stopping_condition: {self.stopping_condition}")

        if (np.isclose(prev_pred, curr_pred, atol=self.stopping_threshold).all()
             and np.array_equal(prev_r, curr_r)):
            return StoppingReason.NO_CHANGE

        return StoppingReason.NOT_STOPPED

    def solve(self, clues: np.ndarray, solution: np.ndarray, max_time: float) -> tuple[np.ndarray, int, StoppingReason]:
        state = self.state.initialise(clues, solution.shape, seed=self.seed)
        solving_traces = []
        finish_time = time.monotonic() + max_time
        while time.monotonic() < finish_time:
            state = self.strategy(state)
            curr_pred = self.aggregator(self.state.grids, clues)
            solving_traces.append(curr_pred)
            if len(solving_traces) <= 1:
                continue
            prev_pred = solving_traces[-2]
            stopping_reason = self._early_stop(curr_pred, prev_pred, solution)
            if stopping_reason != StoppingReason.NOT_STOPPED:
                return np.array(solving_traces), len(solving_traces), stopping_reason
        return np.array(solving_traces), len(solving_traces), StoppingReason.TIMEOUT

    def describe(self) -> dict:
        return {
            "name": self.name,
            "strategy": self.strategy.__class__.__name__,
            "state": self.state.__class__.__name__,
            "aggregator": self.aggregator.__class__.__name__,
            "seed": self.seed,
            "stopping_condition": self.stopping_condition,
            "solved_threshold": self.solved_threshold,
            "stopping_threshold": self.stopping_threshold,
        }
    
# def _load_class_with_kwargs(registry: dict, config: dict, key: str):
#     cls_name = config[key]["name"]
#     cls_kwargs = {k: v for k, v in config[key].items() if k != "name"}
#     cls = registry[cls_name]
#     return cls(**cls_kwargs)

# def build_solver(name: str, config: dict) -> Solver:
#     strategy = _load_class_with_kwargs(STRATEGIES, config, "strategy")
#     state = _load_class_with_kwargs(STATES, config, "state")
#     renderer = _load_class_with_kwargs(RENDERERS, config, "renderer")
#     aggregation = _load_class_with_kwargs(AGGREGATORS, config, "aggregation")

#     return Solver(
#         name=name,
#         strategy=strategy,
#         state=state,
#         renderer=renderer,
#         aggregation=aggregation,
#         seed=config.get("seed"),
#         stopping_condition=config.get("stopping_condition", "close"),
#         solved_threshold=config.get("solved_threshold", 0.2),
#         stopping_threshold=config.get("stopping_threshold", 1e-3),
#     )