import time
from dataclasses import dataclass

import numpy as np
import torch

from puzzle_solvers.generate_traces.config import TracesConfig
from puzzle_solvers.generate_traces.record import make_run_record
from old.solvers import build_solver


@dataclass(frozen=True, slots=True)
class _WorkerState:
    config: TracesConfig
    solvers: dict
    anatomy: dict
    seed: int


_state: _WorkerState | None = None


def build_solver_from_config(name: str, config: TracesConfig):
    if config.clues_shape is None:
        raise RuntimeError("config.clues_shape not set; load the dataset first")
    return build_solver(name, clues_shape=config.clues_shape, model_path=config.model_path)


def _run_seed(worker_seed: int, puzzle_idx: int, solver_idx: int, sample_idx: int) -> int:
    sequence = np.random.SeedSequence([worker_seed, puzzle_idx, solver_idx, sample_idx])
    return int(sequence.generate_state(1)[0])


def pool_init(config: TracesConfig) -> None:
    global _state
    if config.solvers is None:
        raise RuntimeError("config.solvers not set")

    seed = np.random.randint(0, 2**31)
    np.random.seed(seed)
    torch.manual_seed(seed)

    solvers = {name: build_solver_from_config(name, config) for name in config.solvers}
    anatomy = {name: solver.describe() for name, solver in solvers.items()}
    _state = _WorkerState(config, solvers, anatomy, seed)


def _run_solver(solver, solver_idx: int, puzzle_idx: int, clues, grid) -> list[dict]:
    config, seed = _state.config, _state.seed
    steps, num_steps, reason = solver.solve(
        clues=clues,
        solution=grid,
        max_time=config.max_time,
        seed=_run_seed(seed, puzzle_idx, solver_idx, 0),
    )
    return [make_run_record(steps, num_steps, reason, clues, grid)]


def process_puzzle(puzzle_data):
    puzzle_idx, puzzle_dict, clues, grid = puzzle_data

    traces, timing = {}, {}
    for solver_idx, (name, solver) in enumerate(_state.solvers.items()):
        start = time.perf_counter()
        traces[name] = _run_solver(solver, solver_idx, puzzle_idx, clues, grid)
        timing[name] = time.perf_counter() - start

    meta = {**puzzle_dict.get("meta", {}), "solver_anatomy": _state.anatomy}
    return {**puzzle_dict, "meta": meta, "traces": traces}, timing
