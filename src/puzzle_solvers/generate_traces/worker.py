import multiprocessing as mp
import time
import zlib
from dataclasses import dataclass

import numpy as np

from puzzle_solvers.generate_traces.record import make_run_record
from puzzle_solvers.solver import Solver


@dataclass(frozen=True, slots=True)
class _WorkerState:
    solvers: dict[str, Solver]
    anatomy: dict
    max_time: float
    seed: int | None


_state: _WorkerState | None = None


def _generate_run_seed(worker_seed: int, puzzle_idx: int, solver_idx: int) -> int:
    sequence = np.random.SeedSequence([worker_seed, puzzle_idx, solver_idx])
    return int(sequence.generate_state(1)[0])


def pool_init(solvers: dict[str, Solver], max_time: float, seed: int | None) -> None:
    global _state
    if not solvers:
        raise RuntimeError("no solvers to run")

    if seed is None:
        worker_seed = int(np.random.randint(0, 2**31))
    else:
        rank = zlib.crc32(mp.current_process().name.encode())
        worker_seed = int(np.random.SeedSequence([seed, rank]).generate_state(1)[0])

    anatomy = {name: solver.describe() for name, solver in solvers.items()}
    _state = _WorkerState(solvers, anatomy, max_time, worker_seed)


def process_puzzle(puzzle_data):
    puzzle_idx, puzzle_dict, clues, grid = puzzle_data
    state = _state

    traces, timing = {}, {}
    for solver_idx, (name, solver) in enumerate(state.solvers.items()):
        seeded = solver.with_seed(_generate_run_seed(state.seed, puzzle_idx, solver_idx))
        start = time.perf_counter()
        steps, num_steps, reason = seeded.solve(clues=clues, solution=grid, max_time=state.max_time)
        traces[name] = [make_run_record(steps, num_steps, reason, clues, grid)]
        timing[name] = time.perf_counter() - start

    meta = {**puzzle_dict.get("meta", {}), "solver_anatomy": state.anatomy}
    return {**puzzle_dict, "meta": meta, "traces": traces}, timing
