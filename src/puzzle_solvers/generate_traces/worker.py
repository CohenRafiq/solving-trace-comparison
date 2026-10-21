import time
from dataclasses import dataclass

from puzzle_solvers.generate_traces.record import make_run_record
from puzzle_solvers.solver import Solver


@dataclass(frozen=True, slots=True)
class _WorkerState:
    solvers: dict[str, Solver]
    anatomy: dict
    max_time: float


_state: _WorkerState | None = None


def pool_init(solvers: dict[str, Solver], max_time: float) -> None:
    global _state
    if not solvers:
        raise RuntimeError("no solvers to run")

    anatomy = {name: solver.describe() for name, solver in solvers.items()}
    _state = _WorkerState(solvers, anatomy, max_time)


def process_puzzle(puzzle_data):
    _, puzzle_dict, clues, grid = puzzle_data
    state = _state

    traces, timing = {}, {}
    for name, solver in state.solvers.items():
        start = time.perf_counter()
        steps, num_steps, reason = solver.solve(clues=clues, solution=grid, max_time=state.max_time)
        traces[name] = make_run_record(steps, num_steps, reason, clues, grid)
        timing[name] = time.perf_counter() - start

    meta = {**puzzle_dict.get("meta", {}), "solver_anatomy": state.anatomy}
    return {**puzzle_dict, "meta": meta, "traces": traces}, timing
