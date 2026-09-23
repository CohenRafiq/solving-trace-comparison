import time

import numpy as np
import torch

from puzzle_solvers.generate_traces.config import TracesConfig
from puzzle_solvers.generate_traces.record import make_run_record
from puzzle_solvers.solvers import SOLVERS
from puzzle_solvers.solvers.model_solver import ModelSolver

_worker_solvers: dict = {}
_worker_config: TracesConfig | None = None


def _build_solver(name: str, config: TracesConfig):
    if config.clues_shape is None:
        raise RuntimeError("config.clues_shape not set; load the dataset first")
    if name != "model_solver":
        return SOLVERS[name]()
    h, k = config.clues_shape[1], config.clues_shape[2]
    return ModelSolver(model_path=config.model_path, input_shape=(2, h, k))


def pool_init(config: TracesConfig) -> None:
    """Create solver instances once per worker process."""
    global _worker_solvers, _worker_config
    _worker_config = config
    if config.solvers is None:
        raise RuntimeError("config.solvers not set")
    seed = np.random.randint(0, 2**31)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    _worker_solvers = {name: _build_solver(name, config) for name in config.solvers}


def process_puzzle(puzzle_data):
    """Run all solvers on a single puzzle; returns (result_dict, timing_dict)."""
    puzzle_dict, processed_clues, processed_grid = puzzle_data
    config = _worker_config

    result_dict = dict(puzzle_dict)
    result_dict["traces"] = {}
    timing = {}

    for solver_name in config.solvers:
        solver = _worker_solvers[solver_name]
        solver_dict = {}
        t0 = time.time()

        for sampling_ratio, number_of_samples in config.sampling_ratio_runs:
            runs = []
            for _ in range(number_of_samples):
                steps, num_steps, reason = solver.solve(
                    clues=processed_clues,
                    solution=processed_grid,
                    max_time=config.max_time,
                    sampling_ratio=sampling_ratio,
                )
                runs.append(
                    make_run_record(steps, num_steps, reason, processed_clues, processed_grid)
                )
            solver_dict[sampling_ratio] = runs
        result_dict["traces"][solver_name] = solver_dict
        timing[solver_name] = time.time() - t0

    return result_dict, timing
