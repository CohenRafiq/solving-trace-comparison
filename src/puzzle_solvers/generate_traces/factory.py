from omegaconf import DictConfig

from puzzle_solvers.solver import Solver, build_solver

SOLVER_AXES = ("strategy", "state", "aggregator")


def build_solvers(cfg: DictConfig) -> dict[str, Solver]:
    """Instantiate every selected solver in config"""
    group = cfg.solvers
    if isinstance(group, list):
        group = {entry["name"]: entry for entry in group}

    solvers: dict[str, Solver] = {}
    for name, entry in group.items():
        solver = build_solver(entry)
        if solver.name != name:
            raise KeyError(
                f"solver config {name!r} declares name {solver.name!r}; "
                "the file name and the `name` key must agree"
            )
        solvers[name] = solver
    return solvers
