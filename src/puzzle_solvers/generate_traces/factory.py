from dataclasses import replace

from omegaconf import DictConfig

from puzzle_solvers.solver import Solver, build_solver

SOLVER_AXES = ("strategy", "state", "aggregator")


def build_solvers(cfg: DictConfig) -> dict[str, Solver]:
    """Instantiate every selected solver config.

    Each config entry declares `seeds`; it is expanded into one run per seed,
    named `{base}_{seed}` (e.g. ac_sgl with seeds [1, 2, 3] yields ac_sgl_1,
    ac_sgl_2, ac_sgl_3). The run seed is fixed per name and applies to every
    puzzle.
    """
    group = cfg.solvers
    if isinstance(group, list):
        group = {entry["name"]: entry for entry in group}

    solvers: dict[str, Solver] = {}
    for name, entry in group.items():
        base = build_solver(entry)
        if base.name != name:
            raise KeyError(
                f"solver config {name!r} declares name {base.name!r}; "
                "the file name and the `name` key must agree"
            )

        seeds = entry.get("seeds")
        if not seeds:
            raise KeyError(
                f"solver config {name!r} must declare a non-empty `seeds` list; "
                f"got keys {list(entry)}"
            )
        for seed in seeds:
            run = replace(base, name=f"{name}_{seed}")
            run = run.with_seed(int(seed))
            solvers[run.name] = run
    return solvers
