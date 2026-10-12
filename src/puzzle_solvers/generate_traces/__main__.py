from pathlib import Path

import hydra
from omegaconf import DictConfig

from puzzle_solvers.generate_traces.runner import run

CONFIG_PATH = str((Path(__file__).parent.parent / "configs").resolve())


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="generate_traces")
def main(cfg: DictConfig) -> dict:
    """Build a trace dataset from a base dataset, one column per configured solver.

    Overrides are plain hydra CLI syntax, e.g.
        python -m puzzle_solvers.generate_traces max_size=100 'solvers=[ac_parallel,gd_adam]'
    """
    return run(cfg)


if __name__ == "__main__":
    main()
