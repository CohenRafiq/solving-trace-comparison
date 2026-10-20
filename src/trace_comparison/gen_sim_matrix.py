from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from puzzle_solvers.generate_traces.config import resolve

CONFIG_PATH = str((Path(__file__).parent.parent / "configs").resolve())

def run_experiment(dataset_columns, blocks):
    if len(dataset_columns.shape) != 4:
        raise ValueError(f"Dataset columns must be 4D (solvers, steps, width, height), got {len(dataset_columns.shape)}D")
    current_data = dataset_columns
    for i, block in enumerate(blocks):
        try:
            current_data = block(current_data)
        except Exception as e:
            raise ValueError(f"Shape validation failed at block {i + 1} ('{block.name}'): {str(e)}") from e
    expected_shape = (len(dataset_columns), len(dataset_columns))
    assert current_data.shape == expected_shape, f"Expected shape {expected_shape}, got {current_data.shape}"
    return current_data

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="generate_traces")
def main(cfg: DictConfig) -> dict:
    """Build a trace dataset from a base dataset, one column per configured solver.

    Overrides are plain hydra CLI syntax, e.g.
        python -m puzzle_solvers.generate_traces max_size=100 'solvers=[ac_parallel,gd_adam]'
    """
    cfg = resolve(cfg)

    dataset = instantiate(cfg.dataset, file_name=cfg.input, max_size=cfg.max_size)
    # (solvers, n_puzzles, steps_jagged, width, height)
    # jagged because solvers may take different number of steps
    blocks = instantiate(cfg.blocks)
    similarity_matrix = run_experiment(dataset.get_columns(cfg.solvers), blocks)
    df = pd.DataFrame(
        similarity_matrix,
        index=cfg.solvers,
        columns=cfg.solvers,
        dtype=float,
    )
    output_path = Path(cfg.data_dir) / "similarity_matrix" / "nonograms" /cfg.output_file
    df.to_csv(output_path, index=True)
    print(f"Saved similarity matrix to {output_path}")
    print(df)


if __name__ == "__main__":
    main()
