from collections.abc import Iterable
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from .blocks.base import Block, DataKind, check_kind
from .data.dataset import TraceDataset, load_trace_dataset


def run_experiment(dataset: TraceDataset, blocks: Iterable[Block]) -> np.ndarray:
    """Pipe the jagged `(solvers, puzzles)` traces through blocks into an `(S, S)` matrix.

    Each block validates its own input and output (see `Block.__call__`).
    """
    data = dataset.traces
    for i, block in enumerate(blocks, start=1):
        try:
            data = block(data)
        except Exception as e:
            e.add_note(f"raised in pipeline block {i}")
            raise
    check_kind(data, DataKind.MATRIX, "pipeline output")
    return data


@hydra.main(version_base=None, config_path="config", config_name="sim_matrix")
def main(cfg: DictConfig) -> None:
    """Build a solver-vs-solver similarity matrix from a traces jsonl file.

    Overrides are plain hydra CLI syntax, e.g.
        python -m trace_comparison.gen_sim_matrix input=bs_test.jsonl max_size=100
    """
    data_dir = Path(cfg.data_dir)
    dataset = load_trace_dataset(
        data_dir / "traces_dataset" / "nonograms" / cfg.input,
        solvers=cfg.solvers,
        max_size=cfg.max_size,
        puzzle_filter=cfg.puzzle_filter,
    )
    matrix = run_experiment(dataset, instantiate(cfg.blocks))
    df = pd.DataFrame(matrix, index=dataset.solvers, columns=dataset.solvers, dtype=float)

    out = data_dir / "similarity_matrix" / "nonograms" / cfg.output_file
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"Saved similarity matrix to {out} ({dataset.traces.shape[1]} puzzles)\n{df}")


if __name__ == "__main__":
    main()