from collections.abc import Iterable
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import time

from .blocks import BLOCKS
from .blocks.base import Block, DataKind
from .data.dataset import load_trace_dataset

def build_blocks(cfg: DictConfig, grid_shape: tuple[int, int]) -> list[Block]:
    """Instantiate blocks from config."""
    blocks = []
    for block_cfg in cfg.blocks:
        block_type = block_cfg.name  # Access as attribute
        if block_type not in BLOCKS:
            raise ValueError(f"Unknown block type {block_type}")
        block_class = BLOCKS[block_type]
        # Create dict of kwargs excluding 'name', without mutating original
        kwargs = {k: v for k, v in block_cfg.items() if k != "name"}
        blocks.append(block_class(**kwargs))
    
    setup_ctx = {"grid_shape": grid_shape}
    for b in blocks:
        setup_ctx.update(b.get_context())
    for b in blocks:
        b.set_context(setup_ctx)
    return blocks


def run_experiment(data: np.ndarray, blocks: Iterable[Block]) -> np.ndarray:
    """Pipe the jagged `(solvers, puzzles)` traces through blocks into an `(S, S)` matrix.

    Each block validates its own input and output (see `Block.__call__`).
    """
    for i, block in enumerate(blocks, start=1):
        start = time.perf_counter()
        try:
            data = block(data)
        except Exception as e:
            e.add_note(f"raised in pipeline block {i}")
            raise
        print(f"Block {block.name}: {time.perf_counter() - start:.3}")
    DataKind.MATRIX.check(data, "pipeline output")
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
    blocks = build_blocks(cfg, dataset.grid_shape)
    data = dataset.clues if cfg.get("clue_input", False) else dataset.traces
    matrix = run_experiment(data, blocks)
    df = pd.DataFrame(matrix, index=dataset.solvers, columns=dataset.solvers, dtype=float)

    out = data_dir / "similarity_matrix" / "nonograms" / cfg.output_file
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"Saved similarity matrix to {out} ({dataset.traces.shape[1]} puzzles)\n{df}")


if __name__ == "__main__":
    main()
