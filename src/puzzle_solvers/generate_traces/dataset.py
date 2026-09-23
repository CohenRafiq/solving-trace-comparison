import numpy as np

from neural_models.dataset import NonogramsDataset
from puzzle_solvers.generate_traces.record import make_puzzle_dict


def load_dataset(config) -> NonogramsDataset:
    return NonogramsDataset(config.input_path, max_size=config.max_size)


def iter_puzzle_args(dataset: NonogramsDataset):
    """Yield per-puzzle (puzzle_dict, processed_clues, processed_grid) for workers."""
    for idx in range(len(dataset)):
        processed_clues = dataset.clues[idx].numpy().astype(np.float32)
        processed_grid = dataset.grid[idx].numpy().astype(np.float32)
        yield (
            make_puzzle_dict(dataset, idx),
            processed_clues,
            processed_grid,
        )
