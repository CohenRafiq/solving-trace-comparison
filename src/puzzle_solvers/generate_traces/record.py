import numpy as np
import torch

from neural_models.dataset import NonogramsDataset
from old.solvers.base import StoppingReason
from old.solvers.criterion import grid_losses


def to_serialisable(x):
    if torch.is_tensor(x):
        return x.tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (tuple, list)):
        return [to_serialisable(v) for v in x]
    if isinstance(x, dict):
        return {k: to_serialisable(v) for k, v in x.items()}
    return x


def make_puzzle_dict(dataset: NonogramsDataset, idx: int) -> dict:
    raw_clues = dataset.raw_clues[idx]
    raw_grid = dataset.raw_grids[idx]
    grid = dataset.grid[idx].numpy()
    return {
        "clues": raw_clues,
        "grid": torch.tensor(raw_grid).to(dtype=torch.float32).tolist(),
        "meta": {
            "shape": list(grid.shape),
            "density": float(grid.mean()),
            "mean_clue_runs": float(np.mean([len(line) for group in raw_clues for line in group])),
        },
    }


def compute_run_losses(
    steps: np.ndarray, clues: np.ndarray, solution: np.ndarray
) -> tuple[list, list]:
    """A computes mse loss and clue reconstruction loss for each step in the run."""
    n = len(steps)
    if n == 0:
        return [], []
    cr_losses = grid_losses(np.asarray(steps).reshape(n, -1), clues).tolist()
    sol_flat = np.asarray(solution).reshape(1, -1)
    mse_losses = np.mean((np.asarray(steps).reshape(n, -1) - sol_flat) ** 2, axis=1).tolist()
    return cr_losses, mse_losses


def make_run_record(
    steps: np.ndarray,
    num_steps: int,
    reason: StoppingReason,
    clues: np.ndarray,
    solution: np.ndarray,
) -> dict:
    cr_losses, mse_losses = compute_run_losses(steps, clues, solution)
    return {
        "steps": [step for step in steps],
        "cr_losses": cr_losses,
        "mse_losses": mse_losses,
        "num_steps": num_steps,
        "solved": reason == StoppingReason.SOLVED,
        "stopping_reason": reason.name,
    }
