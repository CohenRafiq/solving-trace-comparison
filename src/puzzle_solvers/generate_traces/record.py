import numpy as np
import torch

from neural_models.dataset import NonogramsDataset
from puzzle_solvers.solvers.base import StoppingReason
from trace_comparison.criterion.clue_reconstruction2 import ClueReconstructionLoss

_CR_CRITERION = ClueReconstructionLoss(
    reduction="none",
    aux_weight_start=0.0,
    aux_weight_end=0.0,
    entropy_weight_start=0.0,
    entropy_weight_end=0.0,
).eval()


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
            "mean_clue_runs": float(
                np.mean([len(line) for group in raw_clues for line in group])
            ),
        },
    }


def steps_to_solve(steps: np.ndarray, solution: np.ndarray) -> int:
    sol_flat = np.asarray(solution).flatten()
    for i, step in enumerate(steps):
        if np.array_equal(np.round(step).flatten(), sol_flat):
            return i
    return len(steps)


def compute_run_losses(
    steps: np.ndarray, clues: np.ndarray, solution: np.ndarray
) -> tuple[list, list]:
    n = len(steps)
    if n == 0:
        return [], []
    steps_t = torch.from_numpy(np.asarray(steps)).float().reshape(n, -1)
    clues_t = torch.from_numpy(np.asarray(clues)).float().reshape(-1).unsqueeze(0).expand(n, -1)
    # The criterion expects logits (it sigmoids internally). Recorded steps are
    # probabilities, so invert before evaluating to keep cr_loss ~ 0 at a solved grid.
    steps_logit = torch.logit(steps_t.clamp(1e-6, 1 - 1e-6), eps=1e-6)
    with torch.no_grad():
        per_sample, *_ = _CR_CRITERION(steps_logit, clues_t)
    cr_losses = per_sample.tolist()

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
        "steps_to_solve": steps_to_solve(steps, solution),
        "cr_losses": cr_losses,
        "mse_losses": mse_losses,
        "num_steps": num_steps,
        "solved": reason == StoppingReason.SOLVED,
        "stopping_reason": reason.name,
    }
