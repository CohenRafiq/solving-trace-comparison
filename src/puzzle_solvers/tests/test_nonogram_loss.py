import pytest
import torch

from puzzle_solvers.losses.clue_reconstruction_loss import ClueReconstructionLoss
from utils.grid import derive_clues_from_grid


def _flat_clues(grid: torch.Tensor, k: int) -> torch.Tensor:
    row_clues, col_clues = derive_clues_from_grid(grid, K=k)
    return torch.cat([row_clues.reshape(1, -1), col_clues.reshape(1, -1)], dim=-1)


def test_perfect_prediction_gives_low_loss():
    grid = torch.tensor(
        [[1, 1, 0, 1], [0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]],
        dtype=torch.float32,
    )
    flat = torch.where(grid.reshape(1, -1) > 0.5, torch.tensor(6.0), torch.tensor(-6.0))
    clues = _flat_clues(grid, k=2)

    loss, metrics = ClueReconstructionLoss()(flat, clues, return_metrics=True)

    assert loss < 1e-2
    assert metrics["clue_match_pct"] > 0.99


def test_wrong_prediction_gives_larger_loss():
    loss = ClueReconstructionLoss()

    ones = torch.full((1, 16), 6.0)
    zeros = torch.full((1, 16), -6.0)
    clues = _flat_clues(torch.ones(4, 4), k=2)

    loss_ones = loss(ones, clues)
    loss_zeros = loss(zeros, clues)
    assert loss_zeros > loss_ones


def test_batch_reduction_modes():
    grid = torch.tensor(
        [[1, 1, 0, 1], [0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]],
        dtype=torch.float32,
    )
    flat = torch.where(grid.reshape(1, -1) > 0.5, torch.tensor(6.0), torch.tensor(-6.0))
    clues = _flat_clues(grid, k=2)

    batch = flat.repeat(2, 1)
    batch_clues = clues.repeat(2, 1)
    sum_loss = ClueReconstructionLoss(batch_reduction="sum")(batch, batch_clues)
    mean_loss = ClueReconstructionLoss(batch_reduction="mean")(batch, batch_clues)

    assert torch.isclose(sum_loss, 2 * mean_loss)


def test_invalid_reduction_raises():
    with pytest.raises(ValueError):
        ClueReconstructionLoss(batch_reduction="bogus")
