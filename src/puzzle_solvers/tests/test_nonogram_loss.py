import pytest
import torch

from trace_comparison.criterion.clue_reconstruction import ClueReconstructionLoss
from utils.grid import derive_clues_from_grid


def test_perfect_prediction_gives_zero_loss():
    grid = torch.tensor(
        [[1, 1, 0, 1], [0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]],
        dtype=torch.float32,
    )
    flat = torch.where(grid.reshape(1, -1) > 0.5, torch.tensor(6.0), torch.tensor(-6.0))
    row_clues, col_clues = derive_clues_from_grid(grid, K=2)
    clues = torch.cat([row_clues.reshape(1, -1), col_clues.reshape(1, -1)], dim=-1)

    loss = ClueReconstructionLoss(reduction="mean")
    per_sample, _, _, match = loss(flat, clues)
    assert per_sample < 1e-3
    assert match > 0.99


def test_wrong_prediction_gives_larger_loss():
    loss = ClueReconstructionLoss(reduction="mean")

    ones = torch.full((1, 16), 6.0)
    zeros = torch.full((1, 16), -6.0)
    row, col = derive_clues_from_grid(torch.ones(4, 4), K=2)
    clues = torch.cat([row.reshape(1, -1), col.reshape(1, -1)], dim=-1)

    loss_ones = loss(ones, clues)[0]
    loss_zeros = loss(zeros, clues)[0]
    assert loss_zeros > loss_ones


def test_reduction_modes():
    grid = torch.full((2, 16), 6.0)
    row, col = derive_clues_from_grid(torch.ones(4, 4), K=2)
    clues = torch.cat([row.reshape(1, -1), col.reshape(1, -1)], dim=-1).repeat(2, 1)

    mean = ClueReconstructionLoss(reduction="mean")
    none = ClueReconstructionLoss(reduction="none")
    sum_loss = ClueReconstructionLoss(reduction="sum")

    mean_out = mean(grid, clues)
    none_out = none(grid, clues)
    sum_out = sum_loss(grid, clues)

    assert mean_out[0].shape == ()
    assert none_out[0].shape == (2,)
    assert torch.isclose(sum_out[0], none_out[0].sum())


def test_invalid_reduction_raises():
    with pytest.raises(ValueError):
        ClueReconstructionLoss(reduction="bogus")
