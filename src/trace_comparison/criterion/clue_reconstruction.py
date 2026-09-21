import math

import torch
import torch.nn.functional as F

from neural_models_old.criterion.base import BaseCriterion


class ClueReconstructionLoss(BaseCriterion):
    clue_indices_template: torch.Tensor

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        self.register_buffer("clue_indices_template", torch.tensor([], dtype=torch.float))

    def _grid_to_row_clues(self, grid: torch.Tensor, K: int) -> torch.Tensor:
        shifted = F.pad(grid[..., :-1], (1, 0), value=0.0)
        left_run_ends = grid * (1.0 - shifted)
        cum = torch.cumsum(left_run_ends, dim=-1)

        if (
            self.clue_indices_template.numel() != K
            or self.clue_indices_template.device != grid.device
        ):
            clue_idx = torch.arange(1, K + 1, dtype=grid.dtype, device=grid.device)
            self.clue_indices_template = clue_idx

        weights = torch.relu(1.0 - torch.abs(cum.unsqueeze(-1) - self.clue_indices_template))
        row_lengths = (grid.unsqueeze(-1) * weights).sum(dim=-2)
        return row_lengths

    def _clue_match_percentage(
        self,
        pred_row: torch.Tensor,
        pred_col: torch.Tensor,
        row_clues: torch.Tensor,
        col_clues: torch.Tensor,
    ) -> torch.Tensor:
        row_match = (pred_row.round() == row_clues).all(dim=2).float()
        col_match = (pred_col.round() == col_clues).all(dim=2).float()
        return (row_match.mean(dim=1) + col_match.mean(dim=1)) / 2.0

    def forward(self, grid: torch.Tensor, clues: torch.Tensor):
        grid = torch.sigmoid(grid)
        B, S = grid.shape
        side = int(math.sqrt(S))
        assert side * side == S, "Grid must contain a perfect square number of cells"
        grid = grid.view(B, side, side)

        N = side
        K = clues.shape[-1] // (2 * N)
        row_clues = clues[:, : N * K].view(B, N, K)
        col_clues = clues[:, N * K :].view(B, N, K)
        row_clues = row_clues.to(grid.device, non_blocking=True)
        col_clues = col_clues.to(grid.device, non_blocking=True)

        max_row_runs = row_clues.shape[-1]
        max_col_runs = col_clues.shape[-1]

        pred_row = self._grid_to_row_clues(grid, max_row_runs)
        pred_col = self._grid_to_row_clues(grid.transpose(1, 2), max_col_runs)

        row_err = (pred_row - row_clues.float()) ** 2
        col_err = (pred_col - col_clues.float()) ** 2
        per_sample = row_err.mean(dim=[1, 2]) + col_err.mean(dim=[1, 2])
        clue_match_pct = self._clue_match_percentage(pred_row, pred_col, row_clues, col_clues)

        if self.reduction == "none":
            return per_sample, row_err.mean(dim=[1, 2]), col_err.mean(dim=[1, 2]), clue_match_pct
        elif self.reduction == "mean":
            return per_sample.mean(), row_err.mean(), col_err.mean(), clue_match_pct.mean()
        else:
            return per_sample.sum(), row_err.sum(), col_err.sum(), clue_match_pct.mean()
