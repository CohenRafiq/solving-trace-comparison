import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _runs(grid: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable run lengths (B, N, K) and cumulative run starts of each row."""
    starts = grid * (1.0 - F.pad(grid[..., :-1], (1, 0)))
    cum = starts.cumsum(-1)
    idx = torch.arange(1, K + 1, dtype=grid.dtype, device=grid.device)
    weights = torch.relu(1.0 - (cum.unsqueeze(-1) - idx).abs())
    return (grid.unsqueeze(-1) * weights).sum(-2), cum


def _valid(clues: torch.Tensor) -> torch.Tensor:
    """Mask of real clue slots plus the first padding slot (so 'no more runs' is penalised)."""
    return ((clues > 0) | ((clues == 0).cumsum(-1) == 1)).float()


@dataclass(eq=False)
class ClueReconstructionLoss(nn.Module):
    """Reconstructs nonogram clues from a soft grid using standard sigmoid."""

    aux_weight: float = 0.3
    binarize_weight: float = 1.0
    batch_reduction: str = "sum"  # "sum" (batch-invariant) or "mean"

    def __post_init__(self):
        super().__init__()
        if self.batch_reduction not in ("sum", "mean"):
            raise ValueError("batch_reduction must be 'sum' or 'mean'")

    def _batch(self, per_puzzle: torch.Tensor) -> torch.Tensor:
        return per_puzzle.sum() if self.batch_reduction == "sum" else per_puzzle.mean()

    def _compute(self, probs, row_clues, col_clues, binarize_weight):
        B, N = probs.shape[0], probs.shape[1]
        K = row_clues.shape[-1]
        probs_t = probs.transpose(1, 2)
        pred_row, cum_row = _runs(probs, K)
        pred_col, cum_col = _runs(probs_t, K)

        row_valid, col_valid = _valid(row_clues), _valid(col_clues)
        row_err = (pred_row - row_clues) ** 2 * row_valid
        col_err = (pred_col - col_clues) ** 2 * col_valid

        # Per-puzzle normalisation (B,)
        n_valid = row_valid.sum((1, 2)) + col_valid.sum((1, 2)) + 1e-8
        main_pp = (row_err.sum((1, 2)) + col_err.sum((1, 2))) / n_valid

        aux_pp = 0.0
        for fill, clues, cum in (
            (probs.sum(-1), row_clues, cum_row),
            (probs_t.sum(-1), col_clues, cum_col),
        ):
            aux_pp = aux_pp + (
                (((fill - clues.sum(-1)) / N) ** 2).mean(-1)
                + (((cum[..., -1] - (clues > 0).sum(-1)) / K) ** 2).mean(-1)
            )

        binarize_pp = (probs * (1.0 - probs)).mean((1, 2))

        main, aux, binarize = self._batch(main_pp), self._batch(aux_pp), self._batch(binarize_pp)

        return {
            "loss": main + self.aux_weight * aux + binarize_weight * binarize,
            "main_term": main,
            "aux_term": aux,
            "binarize_term": binarize,
            "row_err": row_err.sum() / (row_valid.sum() + 1e-8),
            "col_err": col_err.sum() / (col_valid.sum() + 1e-8),
            "clue_match_pct": 0.5 * (
                (pred_row.round() == row_clues).all(-1).float().mean()
                + (pred_col.round() == col_clues).all(-1).float().mean()
            ),
            "mean_saturation": (probs - 0.5).abs().mean(),
        }

    def forward(self, grid, clues, return_metrics=False):
        """grid: (B, N*N) logits; clues: (B, 2*N*K) flattened rows then cols."""
        B, S = grid.shape
        N = math.isqrt(S)
        if N * N != S:
            raise ValueError(f"Grid size {S} is not a perfect square")
        if clues.shape[-1] % (2 * N):
            raise ValueError(f"clues last dim {clues.shape[-1]} not divisible by 2*N={2 * N}")

        probs = torch.sigmoid(grid).view(B, N, N)

        row_clues, col_clues = clues.to(grid.device).view(B, 2, N, -1).unbind(1)
        out = self._compute(probs, row_clues, col_clues, self.binarize_weight)

        if not return_metrics:
            return out["loss"]
        return out["loss"], {k: v.item() for k, v in out.items()}

    @staticmethod
    @torch.no_grad()
    def evaluate(steps: np.ndarray, clues: np.ndarray) -> np.ndarray:
        """Per-step clue MSE for probabilities (n, S) against padded clues (2, N, K)."""
        n, S = steps.shape
        N = math.isqrt(S)
        if N * N != S:
            raise ValueError(f"Grid size {S} is not a perfect square")
        if clues.shape[1] < N:
            raise ValueError(f"Expected (2, {N}, K) clues, got {clues.shape}")

        probs = torch.from_numpy(steps).float().view(n, N, N)
        rows, cols = torch.from_numpy(np.asarray(clues[:, :N], dtype=np.float32)).unsqueeze(1).unbind(0)
        K = rows.shape[-1]

        row_loss = ((_runs(probs, K)[0] - rows) ** 2).sum(-1)
        col_loss = ((_runs(probs.transpose(1, 2), K)[0] - cols) ** 2).sum(-1)
        return (row_loss + col_loss).mean(-1).numpy()