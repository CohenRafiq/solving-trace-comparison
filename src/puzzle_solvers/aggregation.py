from dataclasses import dataclass, field

import numpy as np
import torch

from puzzle_solvers.types import Aggregator

from .losses.clue_reconstruction_loss import ClueReconstructionLoss
from .losses.line_automata_loss import LineAutomatonLoss
from .solvers.utils.const import UNKNOWN, EMPTY, FILLED


@dataclass(frozen=True)
class FirstAggregator(Aggregator):
    """Return the first prediction, with -1 replaced by 0.5."""

    def __call__(self, predictions: np.ndarray, clues: np.ndarray) -> np.ndarray:
        arr = predictions[0].copy()
        arr[arr == UNKNOWN] = 0.5
        return arr

@dataclass(frozen=True)
class MeanAggregator(Aggregator):
    """Return the mean of the predictions, ignoring -1 values."""

    def __call__(self, predictions: np.ndarray, clues: np.ndarray) -> np.ndarray:
        if self.unknown_handling == "replace":
            predictions = np.where(predictions == UNKNOWN, 0.5, predictions)
            return np.mean(predictions, axis=0)
        elif self.unknown_handling == "skip":
            valid_mask = predictions != UNKNOWN
            valid_counts = np.sum(valid_mask, axis=0)
            valid_sums = np.sum(np.where(valid_mask, predictions, 0), axis=0)
            return np.divide(valid_sums, valid_counts, out=np.full_like(valid_sums, 0.5), where=valid_counts > 0)

@dataclass(frozen=True)
class WeightedMeanAggregator(Aggregator):
    """Return the weighted mean of the predictions,
    with weights based on clue reconstruction loss."""

    loss: ClueReconstructionLoss = field(default_factory=ClueReconstructionLoss)

    def _compute_sample_weights(
            self, losses: np.ndarray,
            temperature: float = 1.0) -> np.ndarray:
        normalised = (losses - losses.mean()) / (losses.std() + 1e-12)
        inverted = -normalised
        shifted = inverted - inverted.min()  # numerical stability for exp
        exp_scores = np.exp(shifted / temperature)
        return exp_scores / exp_scores.sum()

    def __call__(self, predictions: np.ndarray, clues: np.ndarray) -> np.ndarray:
        filled = np.where(predictions == UNKNOWN, 0.5, predictions)
        k = len(filled)
        losses = np.zeros(k)
        for i in range(k):
            grid = torch.tensor(filled[i:i+1], dtype=torch.float32).reshape(1, -1)
            target = torch.as_tensor(np.asarray(clues, dtype=np.float32)).reshape(1, -1)
            losses[i] = self.loss(grid, target).item()
        weights = self._compute_sample_weights(losses, temperature=1.0)
        if self.unknown_handling == "replace":
            return np.average(filled, axis=0, weights=weights)
        elif self.unknown_handling == "skip":
            valid_mask = predictions != UNKNOWN
            valid_counts = np.sum(valid_mask, axis=0)
            weighted_sums = np.sum(
                np.where(valid_mask, filled * weights[:, None, None], 0), axis=0)
            weight_sum = np.sum(weights[:, None, None] * valid_mask, axis=0)
            return np.where(valid_counts > 0, weighted_sums / weight_sum, 0.5)


@dataclass(frozen=True)
class BacktrackingWeightedAggregator(Aggregator):
    """Weighted mean for backtracking solvers.
    Only weights members that are fully solved (no unknowns)."""

    loss: LineAutomatonLoss = field(default_factory=lambda: LineAutomatonLoss(reduction="none"))
    logit_scale: float = 10.0

    def _compute_sample_weights(self, losses: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        finite_mask = np.isfinite(losses)
        if not np.any(finite_mask):
            return np.ones(len(losses)) / len(losses)
        finite_losses = losses[finite_mask]
        normalised = (finite_losses - finite_losses.mean()) / (finite_losses.std() + 1e-12)
        inverted = -normalised
        shifted = inverted - inverted.min()
        exp_scores = np.exp(shifted / temperature)
        weights = np.zeros_like(losses)
        weights[finite_mask] = exp_scores / exp_scores.sum()
        return weights

    def __call__(self, predictions: np.ndarray, clues: np.ndarray) -> np.ndarray:
        k = len(predictions)
        losses = np.zeros(k)
        for i in range(k):
            grid = predictions[i]
            if np.any(grid == UNKNOWN):
                losses[i] = np.inf  # Unsolved members get infinite loss
            else:
                logits = np.where(grid == FILLED, self.logit_scale,
                            np.where(grid == EMPTY, -self.logit_scale, 0.0))
                logits_t = torch.tensor(logits, dtype=torch.float32).reshape(1, -1)
                target = torch.as_tensor(np.asarray(clues, dtype=np.float32)).reshape(1, -1)
                losses[i] = self.loss(logits_t, target).item()
        weights = self._compute_sample_weights(losses, temperature=1.0)
        if self.unknown_handling == "replace":
            filled = np.where(predictions == UNKNOWN, 0.5, predictions)
            return np.average(filled, axis=0, weights=weights)
        elif self.unknown_handling == "skip":
            valid_mask = predictions != UNKNOWN
            valid_counts = np.sum(valid_mask, axis=0)
            weighted_sums = np.sum(
                np.where(valid_mask, predictions * weights[:, None, None], 0), axis=0)
            weight_sum = np.sum(weights[:, None, None] * valid_mask, axis=0)
            return np.where(valid_counts > 0, weighted_sums / weight_sum, 0.5)
