from dataclasses import dataclass, field

from puzzle_solvers.types import Aggregator
import numpy as np
import torch

from .losses.clue_reconstruction_loss import ClueReconstructionLoss
from .solvers.utils.const import UNKNOWN

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
            return np.where(valid_counts > 0, valid_sums / valid_counts, 0.5)

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
        losses = self.loss(
            torch.tensor(filled, dtype=torch.float32), 
            torch.tensor(clues, dtype=torch.float32)).numpy()
        weights = self._compute_sample_weights(losses, temperature=1.0)
        if self.unknown_handling == "replace":
            return np.average(filled, axis=0, weights=weights)
        elif self.unknown_handling == "skip":
            valid_mask = predictions != UNKNOWN
            valid_counts = np.sum(valid_mask, axis=0)
            weighted_sums = np.sum(
                np.where(valid_mask, filled * weights[:, None, None], 0), axis=0)
            return np.where(valid_counts > 0, weighted_sums / valid_counts, 0.5)