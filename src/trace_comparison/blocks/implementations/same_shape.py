import numpy as np
from dataclasses import dataclass
from scipy.linalg import orthogonal_procrustes

from ..block_types import SameShapeBlock
from ..helpers import minmax, rank

class L2Norm(SameShapeBlock):
    def _apply(self, x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + self.eps)


class FrobeniusNorm(SameShapeBlock):
    def _apply(self, x):
        norm = np.linalg.norm(x, axis=(1, 2), keepdims=True)
        return x / (norm + self.eps)


class ZScoreNorm(SameShapeBlock):
    def _apply(self, x: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return np.where(std > self.eps, x / np.max(std, self.eps), 0.0)


class MinmaxNorm(SameShapeBlock):
    def _apply(self, x: np.ndarray) -> np.ndarray:
        return minmax(x, self.eps, -1)

class RankNorm(SameShapeBlock):
    def _apply(self, x):
        return rank(x, self.eps, -1)


class SubtractPuzzle(SameShapeBlock):
    def _apply(self, x: np.ndarray) -> np.ndarray:
        return x - x.mean(axis=1, keepdims=True)


class LeaceNorm(SameShapeBlock):
    def _apply(self, x: np.ndarray) -> np.ndarray:
        # TODO
        ...


@dataclass
class ProcrustesTransform(SameShapeBlock):
    # Expects preprocessing (subtract puzzle, frobenius norm)
    n_iter: int = 10

    def _apply(self, x: np.ndarray) -> np.ndarray:
        dtype = x.dtype
        x = x.astype(np.float64)
        out = np.empty_like(x)
        ref = x[0]

        for _ in range(self.n_iter):
            for i in range(x.shape[0]):
                R, _ = orthogonal_procrustes(x[i], ref)   # x[i] @ R ≈ ref
                out[i] = x[i] @ R
            ref = out.mean(axis=0)

        return out.astype(dtype)