import numpy as np

from ..block_types import MatrixNormBlock
from ..helpers import minmax, rank

class MatrixMinmax(MatrixNormBlock):
    def _apply(self, x: np.ndarray) -> np.ndarray:
        return minmax(x, self.eps)

class MatrixRank(MatrixNormBlock):
    def _apply(self, x: np.ndarray) -> np.ndarray:
         return rank(x, self.eps)