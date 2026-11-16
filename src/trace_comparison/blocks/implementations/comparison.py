import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.spatial import procrustes
import numpy as np
from dataclasses import dataclass, field
from typing import ClassVar
from sklearn.metrics.pairwise import (
    pairwise_kernels, euclidean_distances, manhattan_distances,
)


from ..block_types import ComparisonBlock

@dataclass
class MeanAbsoluteError(ComparisonBlock):
    is_distance: ClassVar[bool] = True

    def _apply(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.mean(abs(a - b))


@dataclass
class CosineSimilarity(ComparisonBlock):
    def _apply(self, a: np.ndarray, b: np.ndarray) -> float:
        dot_product = np.sum(a * b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))


@dataclass
class CanonicalCorrelationAnalysis(ComparisonBlock):
    n_components: int
    squared: bool

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        output = super().output_shape(input_shape)
        if self.n_components >= input_shape[1] - 1:
            raise ValueError(
                f"n_components={self.n_components} must be < samples - 1 "
                f"({input_shape[1] - 1}), otherwise every correlation is trivially 1"
            )
        return output

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        usvs = [np.linalg.svd(xs, full_matrices=False) for xs in x]
        ranks = [int((sv > self.eps * max(sv[0], self.eps)).sum()) for _, sv, _ in usvs]
        r = min(self.n_components, *ranks)
        if r == 0:  # every solver is constant across samples: nothing to correlate
            return None
        bases = np.stack([u[:, :r] for u, _, _ in usvs])  # (solvers, samples, r)
        return bases

    def _apply(self, a: np.ndarray, b: np.ndarray) -> float:
        rho = np.clip(np.linalg.svd(a.T @ b, compute_uv=False), 0.0, 1.0)
        return float(np.mean(rho**2 if self.squared else rho))


@dataclass
class CentredKernalAlignment(ComparisonBlock):
    def _apply(self, a: np.ndarray, b: np.ndarray) -> float:
        cross = np.linalg.norm(a.T @ b, ord="fro") ** 2
        norm_a = np.linalg.norm(a.T @ a, ord="fro")
        norm_b = np.linalg.norm(b.T @ b, ord="fro")
        return float(cross / (norm_a * norm_b + self.eps))


@dataclass
class MaximumMeanDiscrepancy(ComparisonBlock):
    is_distance: ClassVar[bool] = True

    kernel: str = "rbf"           # any sklearn kernel: "rbf", "laplacian", "linear", ...
    gamma: float | None = None    # None -> median heuristic for rbf/laplacian
    unbiased: bool = False        # biased estimate is never negative
    squared: bool = True          # True -> MMD^2 (no clamp, no sqrt, no ties)
    max_gamma_samples: int = 1000

    _gamma: float | None = field(init=False, default=None, repr=False)

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        """Pick one gamma for ALL solver pairs so the values are comparable."""
        self._gamma = self.gamma
        if self.gamma is None and self.kernel in ("rbf", "laplacian"):
            flat = x.reshape(-1, x.shape[-1])
            rng = np.random.default_rng(0)
            idx = rng.choice(len(flat), min(len(flat), self.max_gamma_samples), replace=False)
            sub = flat[idx]
            if self.kernel == "rbf":
                d = euclidean_distances(sub, squared=True)
                scale = 2.0          # exp(-gamma * ||u-v||^2), gamma = 1 / (2 * median d^2)
            else:
                d = manhattan_distances(sub)
                scale = 1.0          # exp(-gamma * ||u-v||_1), gamma = 1 / median d
            d = d[d > 0]
            if d.size:
                self._gamma = 1.0 / (scale * np.median(d))
            # else: leave None -> sklearn default 1 / n_features
        return x

    def _apply(self, a: np.ndarray, b: np.ndarray) -> float:
        def k(u, v):
            return pairwise_kernels(
                u, v, metric=self.kernel, gamma=self._gamma, filter_params=True
            )

        kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
        n, m = len(a), len(b)

        if self.unbiased and n > 1 and m > 1:
            aa = (kaa.sum() - np.trace(kaa)) / (n * (n - 1))
            bb = (kbb.sum() - np.trace(kbb)) / (m * (m - 1))
        else:
            aa, bb = kaa.mean(), kbb.mean()

        mmd2 = float(aa + bb - 2 * kab.mean())
        if self.squared:
            return mmd2                        # may be slightly negative if unbiased; fine for ranking
        return float(np.sqrt(max(mmd2, 0.0)))  # clamp only needed for the root


@dataclass
class ProcrustesCompare(ComparisonBlock):
    is_distance: ClassVar[bool] = True

    def _apply(self, a, b):
        _, _, disparity = procrustes(a, b)
        return disparity