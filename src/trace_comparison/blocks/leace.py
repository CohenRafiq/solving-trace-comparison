import numpy as np

from trace_comparison.blocks.base import Block, DataKind


class LEACENormaliser(Block):
    """
    input_shape: (solvers, n_puzzles, ...)
    output_shape: (solvers, n_puzzles, ...)

    Normalises with LEACE, removing confounds.
    """
    input_kind = DataKind.DENSE
    output_kind = DataKind.DENSE

    _FNS = {
        "mean": lambda x: x.mean(axis=1),
        "std": lambda x: x.std(axis=1),
    }

    def __init__(self, confounds: tuple[str, ...] = ("mean", "std"), lam: float = 1e-2) -> None:
        unknown = set(confounds) - self._FNS.keys()
        if unknown:
            raise ValueError(f"Unknown confounds: {unknown}")
        self.confound_fns = [self._FNS[c] for c in confounds]
        self.lam = lam

    def output_shape(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError(f"Input shape must have at least 2 dimensions, got {input_shape}")
        return input_shape

    def fit_leace(self, x, z, tol=1e-8):
        mu = x.mean(0)
        Xc, Zc = x - mu, z - z.mean(0)
        N, d = Xc.shape
        Sxx = Xc.T @ Xc / N
        Sxz = Xc.T @ Zc / N
        Sxx = Sxx + self.lam * np.trace(Sxx) / d * np.eye(d)   # ridge

        evals, evecs = np.linalg.eigh(Sxx)
        keep = evals > tol * evals.max()
        U, s = evecs[:, keep], np.sqrt(evals[keep])
        W, W_pinv = (U / s).T, U * s

        Q, S, _ = np.linalg.svd(W @ Sxz, full_matrices=False)
        Q = Q[:, S > tol * max(S.max(), 1e-12)]
        return mu, W_pinv @ (Q @ Q.T) @ W

    def forward(self, x):
        flat = x.reshape(x.shape[0] * x.shape[1], -1)            # (N, d)
        z = np.stack([fn(flat) for fn in self.confound_fns], axis=1)   # (N, k)
        mu, M = self.fit_leace(flat, z)
        out = flat - (flat - mu) @ M.T
        return out.reshape(x.shape)
