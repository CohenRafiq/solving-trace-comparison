from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from trace_comparison.blocks.resample import Resample
from trace_comparison.dataset import TraceDataset


class ClfDataset(Dataset):
    """Pairs a resampled trace `(k, W, H)` with its solver label."""

    def __init__(
        self,
        traces: np.ndarray,          # (M, k, W, H), float32
        labels: np.ndarray,          # (M,), int64
        classes: Sequence[str] | None = None,
        channel_first: bool = False,
    ) -> None:
        assert traces.ndim == 4, f"expected (M, k, W, H), got {traces.shape}"
        assert len(traces) == len(labels)
        self.traces = traces
        self.labels = labels
        self.classes = tuple(classes) if classes is not None else tuple(range(int(labels.max()) + 1))
        self.channel_first = channel_first

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        x = torch.from_numpy(self.traces[i])          # (k, W, H) = (C, W, H)
        if self.channel_first:
            x = x.permute(0, 2, 1).contiguous()       # (C, H, W)
        return x.float(), int(self.labels[i])


def build_clf_datasets(
    ds: TraceDataset,
    k: int,
    val_frac: float = 0.2,
    seed: int = 0,
    channel_first: bool = False,
) -> tuple[Dataset, Dataset]:
    """Resample every (solver, puzzle) trace to `k` steps; split across *puzzles*."""

    # (S, N, k, W, H) — the block handles ragged -> dense for you.
    dense = Resample(k).forward(ds.traces).astype(np.float32, copy=False)
    S, N = dense.shape[:2]

    # Flatten in (solver, puzzle) order so that sample index i corresponds to
    # (s, p) = (i // N, i % N).
    traces  = dense.reshape(S * N, *dense.shape[2:])      # (S*N, k, W, H)
    labels  = np.repeat(np.arange(S, dtype=np.int64), N)  # solver index
    puzzle  = np.tile (np.arange(N, dtype=np.int64), S)   # puzzle id per sample

    # ---- split over puzzles, not samples --------------------------------------
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val_p = max(1, int(round(val_frac * N)))
    n_val_p = min(n_val_p, N - 1)                          # keep >=1 train puzzle
    val_puzzles   = np.sort(perm[:n_val_p])

    # Boolean mask over flattened samples: include all solvers of a kept puzzle.
    in_val = np.isin(puzzle, val_puzzles)
    train_idx = np.flatnonzero(~in_val)
    val_idx   = np.flatnonzero( in_val)

    full = ClfDataset(traces, labels, ds.solvers, channel_first=channel_first)
    return (
        Subset(full, train_idx),
        Subset(full, val_idx),
    )