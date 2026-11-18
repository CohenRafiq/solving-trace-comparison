"""Data helpers shared by every training script (classifier, imitation, ...).

Everything here works on the *flattened* ``(solver, puzzle)`` layout: sample ``i``
corresponds to ``(s, p) = (i // N, i % N)``.  The train/val split is always over
*puzzles* (all solvers of a puzzle land on the same side), so val metrics measure
generalisation to unseen puzzles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from trace_comparison.blocks.implementations.alignment import Resample
from trace_comparison.data.dataset import TraceDataset


@dataclass(frozen=True)
class FlatTraces:
    """Every (solver, puzzle) trace resampled to ``k`` steps and flattened."""

    traces: torch.Tensor  # (S*N, k, W, H) float32
    solver_idx: torch.Tensor  # (S*N,) int64
    puzzle_idx: torch.Tensor  # (S*N,) int64
    solvers: tuple[str, ...]
    n_puzzles: int


def flatten_traces(ds: TraceDataset, k: int) -> FlatTraces:
    """Resample to ``k`` steps (ragged -> dense) and flatten in (solver, puzzle) order."""
    dense = Resample(k).forward(ds.traces).astype(np.float32, copy=False)  # (S, N, k, W, H)
    S, N = dense.shape[:2]
    return FlatTraces(
        traces=torch.from_numpy(dense.reshape(S * N, *dense.shape[2:])),
        solver_idx=torch.from_numpy(np.repeat(np.arange(S, dtype=np.int64), N)),
        puzzle_idx=torch.from_numpy(np.tile(np.arange(N, dtype=np.int64), S)),
        solvers=tuple(ds.solvers),
        n_puzzles=N,
    )


def split_by_puzzle(
    puzzle_idx: torch.Tensor, n_puzzles: int, val_frac: float = 0.2, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(train_idx, val_idx)`` over flattened samples, split across puzzles."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_puzzles)
    n_val = min(max(1, int(round(val_frac * n_puzzles))), n_puzzles - 1)  # keep >=1 train puzzle
    in_val = np.isin(puzzle_idx.numpy(), np.sort(perm[:n_val]))
    return torch.from_numpy(np.flatnonzero(~in_val)), torch.from_numpy(np.flatnonzero(in_val))


def select(tensors: tuple[torch.Tensor, ...], idx: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Index every tensor in a data tuple with the same indices."""
    return tuple(t[idx] for t in tensors)


# --------------------------------------------------------------------- clues


def clue_shape(grid_shape: tuple[int, int]) -> tuple[int, int, int]:
    """Padded clue array shape ``(2, L, M)``: [rows, cols] x lines x blocks per line.

    ``L = max(W, H)`` and a line of length ``L`` holds at most ``ceil(L / 2)`` blocks.
    """
    length = max(grid_shape)
    return 2, length, (length + 1) // 2


def encode_clues(clues, shape: tuple[int, int, int]) -> np.ndarray:
    """``[[[1], [3], ...], [[2], [1, 1], ...]]`` -> zero-padded ``(2, L, M)`` array.

    Block sizes are divided by ``L`` so they lie in ``[0, 1]``.  Blocks are
    left-aligned within a line; an empty line / ``[0]`` encodes as zeros.
    """
    _, length, max_blocks = shape
    out = np.zeros(shape, dtype=np.float32)
    if len(clues) != 2:
        raise ValueError(f"expected [row_clues, col_clues], got {len(clues)} groups")
    for axis, lines in enumerate(clues):
        if len(lines) > length:
            raise ValueError(f"{len(lines)} clue lines exceed padded length {length}")
        for i, line in enumerate(lines):
            if len(line) > max_blocks:
                raise ValueError(f"line {line} has more than {max_blocks} blocks")
            out[axis, i, : len(line)] = np.asarray(line, dtype=np.float32) / length
    return out


def encode_all_clues(clues, grid_shape: tuple[int, int]) -> torch.Tensor:
    """Per-puzzle clues -> ``(N, 2*L*M)`` float32 feature matrix."""
    shape = clue_shape(grid_shape)
    return torch.from_numpy(np.stack([encode_clues(c, shape).ravel() for c in clues]))