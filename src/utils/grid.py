from collections.abc import Iterable
from itertools import groupby

import numpy as np
import torch

from utils.puzzle_types import Grid


def get_line_clues(line: Iterable[int] | torch.Tensor, K: int | None = None):
    is_tensor = isinstance(line, torch.Tensor)
    if is_tensor:
        values = line.detach().cpu().tolist()  # type: ignore[union-attr]
    else:
        values = list(line)
    runs = [len(list(g)) for v, g in groupby(values) if v]

    if K is None:
        return runs

    padded = runs[:K] + [0] * (K - len(runs))
    return (
        torch.tensor(padded, dtype=torch.float32) if is_tensor else np.array(padded, dtype=np.int32)
    )


def derive_clues_from_grid(grid: Grid, K: int | None = None):
    if isinstance(grid, torch.Tensor):
        return _derive_clues_from_grid_torch(grid, K)
    if isinstance(grid, np.ndarray):
        return _derive_clues_from_grid_np(grid, K)
    row_clues = [get_line_clues(row, K) for row in grid]
    col_clues = [get_line_clues(col, K) for col in zip(*grid, strict=True)]
    return row_clues, col_clues


def _derive_clues_from_grid_np(grid: np.ndarray, K: int | None):
    row_clues = [get_line_clues(row, K) for row in grid]
    col_clues = [get_line_clues(col, K) for col in grid.T]
    return row_clues, col_clues


def _derive_clues_from_grid_torch(grid: torch.Tensor, K: int | None):
    H, W = grid.shape
    K = K or max(H, W)
    row_clues = torch.stack([get_line_clues(row, K) for row in grid])
    col_clues = torch.stack([get_line_clues(col, K) for col in grid.T])
    return row_clues, col_clues


def batch_line_clues(lines: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    M, L = lines.shape
    padded = np.zeros((M, L + 2), dtype=lines.dtype)
    padded[:, 1:-1] = lines
    diff = np.diff(padded, axis=1)

    is_start = diff == 1
    is_end = diff == -1

    start_counts = np.cumsum(is_start, axis=1)
    num_runs = start_counts[:, -1]

    rows, starts = np.nonzero(is_start)
    _, ends = np.nonzero(is_end)
    run_index = start_counts[rows, starts] - 1

    run_lengths = np.zeros((M, K), dtype=np.int32)
    keep = run_index < K
    run_lengths[rows[keep], run_index[keep]] = (ends - starts)[keep]

    return run_lengths, num_runs


def normalise_clues(clues: np.ndarray | torch.Tensor) -> np.ndarray:
    arr = clues.detach().cpu().numpy() if isinstance(clues, torch.Tensor) else np.asarray(clues)
    rows, cols = grid_shape_from_clues(arr)
    k_row = (cols + 1) // 2
    k_col = (rows + 1) // 2
    flat_len = rows * k_row + cols * k_col

    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr.reshape(-1)

    if arr.ndim == 1:
        if arr.shape[0] != flat_len:
            raise ValueError(
                f"Expected {flat_len} flattened clue values for a {rows}x{cols} "
                f"grid ({rows}x{k_row} row clues + {cols}x{k_col} column clues), "
                f"got {arr.shape[0]}"
            )
        if k_row != k_col:
            raise ValueError(
                "Flattened clues are only supported for square grids; pass "
                "row/column clues as a (2, ...) array instead"
            )
        row_clues = arr[: rows * k_row].reshape(rows, k_row)
        col_clues = arr[rows * k_row :].reshape(cols, k_col)
        return np.stack([row_clues, col_clues])

    if arr.ndim == 3 and arr.shape[0] == 2:
        return arr

    raise ValueError(
        f"clues must be a tensor/array of shape ({flat_len},) or "
        f"(1, {flat_len}) with the flat [row clues; column clues] dataloader "
        f"layout, or a stacked array of shape (2, H, K). Got shape {arr.shape}"
    )


def ternarise(arr: np.ndarray, epsilon: float = 1e-2) -> np.ndarray:
    # -1 = unknown, 0 = empty, 1 = filled
    out = np.full(arr.shape, -1, dtype=int)
    out[arr <= epsilon] = 0
    out[arr >= 1 - epsilon] = 1
    return out


def unpad_clue(clue) -> tuple[int, ...]:
    if hasattr(clue, "tolist"):
        clue = clue.tolist()
    flat: list[int] = []
    stack = [clue]
    while stack:
        item = stack.pop(0)
        if isinstance(item, (list, tuple)):
            stack = list(item) + stack
        else:
            flat.append(int(item))
    return tuple(b for b in flat if b > 0)

def clues_to_tuples(clues: np.ndarray) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """(2, N, K) padded clues -> per-axis tuples of unpadded int clues."""
    ints = np.asarray(clues).astype(int).tolist()
    return tuple(
        tuple(tuple(v for v in line if v > 0) for line in half) for half in ints
    )

def pad_clues(clues: list[list[list[int]]], pad_value=0):
    grid_side_length = max(len(clues[0]), len(clues[1]))
    k = (grid_side_length + 1) // 2
    target_shape = (2, grid_side_length, k)
    padded_tensor = torch.full(target_shape, pad_value, dtype=torch.float32)
    for i, clue_set in enumerate(clues):
        for j, clue in enumerate(clue_set):
            padded_tensor[i, j, : len(clue)] = torch.tensor(clue, dtype=torch.float32)

    return padded_tensor


def check_clue(line: np.ndarray, clue: np.ndarray, epsilon: float = 1e-2) -> bool:
    rounded = ternarise(np.asarray(line), epsilon).tolist()
    if -1 in rounded:
        return False
    return get_line_clues(rounded) == list(clue)


def is_solved(clues: np.ndarray, grid: np.ndarray, epsilon: float = 1e-2) -> bool:
    if clues.ndim == 3:
        row_clues, col_clues = clues[0], clues[1]
    else:
        num_rows = len(grid)
        row_clues, col_clues = clues[:num_rows], clues[num_rows:]

    return all(check_clue(grid[i], row_clues[i], epsilon) for i in range(len(grid))) and all(
        check_clue(grid[:, j], col_clues[j], epsilon) for j in range(grid.shape[1])
    )


def grid_shape_from_clues(clues: np.ndarray) -> tuple[int, int]:
    # TODO handle rectangular grids
    if clues.ndim == 3 and clues.shape[0] == 2:
        return clues.shape[1], clues.shape[1]
    if clues.ndim == 1:
        flat_len = clues.shape[0]
        for rows in range(1, flat_len):
            cols = flat_len - rows
            k_row = (cols + 1) // 2
            k_col = (rows + 1) // 2
            if rows * k_row + cols * k_col == flat_len:
                return rows, cols
    raise ValueError(f"Cannot determine grid shape from clues with shape {clues.shape}")


def blank_grid(rows: int, cols: int) -> np.ndarray:
    return np.full((rows, cols), 0.5, dtype=np.float32)
