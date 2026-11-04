from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

Runs = dict[str, dict[str, Any]]  # {solver: run record}

#: Creation-time puzzle filter: keep a puzzle only if every solver solved it,
#: keep everything, or an arbitrary predicate over ``{solver: run}``.
_FILTERS: dict[str | None, Callable[[Runs], bool]] = {
    None: lambda runs: True,
    "all_solved": lambda runs: all(r.get("solved", False) for r in runs.values()),
}


@dataclass(frozen=True, eq=False)
class TraceDataset:
    """`(solvers, puzzles)` object array of per-trace `(steps, width, height)` grids."""

    traces: np.ndarray  # (S, N) dtype=object; cell = float (T, W, H)
    solvers: tuple[str, ...]  # row order of `traces`
    grid_shape: tuple[int, int]  # (width, height), from meta.shape
    #: Per kept puzzle (column order of `traces`): ``[row_clues, col_clues]``, each a
    #: list of int lists.  ``None`` if any record lacks ``"clues"``.
    clues: tuple | None = None

    @property
    def shape(self) -> tuple:
        """Logical shape; the ragged step axis is `None`."""
        return (*self.traces.shape, None, *self.grid_shape)


def load_trace_dataset(
    path: str | Path,
    solvers: list[str] | None = None,
    max_size: int | None = None,
    puzzle_filter: str | None | Callable[[Runs], bool] = "all_solved",
) -> TraceDataset:
    keep = puzzle_filter if callable(puzzle_filter) else _FILTERS.get(puzzle_filter)
    if keep is None:
        raise ValueError(
            f"Unknown puzzle_filter {puzzle_filter!r}; "
            f"expected one of {list(_FILTERS)} or a callable"
        )

    selected = tuple(solvers) if solvers is not None else None
    grid_shape = None
    rows: list[list[np.ndarray]] = []  # rows[puzzle][solver]
    puzzle_clues: list[Any] = []  # puzzle_clues[puzzle]

    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            where = f"{path} line {i + 1}"
            if grid_shape is None:
                grid_shape = tuple(record["meta"]["shape"])
                if selected is None:
                    selected = tuple(record["traces"])
            if selected is None:
                raise ValueError(f"{path} contains no solver names")
            try:
                runs = {s: record["traces"][s] for s in selected}
            except KeyError as e:
                available = list(record["traces"])
                raise ValueError(
                    f"{where}: no run for solvers {list(selected)} ({e!r}); "
                    f"available {available}"
                ) from e
            if not keep(runs):
                continue
            rows.append([np.asarray(runs[s]["steps"], dtype=np.float32) for s in selected])
            puzzle_clues.append(record.get("clues"))
            if max_size is not None and len(rows) >= max_size:
                break

    if grid_shape is None or selected is None:
        raise ValueError(f"{path} contains no records")
    if not rows:
        raise ValueError(f"{path}: no puzzles kept (puzzle_filter={puzzle_filter!r})")

    traces: np.ndarray = np.empty((len(selected), len(rows)), dtype=object)
    for s, p in np.ndindex(traces.shape):  # cell-wise: numpy would try to stack ragged arrays
        traces[s, p] = rows[p][s]
    clues = tuple(puzzle_clues) if all(c is not None for c in puzzle_clues) else None
    return TraceDataset(traces, selected, grid_shape, clues)