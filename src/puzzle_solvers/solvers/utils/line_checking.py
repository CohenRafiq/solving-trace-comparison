"""Nonogram line analysis: feasibility, forced cells and marginals.

A line is described by `blocks` (the clue, e.g. (3, 1)) and `known`, a tuple of
cells that are UNKNOWN (-1), EMPTY (0) or FILLED (1).

Two engines, chosen by what the caller needs:

* Reachability engine (`line_feasible`, `line_forced`): the clue is compiled to a
  tiny automaton whose state set is one Python int, so each cell costs a few
  bit operations. Needs no counts, so it is the fast path for propagation.
* Counting engine (`line_counts`, `line_marginal`): counts placements block by
  block. It is only run for lines the reachability engine has already shown to
  be feasible, so contradictions (common while backtracking) cost almost nothing.
  Three ideas keep it cheap:
    - Slack band: block i can only start in a window `slack + 1` wide, so a layer
      is a short list indexed by shift from the leftmost start.
    - Prefix sums: valid predecessors of a placement form one contiguous range,
      so summing over them is O(1).
    - Mirroring: the backward pass is the same sweep on the reversed line.
"""

from __future__ import annotations

from functools import cache, lru_cache
from itertools import accumulate
from .const import EMPTY, FILLED, UNKNOWN

import numpy as np


def _spare_cells(blocks: tuple[int, ...], line_length: int) -> int:
    """Spare cells once the blocks are packed tightly to the left (negative if impossible)."""
    return line_length - (sum(blocks) + len(blocks) - 1)


def _sweep(
    blocks: tuple[int, ...],
    known: tuple[int, ...],
    spare: int,
) -> tuple[list[list[int]], int] | None:
    """Forward pass. Returns (layer_counts, total_placements), or None if no placement exists.

    layer_counts[block_index][extra_offset] = number of ways to place blocks 0..block_index
    with block_index shifted `extra_offset` cells right of its leftmost possible start.
    `total_placements` counts complete placements.
    Requires at least one block and spare >= 0.
    """
    line_length = len(known)
    filled_prefix = [0] * (line_length + 1)      # FILLED cells in known[:i]
    empty_prefix = [0] * (line_length + 1)       # EMPTY cells in known[:i]
    after_last_filled = [0] * (line_length + 1)  # smallest e such that known[e:i] has no FILLED
    for i, cell in enumerate(known):
        filled_prefix[i + 1] = filled_prefix[i] + (cell == FILLED)
        empty_prefix[i + 1] = empty_prefix[i] + (cell == EMPTY)
        after_last_filled[i + 1] = i + 1 if cell == FILLED else after_last_filled[i]

    layer_counts: list[list[int]] = []
    previous_prefix_sums: list[int] | None = None  # prefix sums of the previous layer
    previous_block_size = previous_block_start = 0
    leftmost_start = 0  # leftmost start of the current block

    for block_size in blocks:
        layer = [0] * (spare + 1)
        for extra_offset in range(spare + 1):
            start = leftmost_start + extra_offset
            if empty_prefix[start + block_size] != empty_prefix[start]:
                continue  # block would cover an EMPTY cell
            if previous_prefix_sums is None:
                layer[extra_offset] = filled_prefix[start] == 0  # nothing FILLED to its left
            else:
                # The previous block must end before `start` and leave no FILLED
                # cell in the gap, so its extra_offset lies in [lowest, extra_offset].
                lowest = max(
                    0,
                    after_last_filled[start] - previous_block_size - previous_block_start,
                )
                if lowest <= extra_offset:
                    layer[extra_offset] = (
                        previous_prefix_sums[extra_offset + 1]
                        - previous_prefix_sums[lowest]
                    )
        if not any(layer):
            return None
        layer_counts.append(layer)
        previous_prefix_sums = [0, *accumulate(layer)]
        previous_block_size, previous_block_start = block_size, leftmost_start
        leftmost_start += block_size + 1

    # The last block must leave no FILLED cell to its right.
    lowest = max(
        0,
        after_last_filled[line_length] - previous_block_size - previous_block_start,
    )
    total_placements = (
        previous_prefix_sums[-1] - previous_prefix_sums[min(lowest, spare + 1)]
    )
    return (layer_counts, total_placements) if total_placements else None


# --- Reachability engine -----------------------------------------------------


@cache
def _compile_clue(blocks: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Compile a clue to bitmasks. Bit p means "p symbols of the clue pattern matched".

    The pattern is FILLED*size per block with one mandatory EMPTY between blocks,
    e.g. (3, 1) -> 1 1 1 0 1. Extra empties may repeat at both ends and after
    each separator. Returns (needs_filled, needs_empty, empty_loops, final_state).
    """
    pattern: list[int] = []
    for index, size in enumerate(blocks):
        if index:
            pattern.append(EMPTY)
        pattern.extend([FILLED] * size)

    final_state = len(pattern)
    needs_filled = needs_empty = 0
    empty_loops = 1 | (1 << final_state)
    for state, symbol in enumerate(pattern):
        if symbol == FILLED:
            needs_filled |= 1 << state
        else:
            needs_empty |= 1 << state
            empty_loops |= 1 << (state + 1)
    return needs_filled, needs_empty, empty_loops, final_state


def _reachable(blocks: tuple[int, ...], known: tuple[int, ...]) -> list[int] | None:
    """States reachable after each prefix of the line (index 0 = empty prefix).

    None as soon as every state dies, or if the clue cannot end in the final state.
    """
    needs_filled, needs_empty, empty_loops, final_state = _compile_clue(blocks)
    active = 1
    states = [active]
    for cell in known:
        following = 0
        if cell != EMPTY:  # FILLED or UNKNOWN: consume a filled symbol
            following = (active & needs_filled) << 1
        if cell != FILLED:  # EMPTY or UNKNOWN: consume a separator or idle in a gap
            following |= ((active & needs_empty) << 1) | (active & empty_loops)
        if not following:
            return None
        active = following
        states.append(active)
    return states if active >> final_state & 1 else None


def _line_feasible(blocks: tuple[int, ...], known: tuple[int, ...]) -> bool:
    """Whether the partial line can still be completed to match `blocks`."""
    if _spare_cells(blocks, len(known)) < 0:
        return False
    return _reachable(blocks, known) is not None

# --- Public API --------------------------------------------------------------

def consistent_at_cell(grid: np.ndarray, row: int, col: int, blocks) -> bool:
    """Whether the partial grid is consistent with the clues at a given cell."""
    row_blocks, col_blocks = blocks
    return (
        _line_feasible(tuple(grid[row, :].tolist()), row_blocks[row])
        and _line_feasible(tuple(grid[:, col].tolist()), col_blocks[col])
    )

@lru_cache(maxsize=100_000)
def line_counts(
    blocks: tuple[int, ...], known: tuple[int, ...]
) -> tuple[tuple[int, ...], int] | None:
    """(counts, total) over placements consistent with `known`; None if there are none.

    counts[i] is the number of valid placements with cell i filled.
    """
    length = len(known)
    if not blocks:
        return ((0,) * length, 1) if FILLED not in known else None
    if not line_feasible(blocks, known):  # cheap rejection before counting
        return None

    spare_cells = _spare_cells(blocks, length)
    layers, total = _sweep(blocks, known, spare_cells)  # type: ignore[misc]  # feasible, so not None
    mirrored_layers, _ = _sweep(blocks[::-1], known[::-1], spare_cells)  # type: ignore[misc]

    # Placements through (block, start) = ways to its left * ways to its right.
    # Mirroring flips the shift, so the right-hand count is mirrored[spare_cells - shift].
    fill_delta = [0] * (length + 1)
    offset = 0
    for index, size in enumerate(blocks):
        mirrored = mirrored_layers[len(blocks) - 1 - index]
        for shift, ways_left in enumerate(layers[index]):
            if ways_left:
                through = ways_left * mirrored[spare_cells - shift]
                fill_delta[offset + shift] += through
                fill_delta[offset + shift + size] -= through
        offset += size + 1

    return tuple(accumulate(fill_delta))[:length], total


def find_forced_cells_from_line(
    blocks: tuple[int, ...], known: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """(must_fill, must_empty) masks over unknown cells.

    Uses reachability only: a value is possible at a cell iff some state reachable
    from the left can step into a state that can still finish on the right.
    """
    length = len(known)
    if _spare_cells(blocks, length) < 0:
        raise ValueError("line is contradictory: too many cells for the line")
    reachable = _reachable(blocks, known)
    if reachable is None:
        raise ValueError("line is contradictory: no reachable states")

    needs_filled, needs_empty, empty_loops, final_state = _compile_clue(blocks)
    must_fill = np.zeros(length, dtype=bool)
    must_empty = np.zeros(length, dtype=bool)
    can_finish = 1 << final_state  # states from which known[position:] can complete
    for position in range(length - 1, -1, -1):
        cell = known[position]
        via_filled = (can_finish >> 1) & needs_filled
        via_empty = ((can_finish >> 1) & needs_empty) | (can_finish & empty_loops)
        filled_ok = cell != EMPTY and bool(reachable[position] & via_filled)
        empty_ok = cell != FILLED and bool(reachable[position] & via_empty)
        if cell == UNKNOWN:
            must_fill[position] = filled_ok and not empty_ok
            must_empty[position] = empty_ok and not filled_ok
        can_finish = (via_filled if cell != EMPTY else 0) | (via_empty if cell != FILLED else 0)
    return must_fill, must_empty


def line_marginal(blocks: tuple[int, ...], known: tuple[int, ...]) -> np.ndarray:
    """P(cell filled) over valid placements; 0.5 everywhere if the line is contradictory."""
    result = line_counts(blocks, known)
    if result is None:
        return np.full(len(known), 0.5)
    counts, total = result
    return np.fromiter((c / total for c in counts), dtype=np.float64, count=len(known))

def get_line(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    """View of a row or column of a 2D array."""
    return array[index, :] if axis == 0 else array[:, index]