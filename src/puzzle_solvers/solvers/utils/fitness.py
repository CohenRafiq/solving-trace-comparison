import numpy as np
from functools import lru_cache
from itertools import accumulate, combinations


@lru_cache(maxsize=10_000)
def _valid_masks(n: int, clues: tuple[int, ...]) -> np.ndarray:
    """All n-bit masks (index 0 -> MSB) whose runs equal `clues`. Treat as read-only."""
    offs = (0, *accumulate(clues))           # offs[i] = total length of blocks before i
    return np.array(
        [sum(((1 << b) - 1) << (n - p - o - b) for b, o, p in zip(clues, offs, ps))
         for ps in combinations(range(n - sum(clues) + 1), len(clues))],
        dtype=np.int64,
    )


@lru_cache(maxsize=10_000)
def _weights(n: int) -> np.ndarray:
    """Powers of two for MSB-first packing of n bits."""
    return 1 << np.arange(n - 1, -1, -1, dtype=np.int64)


@lru_cache(maxsize=10_000)
def _fitness_cached(guess: int, n: int, clues: tuple[int, ...]) -> float:
    """Cached on (packed line, length, clues): 1 - nearest-valid Hamming distance / n."""
    valid = _valid_masks(n, clues)
    if not valid.size:
        return 0.0
    return 1.0 - int(np.bitwise_count(valid ^ guess).min()) / n


def line_fitness(line: np.ndarray, clues: tuple[int, ...]) -> float:
    """Fitness in [0, 1] of a single bool line vs. its clues (cached)."""
    return _fitness_cached(int(line @ _weights(line.size)), line.size, tuple(clues))


def line_fitness_batch(lines: np.ndarray, clues: tuple[int, ...]) -> np.ndarray:
    """lines: (k, n) bool array -> (k,) fitnesses. Shares the same cache as line_fitness."""
    n, clues = lines.shape[1], tuple(clues)
    return np.fromiter(
        (_fitness_cached(g, n, clues) for g in (lines @ _weights(n)).tolist()),
        dtype=np.float64, count=len(lines),
    )


def fitness(grid: np.ndarray, blocks: tuple[tuple[tuple[int, ...], ...], ...]) -> float:
    """
    Total fitness of `grid` (h, w) bool.
    blocks[0] = row clues (length h), blocks[1] = column clues (length w).
    Range is [0, h + w]; a perfect solution scores exactly h + w.
    """
    rows, cols = blocks
    return (sum(map(line_fitness, grid, rows)) +
            sum(map(line_fitness, grid.T, cols)))

def population_fitness(population: np.ndarray, blocks) -> np.ndarray:
    """(k, h, w) bool -> (k,) total fitness per individual; a solution scores h + w."""
    rows, cols = blocks
    if len(rows) != population.shape[1] or len(cols) != population.shape[2]:
        raise ValueError("clue count does not match grid shape")
    total = np.zeros(len(population))
    for r, clue in enumerate(rows):
        total += line_fitness_batch(population[:, r, :], clue)
    for c, clue in enumerate(cols):
        total += line_fitness_batch(population[:, :, c], clue)
    return total


def cache_info():
    """Hit/miss stats, handy when tuning a GA/annealing loop."""
    return {f.__name__: f.cache_info() for f in (_valid_masks, _fitness_cached)}
