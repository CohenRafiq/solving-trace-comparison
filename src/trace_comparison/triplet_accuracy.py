"""Triplet accuracy of a solver similarity matrix.

Usage: python triplet_accuracy.py [name]
Reads $DATA_DIR/similarity_matrices/nonograms/<name>.csv.
"""

import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

#: (more related level, less related level) comparisons to score.
LEVEL_PAIRS = [(2, 1), (2, 0), (1, 0)]

def relatedness_levels(names) -> np.ndarray:
    """`level[i, j]`: 2 = same `XX_YYY` variant, 1 = same `XX` solver only, 0 = different."""
    parts = [n.split("_") for n in names]
    solver = np.array([p[0] for p in parts])
    variant = np.array(["_".join(p[:2]) for p in parts])
    return (solver[:, None] == solver).astype(int) + (variant[:, None] == variant)


def triplet_accuracy(sim: np.ndarray, level: np.ndarray) -> pd.DataFrame:
    """How often an anchor is more similar to a more related item than to a less related one.

    For each anchor `a` and each pair (p, q) with `level[a, p] > level[a, q]`, the
    triplet is correct if `sim[a, p] > sim[a, q]` (ties count as wrong). Returns
    correct/total/accuracy per level comparison; pure array maths, no solver names.
    """
    n = len(sim)
    off_diag = ~np.eye(n, dtype=bool)  # an anchor is never compared with itself
    counts = np.zeros((len(LEVEL_PAIRS), 2), dtype=np.int64)  # (correct, total)
    sim_rows = sim[off_diag].reshape(n, -1)
    level_rows = level[off_diag].reshape(n, -1)
    for scores, levels in zip(sim_rows, level_rows, strict=True):
        hit = scores[:, None] > scores  # hit[p, q]: p ranked above q
        for k, (hi, lo) in enumerate(LEVEL_PAIRS):
            pairs = (levels[:, None] == hi) & (levels == lo)
            counts[k] += (pairs & hit).sum(), pairs.sum()
    table = pd.DataFrame(
        counts, columns=["correct", "total"], index=[f"L{hi} > L{lo}" for hi, lo in LEVEL_PAIRS]
    )
    table["accuracy"] = table["correct"] / table["total"]
    return table


def load_sim_matrix(name: str) -> pd.DataFrame:
    base = Path(os.environ.get("DATA_DIR", "data"))
    sim = pd.read_csv(
        base / "similarity_matrix" / "nonograms" / f"{name.removesuffix('.csv')}.csv", index_col=0
    )
    if not sim.index.equals(sim.columns):
        raise ValueError("Similarity matrix rows and columns must match")
    return sim


def main() -> None:
    parser = ArgumentParser(description="Compute triplet accuracy for a similarity matrix")
    parser.add_argument(
        "name",
        nargs="?",
        default="temp",
        help="matrix file name, with or without .csv",
    )
    sim = load_sim_matrix(parser.parse_args().name)
    table = triplet_accuracy(sim.to_numpy(), relatedness_levels(sim.index))
    print(f"Overall triplet accuracy: {table['correct'].sum() / table['total'].sum():.4f}")
    print(table)


if __name__ == "__main__":
    main()
