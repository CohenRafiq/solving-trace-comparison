import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from neural_models.dataset import NonogramsDataset
from puzzle_solvers.generate_traces.config import resolve
from puzzle_solvers.generate_traces.factory import build_solvers
from puzzle_solvers.generate_traces.record import make_puzzle_dict, to_serialisable
from puzzle_solvers.generate_traces.worker import pool_init, process_puzzle

FLUSH_EVERY = 10
PROGRESS_EVERY = 50


def _iter_puzzle_args(dataset: NonogramsDataset):
    for idx in range(len(dataset)):
        yield (
            idx,
            make_puzzle_dict(dataset, idx),
            dataset.clues[idx].numpy().astype(np.float32),
            dataset.grid[idx].numpy().astype(np.float32),
        )


def _iter_results(cfg: DictConfig, solvers, puzzle_args):
    initargs = (solvers, float(cfg.max_time), cfg.get("seed"))

    if cfg.workers == 1:
        pool_init(*initargs)
        yield from map(process_puzzle, puzzle_args)
        return

    with mp.get_context("spawn").Pool(
        processes=cfg.workers, initializer=pool_init, initargs=initargs
    ) as pool:
        yield from pool.imap_unordered(process_puzzle, puzzle_args, chunksize=cfg.chunksize)


def _count_solved(record: dict, solved: Counter, attempts: Counter) -> None:
    for name, runs in record["traces"].items():
        solved[name] += sum(r["solved"] for r in runs)
        attempts[name] += len(runs)


def _print_timing_table(totals: Counter, solved: Counter, attempts: Counter, generated: int) -> None:
    if not totals:
        return

    grand_total = sum(totals.values())
    print(f"\n{'Solver':<35} {'Avg (s)':>10} {'Total (s)':>10} {'% Total':>8} {'Accuracy':>10}")
    print("-" * 76)
    for name, total in totals.most_common():
        pct = 100 * total / grand_total if grand_total > 0 else 0
        acc = 100 * solved[name] / max(attempts[name], 1)
        print(f"{name:<35} {total / generated:>10.4f} {total:>10.2f} {pct:>7.1f}% {acc:>9.1f}%")
    print("-" * 76)
    print(f"{'TOTAL':<35} {grand_total / generated:>10.4f} {grand_total:>10.2f}")

def run(cfg: DictConfig) -> dict:
    start = time.perf_counter()
    cfg = resolve(cfg)

    dataset = instantiate(cfg.dataset, file_name=cfg.input, max_size=cfg.max_size)
    total_puzzles = len(dataset)
    solvers = build_solvers(cfg)

    print(f"Loaded {total_puzzles} puzzles")
    print(f"Solvers: {list(solvers)}")

    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals: Counter = Counter()
    solved: Counter = Counter()
    attempts: Counter = Counter()
    generated = 0

    with output_path.open("w") as f:
        results = _iter_results(cfg, solvers, _iter_puzzle_args(dataset))
        for generated, (record, timing) in enumerate(results, start=1):
            f.write(json.dumps(to_serialisable(record)) + "\n")
            totals.update(timing)
            _count_solved(record, solved, attempts)
            if generated % FLUSH_EVERY == 0:
                f.flush()
            if cfg.workers != 1 and (generated % PROGRESS_EVERY == 0 or generated == total_puzzles):
                print(f"  [{generated}/{total_puzzles}] puzzles done...")

    elapsed = time.perf_counter() - start

    print("\n=== Generation completed ===")
    print(f"Workers: {cfg.workers}  chunksize: {cfg.chunksize}")
    print(f"Puzzles: {generated}")
    print(f"Time:    {elapsed:.2f}s")
    _print_timing_table(totals, solved, attempts, generated)
    print(f"\nOutput: {output_path}")

    return {"generated": generated, "elapsed": elapsed, "output": str(output_path)}
