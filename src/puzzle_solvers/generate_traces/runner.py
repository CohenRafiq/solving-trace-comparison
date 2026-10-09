import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import numpy as np

from neural_models.dataset import NonogramsDataset

from puzzle_solvers.generate_traces.config import TracesConfig
from puzzle_solvers.generate_traces.record import make_puzzle_dict, to_serialisable
from puzzle_solvers.generate_traces.worker import build_solver_from_config, pool_init, process_puzzle

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


def _iter_results(config: TracesConfig, puzzle_args):
    if config.workers == 1:
        pool_init(config)
        yield from map(process_puzzle, puzzle_args)
        return

    with mp.get_context("spawn").Pool(
        processes=config.workers,
        initializer=pool_init,
        initargs=(config,),
    ) as pool:
        yield from pool.imap_unordered(
            process_puzzle, puzzle_args, chunksize=config.chunksize
        )


def _print_timing_table(totals: Counter, generated: int) -> None:
    if not totals:
        return

    grand_total = sum(totals.values())
    print(f"\n{'Solver':<35} {'Avg (s)':>10} {'Total (s)':>10} {'% Total':>8}")
    print("-" * 65)
    for name, total in totals.most_common():
        pct = 100 * total / grand_total if grand_total > 0 else 0
        print(f"{name:<35} {total / generated:>10.4f} {total:>10.2f} {pct:>7.1f}%")
    print("-" * 65)
    print(f"{'TOTAL':<35} {grand_total / generated:>10.4f} {grand_total:>10.2f}")


def run(config: TracesConfig) -> dict:
    start = time.perf_counter()
    config = config.resolve()

    dataset = NonogramsDataset(config.input, max_size=config.max_size)
    config.clues_shape = tuple(dataset.clues[0].shape)
    total_puzzles = len(dataset)
    print(f"Loaded {total_puzzles} puzzles")
    print(f"Solvers: {config.solvers}")

    for name in config.solvers:
        build_solver_from_config(name, config)

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals: Counter = Counter()
    generated = 0

    with output_path.open("w") as f:
        results = _iter_results(config, _iter_puzzle_args(dataset))
        for generated, (record, timing) in enumerate(results, start=1):
            f.write(json.dumps(to_serialisable(record)) + "\n")
            totals.update(timing)

            if generated % FLUSH_EVERY == 0:
                f.flush()
            if config.workers != 1 and (generated % PROGRESS_EVERY == 0 or generated == total_puzzles):
                print(f"  [{generated}/{total_puzzles}] puzzles done...")

    elapsed = time.perf_counter() - start

    print("\n=== Generation completed ===")
    print(f"Workers: {config.workers}  chunksize: {config.chunksize}")
    print(f"Puzzles: {generated}")
    print(f"Time:    {elapsed:.2f}s")
    _print_timing_table(totals, generated)
    print(f"\nOutput: {output_path}")

    return {"generated": generated, "elapsed": elapsed, "output": str(output_path)}