import gc
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict

from puzzle_solvers.generate_traces.config import TracesConfig
from puzzle_solvers.generate_traces.dataset import iter_puzzle_args, load_dataset
from puzzle_solvers.generate_traces.record import to_serialisable
from puzzle_solvers.generate_traces.worker import _build_solver, pool_init, process_puzzle


def _flush_batch(f, batch_records) -> None:
    for record in batch_records:
        f.write(json.dumps(to_serialisable(record)) + "\n")
    f.flush()


def _maybe_flush(f, batch_records, all_timings, total_puzzles, parallel: bool) -> None:
    generated = len(all_timings)
    if len(batch_records) < 10 and generated < total_puzzles:
        return
    _flush_batch(f, batch_records)
    batch_records.clear()
    gc.collect()
    if parallel and (generated % 50 == 0 or generated == total_puzzles):
        print(f"  [{generated}/{total_puzzles}] puzzles done...")


def validate_solvers(config: TracesConfig) -> None:
    """Dry-build every configured solver in the main process to fail fast."""
    if config.solvers is None:
        raise RuntimeError("config.solvers not set")
    for name in config.solvers:
        _build_solver(name, config)


def _print_timing_table(all_timings, total_puzzles) -> None:
    if not all_timings:
        return
    totals: defaultdict[str, float] = defaultdict(float)
    for t in all_timings:
        for name, elapsed in t.items():
            totals[name] += elapsed

    print(f"\n{'Solver':<35} {'Avg (s)':>10} {'Total (s)':>10} {'% Total':>8}")
    print("-" * 65)
    grand_total = sum(totals.values())
    for name in sorted(totals.keys(), key=lambda n: -totals[n]):
        avg = totals[name] / total_puzzles
        pct = 100 * totals[name] / grand_total if grand_total > 0 else 0
        print(f"{name:<35} {avg:>10.4f} {totals[name]:>10.2f} {pct:>7.1f}%")
    print("-" * 65)
    print(f"{'TOTAL':<35} {grand_total / total_puzzles:>10.4f} {grand_total:>10.2f}")


def run(config: TracesConfig) -> dict:
    """Generate the traces dataset described by `config`."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    start_total = time.time()
    config = config.resolve()
    assert config.data_dir is not None
    assert config.output_path is not None
    assert config.solvers is not None

    dataset = load_dataset(config)
    config.clues_shape = tuple(dataset.clues[0].shape)
    total_puzzles = len(dataset)
    print(f"Loaded {total_puzzles} puzzles")
    print(f"Solvers: {config.solvers}")

    validate_solvers(config)

    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)

    puzzle_args_gen = iter_puzzle_args(dataset)
    batch_records = []
    all_timings = []

    with open(config.output_path, "w") as f:
        if config.n_workers == 1:
            pool_init(config)
            for puzzle_data in puzzle_args_gen:
                result, timing = process_puzzle(puzzle_data)
                batch_records.append(result)
                all_timings.append(timing)
                _maybe_flush(f, batch_records, all_timings, total_puzzles, parallel=False)
        else:
            ctx = mp.get_context()
            with ctx.Pool(
                processes=config.n_workers,
                initializer=pool_init,
                initargs=(config,),
            ) as pool:
                for result, timing in pool.imap_unordered(
                    process_puzzle, puzzle_args_gen, chunksize=config.chunksize
                ):
                    batch_records.append(result)
                    all_timings.append(timing)
                    _maybe_flush(f, batch_records, all_timings, total_puzzles, parallel=True)

        if batch_records:
            _flush_batch(f, batch_records)

    generated = len(all_timings)
    elapsed = time.time() - start_total

    print("\n=== Generation completed ===")
    print(f"Workers: {config.n_workers}  chunksize: {config.chunksize}")
    print(f"Puzzles: {generated}")
    print(f"Time:    {elapsed:.2f}s")

    _print_timing_table(all_timings, generated)
    print(f"\nOutput: {config.output_path}")

    return {"generated": generated, "elapsed": elapsed, "output": config.output_path}
