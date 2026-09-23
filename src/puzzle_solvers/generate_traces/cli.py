import argparse

from puzzle_solvers.generate_traces.config import TracesConfig
from puzzle_solvers.generate_traces.runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate solving traces dataset with parallel processing"
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default=None,
        help="Data directory path (default: $DATA_DIR or ./data)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="5x5-200_000-test.jsonl",
        help="Base dataset file name (default: 5x5-200_000-test.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: data/traces_dataset/nonograms/test.jsonl)",
    )
    parser.add_argument(
        "-n",
        "--max-size",
        type=int,
        default=5,
        help="Maximum number of puzzles to process (default: 5)",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=None,
        help="Solver names to use (default: all registered solvers)",
    )
    parser.add_argument(
        "-t",
        "--max-time",
        type=float,
        default=2.0,
        help="Wall-clock time budget per puzzle per solver (default: 2.0)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count)",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=10, help="Batch size for writing (default: 10)"
    )
    parser.add_argument(
        "-c",
        "--chunksize",
        type=int,
        default=None,
        help="Chunk size for multiprocessing (default: auto)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model checkpoint path for model_solver (default: $MODEL_DIR/recursive_gridmlp.pt)",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    config = TracesConfig(
        data_dir=args.data_dir,
        input_path=args.input,
        output_path=args.output,
        max_size=args.max_size,
        solvers=args.solvers,
        max_time=args.max_time,
        n_workers=args.workers,
        batch_size=args.batch_size,
        chunksize=args.chunksize,
        model_path=args.model_path,
    )
    return run(config)
