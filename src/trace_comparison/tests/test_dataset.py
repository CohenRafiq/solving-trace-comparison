"""Jagged trace loading plus the block input/output data-contract checks.

The dataset contract is a (solvers, puzzles) object array of per-trace
(steps, width, height) grids with a ragged step axis; everything downstream of
`Resample` is a dense float array, ending in a square (S, S) matrix.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from trace_comparison.blocks.resample import Resample
from trace_comparison.blocks.rmse import RMSESimilarity
from trace_comparison.dataset import TraceDataset, load_trace_dataset
from trace_comparison.gen_sim_matrix import run_experiment

DATA_DIR = Path(__file__).parents[3] / "data" / "traces_dataset" / "nonograms"
BS_TEST = DATA_DIR / "bs_test.jsonl"
LARGE_TEST = DATA_DIR / "large.jsonl"

BS_SOLVERS = (
    "backtracking_search",
    "backtracking_population",
    "backtracking_population_weighted",
    "backtracking_population_skip",
)
#: First-record order of large.jsonl, as the loader with solvers=None returns it.
LARGE_SOLVERS = (
    "ac_single",
    "ac_population",
    "ac_parallel",
    "backtracking_population",
    "backtracking_search",
    "gd_adam",
    "gd_sgd",
    "genetic_roulette",
    "genetic_tournament",
    "nn_recursive_mlp",
)

requires_bs = pytest.mark.skipif(not BS_TEST.is_file(), reason=f"missing {BS_TEST}")


def write_records(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def make_record(solvers, solved=None, num_steps=3, grid=(2, 2)):
    """A minimal jsonl record; step `s` carries the value `s` so resampling is testable."""
    solved = {} if solved is None else solved
    traces = {}
    for solver in solvers:
        steps = [np.full(grid, float(step), dtype=np.float32).tolist() for step in range(num_steps)]
        traces[solver] = {
            "steps": steps,
            "num_steps": num_steps,
            "solved": solved.get(solver, True),
        }
    return {"clues": [], "grid": [], "meta": {"shape": list(grid)}, "traces": traces}


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@requires_bs
def test_load_bs_test_keeps_jagged_steps():
    dataset = load_trace_dataset(BS_TEST)
    records = read_records(BS_TEST)
    expected = np.array(
        [[record["traces"][solver]["num_steps"] for record in records] for solver in BS_SOLVERS],
        dtype=np.int32,
    )

    assert dataset.traces.shape == (len(BS_SOLVERS), len(records))
    assert dataset.solvers == BS_SOLVERS
    assert dataset.grid_shape == (5, 5)
    assert dataset.traces.dtype == object
    assert dataset.shape == (len(BS_SOLVERS), len(records), None, 5, 5)
    np.testing.assert_array_equal(dataset.step_counts, expected)
    for cell, steps in zip(dataset.traces.flat, expected.ravel(), strict=True):
        assert cell.shape == (steps, 5, 5)
        assert cell.dtype == np.float32
    # every bs_test puzzle is solved by every solver, so all_solved keeps them all
    assert dataset.puzzle_ids.tolist() == list(range(len(records)))


def test_missing_solver_raises(tmp_path):
    path = write_records(tmp_path / "traces.jsonl", [make_record(["a", "b"])])
    with pytest.raises(
        ValueError, match="no run for solvers \\['a', 'c'\\].*available \\['a', 'b'\\]"
    ):
        load_trace_dataset(path, solvers=["a", "c"], puzzle_filter=None)


def test_solvers_none_follows_first_record_and_ignores_extras(tmp_path):
    path = write_records(
        tmp_path / "traces.jsonl",
        [make_record(["b", "a"]), make_record(["b", "a", "c"])],
    )
    dataset = load_trace_dataset(path)
    assert dataset.solvers == ("b", "a")
    assert dataset.traces.shape == (2, 2)


def test_puzzle_filters(tmp_path):
    path = write_records(
        tmp_path / "traces.jsonl",
        [
            make_record(["a", "b"], solved={"b": False}, num_steps=2),
            make_record(["a", "b"], num_steps=4),
        ],
    )
    kept = load_trace_dataset(path, puzzle_filter="all_solved")
    assert kept.traces.shape == (2, 1)
    assert kept.puzzle_ids.tolist() == [1]
    assert kept.step_counts.tolist() == [[4], [4]]

    everything = load_trace_dataset(path, puzzle_filter=None)
    assert everything.traces.shape == (2, 2)
    assert everything.step_counts.tolist() == [[2, 4], [2, 4]]

    predicate = load_trace_dataset(path, puzzle_filter=lambda runs: runs["a"]["num_steps"] == 4)
    assert predicate.puzzle_ids.tolist() == [1]

    with pytest.raises(ValueError, match="puzzle_filter"):
        load_trace_dataset(path, puzzle_filter="drop_all")


def test_no_puzzles_kept_raises(tmp_path):
    path = write_records(tmp_path / "traces.jsonl", [make_record(["a"], solved={"a": False})])
    with pytest.raises(ValueError, match="no puzzles kept"):
        load_trace_dataset(path)


def test_get_columns_subset_and_reorder(tmp_path):
    path = write_records(tmp_path / "traces.jsonl", [make_record(["a", "b", "c"])])
    dataset = load_trace_dataset(path, puzzle_filter=None)

    assert dataset.get_columns(None) is dataset
    subset = dataset.get_columns(["c", "a"])
    assert subset.solvers == ("c", "a")
    assert isinstance(subset, TraceDataset)
    np.testing.assert_array_equal(subset.step_counts, dataset.step_counts[[2, 0]])
    for kept, original in zip(subset.traces[0], dataset.traces[2], strict=True):
        np.testing.assert_array_equal(kept, original)

    with pytest.raises(ValueError, match="Unknown solvers \\['nope'\\]"):
        dataset.get_columns(["nope"])


def test_to_dense_resamples_with_nearest_neighbour(tmp_path):
    path = write_records(
        tmp_path / "traces.jsonl",
        [make_record(["a", "b"], num_steps=2), make_record(["a", "b"], num_steps=4)],
    )
    dataset = load_trace_dataset(path, puzzle_filter=None)

    dense = dataset.to_dense(k=3)
    assert dense.shape == (2, 2, 3, 2, 2)
    assert dense.dtype == np.float32
    expected_indices = np.linspace(0, 3, 3).round().astype(int)
    np.testing.assert_array_equal(dense[1, 1], dataset.traces[1, 1][expected_indices])
    short_indices = np.linspace(0, 1, 3).round().astype(int)
    np.testing.assert_array_equal(dense[0, 0], dataset.traces[0, 0][short_indices])


def test_to_dense_pads_only_to_own_max(tmp_path):
    path = write_records(
        tmp_path / "traces.jsonl",
        [make_record(["a", "b"], num_steps=2), make_record(["a", "b"], num_steps=4)],
    )
    dataset = load_trace_dataset(path, puzzle_filter=None)

    padded = dataset.to_dense(pad_value=-1.0)
    assert padded.shape == (2, 2, 4, 2, 2)
    np.testing.assert_array_equal(padded[0, 0, :2], dataset.traces[0, 0])
    assert np.all(padded[0, 0, 2:] == -1.0)
    np.testing.assert_array_equal(padded[1, 1], dataset.traces[1, 1])


def test_block_kind_checks(tmp_path):
    path = write_records(tmp_path / "traces.jsonl", [make_record(["a", "b"])])
    dataset = load_trace_dataset(path, puzzle_filter=None)

    with pytest.raises(ValueError, match="expected dense array, got a JAGGED"):
        RMSESimilarity()(dataset.traces)
    with pytest.raises(ValueError, match="expected JAGGED"):
        Resample(2)(dataset.to_dense(k=2))


def test_run_experiment_checks_block_kinds(tmp_path):
    path = write_records(tmp_path / "traces.jsonl", [make_record(["a", "b"])])
    dataset = load_trace_dataset(path, puzzle_filter=None)

    with pytest.raises(ValueError, match="expects dense input"):
        run_experiment(dataset, [RMSESimilarity()])
    with pytest.raises(ValueError, match="must end with a MATRIX block"):
        run_experiment(dataset, [Resample(2)])


@requires_bs
def test_run_experiment_end_to_end():
    dataset = load_trace_dataset(BS_TEST, max_size=8, puzzle_filter=None)
    matrix = run_experiment(dataset, [Resample(20), RMSESimilarity()])

    assert matrix.shape == (len(BS_SOLVERS), len(BS_SOLVERS))
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-6)
    np.testing.assert_allclose(np.diag(matrix), 1.0)


def test_load_large_default_config():
    if not LARGE_TEST.is_file():
        pytest.skip(f"missing {LARGE_TEST}")
    dataset = load_trace_dataset(LARGE_TEST, max_size=20)

    assert dataset.solvers == LARGE_SOLVERS
    assert dataset.traces.shape == (len(LARGE_SOLVERS), 20)
    assert dataset.grid_shape == (5, 5)
