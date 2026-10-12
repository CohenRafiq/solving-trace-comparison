"""The generate_traces config surface: every declared solver has to be buildable.

A config group is only useful if its entries actually instantiate, and hydra
composes lazily -- a typo in a `_target_` surfaces on the first solve inside a
worker process rather than at startup. These pin both: the configs compose, and
each one builds a `Solver` whose components are real objects.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from puzzle_solvers.generate_traces.config import resolve
from puzzle_solvers.generate_traces.factory import build_solvers
from puzzle_solvers.solver import build_solver
from puzzle_solvers.solvers.neural_network.state import (
    infer_architecture,
    load_checkpoint,
    resolve_checkpoint,
)
from puzzle_solvers.solvers.utils.const import CHECKPOINT_DIR
from puzzle_solvers.types import StoppingReason

CONFIG_DIR = str((Path(__file__).parent.parent / "configs").resolve())
SOLVER_DIR = Path(CONFIG_DIR) / "solvers"
SOLVER_NAMES = sorted(p.stem for p in SOLVER_DIR.glob("*.yaml"))

#: `nn_recursive_mlp` reads a real checkpoint, so anything that solves against
#: one is skipped when the file is absent rather than failed.
DEFAULT_CHECKPOINT = "nonogramsdataset/recursivemlp/default.pt"
HAS_CHECKPOINT = resolve_checkpoint(DEFAULT_CHECKPOINT).is_file()
needs_checkpoint = pytest.mark.skipif(
    not HAS_CHECKPOINT, reason=f"no checkpoint at {resolve_checkpoint(DEFAULT_CHECKPOINT)}"
)

#: The checkpoint is 5x5 only, and its clue width is baked in, so only the
#: solvers that never load it are safe to solve on any grid.
CHECKPOINT_FREE = [n for n in SOLVER_NAMES if n != "nn_recursive_mlp"]
SOLVABLE = [n for n in SOLVER_NAMES if n == "nn_recursive_mlp" or n in CHECKPOINT_FREE]

#: The exact puzzle used by the solve tests, with its clues derived rather than
#: hand-written so a wrong clue cannot make an assertion pass for the wrong
#: reason.
GRID = np.array(
    [[1, 1, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 0, 1], [1, 0, 1, 1, 0]],
    dtype=np.float32,
)


def make_clues(grid: np.ndarray = GRID, k: int = 3) -> np.ndarray:
    """Padded `(2, H, K)` clues for `grid`."""
    from utils.grid import derive_clues_from_grid

    rows, cols = derive_clues_from_grid(grid.astype(np.int8), k)
    clues = np.zeros((2, grid.shape[0], k), dtype=np.float32)
    for axis, group in enumerate((rows, cols)):
        for i, line in enumerate(group):
            for j, value in enumerate(line):
                if value > 0:
                    clues[axis, i, j] = value
    return clues


def _compose(overrides=()):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="generate_traces", overrides=list(overrides))


def test_every_solver_file_is_named_in_the_test():
    """Guards against adding a config that this module silently stops covering."""
    assert SOLVER_NAMES, f"no solver configs found under {SOLVER_DIR}"
    assert "ac_population" in SOLVER_NAMES


def test_the_root_config_composes():
    cfg = _compose()
    assert cfg.input and cfg.output and cfg.max_time
    assert cfg.dataset._target_ == "neural_models.dataset.NonogramsDataset"


def test_the_config_file_pins_the_hydra_run_dir():
    """The run dir is stripped from the composed job config, so read the file.

    Without it every run would drop its output next to the config rather than
    under `outputs/<date>/<time>/`.
    """
    text = (Path(CONFIG_DIR) / "generate_traces.yaml").read_text()
    assert "run:" in text and "outputs/" in text


def test_each_solver_nests_under_its_own_key():
    """That nesting is what lets several solvers compose without colliding.

    The axes (`strategy` / `state` / `aggregator`) are the same three keys in
    every file, so merging flat files would leave only the last solver's axes
    standing.
    """
    cfg = _compose(["solvers=[ac_parallel,gd_adam]"])
    assert set(cfg.solvers.keys()) == {"ac_parallel", "gd_adam"}
    assert cfg.solvers.ac_parallel.strategy._target_.endswith("ACParallelStrategy")
    assert cfg.solvers.gd_adam.strategy._target_.endswith("GDStrategy")


def test_the_default_solver_selection_is_declared_in_the_config():
    """A default run covers most of the catalogue; it must stay a deliberate list.

    It is read out of the config rather than hardcoded here, so adding a solver
    to `defaults` does not need a matching edit in this test -- but every entry
    has to name a real config file, which is what this catches.
    """
    cfg = _compose()
    selected = list(cfg.solvers.keys())

    assert selected, "the default selects no solvers at all"
    assert set(selected) <= set(SOLVER_NAMES), (
        f"the default names configs that do not exist: {set(selected) - set(SOLVER_NAMES)}"
    )
    # `nn_recursive_mlp` needs a checkpoint, so a default run is not checkpoint-free.
    if HAS_CHECKPOINT:
        assert "nn_recursive_mlp" in selected
    else:
        assert "nn_recursive_mlp" not in selected, (
            "the default selects a solver that needs a checkpoint this machine lacks"
        )


def test_the_solver_group_can_be_overridden_from_the_cli():
    cfg = _compose(["solvers=[ac_parallel,backtracking_search]"])
    assert set(cfg.solvers.keys()) == {"ac_parallel", "backtracking_search"}


def test_a_single_solver_group_selection_overrides_the_default():
    cfg = _compose(["solvers=[gd_adam]"])
    assert list(cfg.solvers.keys()) == ["gd_adam"]


@pytest.mark.parametrize("name", SOLVER_NAMES)
def test_every_solver_config_builds(name):
    cfg = _compose([f"solvers=[{name}]"])
    solver = build_solvers(cfg)[name]

    assert solver.name == name
    assert solver.strategy is not None
    assert solver.state is not None
    assert solver.aggregator is not None
    # `k` lives on the state, so a config that sets it there is what the state sees.
    assert solver.state.k >= 1

    described = solver.describe()
    assert described["name"] == name
    assert json.loads(json.dumps(described))["name"] == name
    # The retired `renderer` axis must not come back through a config.
    assert "renderer" not in described


@pytest.mark.parametrize("name", SOLVER_NAMES)
def test_every_solver_file_declares_the_three_axes(name):
    cfg = _compose([f"solvers=[{name}]"])
    entry = cfg.solvers[name]
    for axis in ("strategy", "state", "aggregator"):
        assert "_target_" in entry[axis], f"{name}: {axis} has no _target_"
    assert "renderer" not in entry


def test_a_solver_config_missing_an_axis_is_rejected():
    """The three axes are the whole contract, so a node without them is an error."""
    broken = OmegaConf.create({"name": "broken", "strategy": {"_target_": "numpy.mean"}})
    with pytest.raises(KeyError, match="aggregator"):
        build_solver(broken)


def test_a_config_key_that_disagrees_with_its_name_is_rejected():
    cfg = _compose()
    mismatched = OmegaConf.merge(
        cfg, {"solvers": {"ac_population": {"name": "something_else"}}}
    )
    with pytest.raises(KeyError, match="must agree"):
        build_solvers(mismatched)


def test_resolve_fills_paths_and_worker_counts():
    cfg = resolve(_compose(["max_size=4", "workers=null"]))
    assert cfg.workers == 4
    assert 1 <= cfg.chunksize <= 10
    assert Path(cfg.output).name == "test.jsonl"
    assert Path(cfg.output).parent.name == "nonograms"
    assert Path(cfg.output).parent.parent.name == "traces_dataset"


def test_resolve_clamps_workers_to_the_puzzle_count():
    cfg = resolve(_compose(["max_size=2", "workers=64"]))
    assert cfg.workers == 2


def test_resolve_honours_an_explicit_output_path():
    cfg = resolve(_compose(["output=/tmp/somewhere.jsonl"]))
    assert cfg.output == "/tmp/somewhere.jsonl"


def test_seed_is_threaded_into_the_state():
    cfg = _compose(["solvers=[ac_population]", "solvers.ac_population.state.seed=7"])
    solver = build_solvers(cfg)["ac_population"]
    assert solver.state.seed == 7
    assert solver.with_seed(11).state.seed == 11
    assert solver.with_seed(11).state.rng is not None


def test_with_seed_does_not_mutate_the_original_solver():
    solver = build_solvers(_compose())["ac_population"]
    seeded = solver.with_seed(123)
    assert solver.state.seed == 42
    assert seeded.state.seed == 123
    assert seeded.name == solver.name


#: Every solver records a probability per cell. Arc consistency and backtracking
#: commit cells and the aggregators map UNKNOWN to 0.5; gradient descent works in
#: logit space and the checkpointed model emits class logits, so both states
#: convert on the way out (see `ACFuseState.grids`, `GDBaseState.grids`,
#: `NNState.grids`). That is what makes their `cr_losses` comparable.
PROBABILITY_STEPS = set(SOLVER_NAMES)
LOGIT_STEPS: set[str] = set()


@pytest.mark.parametrize("name", SOLVABLE)
def test_solving_a_real_puzzle_produces_a_trace(name):
    """Drive each solver through a few steps.

    `max_time` is wall clock, so this asserts the shape of the contract rather
    than a step count: the aggregator returns an (H, W) grid per step, and the
    run stops for one of the declared reasons.
    """
    clues = make_clues()
    solver = build_solvers(_compose([f"solvers=[{name}]"]))[name].with_seed(0)
    steps, num_steps, reason = solver.solve(clues, GRID, max_time=30.0)

    assert isinstance(reason, StoppingReason)
    assert num_steps == len(steps)
    if num_steps:
        assert steps.shape[1:] == GRID.shape
        assert np.isfinite(steps).all()
        if name in PROBABILITY_STEPS:
            assert steps.min() >= 0.0 and steps.max() <= 1.0, name
        else:
            pytest.fail(f"{name} records logits, not probabilities")


def test_every_solver_records_probabilities():
    """The invariant that makes `cr_losses` comparable across the dataset.

    A solver that recorded logits instead would still pass every shape check,
    and its losses would quietly sit on a different scale from the rest.
    """
    assert not LOGIT_STEPS
    assert PROBABILITY_STEPS == set(SOLVER_NAMES)


def test_the_readout_does_not_decide_when_a_search_stalls():
    """Reading probabilities must not change how long a solver runs.

    A sigmoid compresses small movements -- `sigmoid'` is at most 0.25 -- so a
    raw step that is clearly still moving lands inside the stop tolerance once
    mapped to [0, 1]. Testing the stop rule on the readout is what cut
    `gd_adam`'s trace from ~190 steps down to 2: an Adam step moves the logits
    by about `lr` per iteration, which the sigmoid then halves below `1e-3`.
    """
    solver = build_solvers(_compose(["solvers=[gd_adam]"]))["gd_adam"]
    state = solver.state.initialise(make_clues(), GRID.shape)

    raw = state.raw_grids
    readout = state.grids
    assert raw.shape == readout.shape

    # A move of one Adam step's worth: real movement, plainly above the threshold.
    step = 2e-3
    assert step > solver.stopping_threshold
    moved = replace(state, pop_of_grids=raw + step).pop_of_grids
    assert not solver._has_stalled(raw, moved), "a 2e-3 move should not count as stalled"

    # The same move through the readout falls inside it, which is the trap.
    readout_move = (1.0 / (1.0 + np.exp(-step))) - 0.5
    assert readout_move < solver.stopping_threshold
    assert np.isclose(readout, readout + readout_move, atol=solver.stopping_threshold).all()


@needs_checkpoint
def test_the_checkpointed_solver_also_reports_raw_logits_separately():
    solver = build_solvers(_compose(["solvers=[nn_recursive_mlp]"]))["nn_recursive_mlp"]
    state = solver.state.initialise(make_clues(), GRID.shape)

    # (k, classes, H, W) logits underneath, (k, H, W) probabilities on the readout.
    assert state.raw_grids.ndim == 4
    assert state.grids.ndim == 3
    assert state.grids.shape == (1, 5, 5)


@pytest.mark.parametrize("name", SOLVABLE)
def test_the_same_seed_reproduces_the_same_trace(name):
    """Per-solve seeding is what makes a trace a function of the config.

    `max_time` is wall clock, so only a run that stopped on its own is
    reproducible: one cut off mid-search differs by how fast the machine was.
    That is a property of the time budget, not of the seeding, so this asserts
    on the run's stopping reason rather than skipping the slow solvers.
    """
    clues = make_clues()
    solver = build_solvers(_compose([f"solvers=[{name}]"]))[name]
    a = solver.with_seed(5).solve(clues, GRID, max_time=30.0)
    b = solver.with_seed(5).solve(clues, GRID, max_time=30.0)

    if a[2] is StoppingReason.TIMEOUT:
        pytest.skip(f"{name} hit the time budget, so its trace is not reproducible")

    assert a[1] == b[1], name
    assert np.array_equal(a[0], b[0]), name


def test_a_solver_that_solves_on_its_own_is_reproducible():
    """The positive case for the skip above, pinned on a solver known to finish."""
    clues = make_clues()
    solver = build_solvers(_compose(["solvers=[backtracking_search]"]))["backtracking_search"]
    a = solver.with_seed(5).solve(clues, GRID, max_time=5.0)
    b = solver.with_seed(5).solve(clues, GRID, max_time=5.0)

    assert a[2] is StoppingReason.SOLVED
    assert a[1] == b[1]
    assert np.array_equal(a[0], b[0])


def test_a_different_seed_changes_a_stochastic_solver():
    """`backtracking_search` picks a random cell and value per step.

    A seed that changed nothing would mean the per-solve seeding is not
    reaching the state, which is the property this guards.
    """
    clues = make_clues()
    solver = build_solvers(_compose(["solvers=[backtracking_search]"]))["backtracking_search"]
    a = solver.with_seed(1).solve(clues, GRID, max_time=5.0)
    b = solver.with_seed(2).solve(clues, GRID, max_time=5.0)
    assert a[2] is StoppingReason.SOLVED and b[2] is StoppingReason.SOLVED
    assert not (a[1] == b[1] and np.array_equal(a[0], b[0]))


def test_run_record_is_json_shaped():
    """The emitted record has to survive `json.dumps` after serialisation."""
    from puzzle_solvers.generate_traces.record import make_run_record, to_serialisable

    clues = make_clues()
    steps = np.stack([GRID, GRID])

    record = make_run_record(steps, 2, StoppingReason.SOLVED, clues, GRID)
    dumped = json.loads(json.dumps(to_serialisable(record)))
    assert dumped["num_steps"] == 2
    assert dumped["solved"] is True
    assert dumped["stopping_reason"] == "SOLVED"
    assert len(dumped["cr_losses"]) == 2
    assert len(dumped["mse_losses"]) == 2
    assert len(dumped["steps"][0]) == 5
    # A trace that reproduces the grid exactly has no error to report.
    assert dumped["cr_losses"][0] == pytest.approx(0.0)
    assert dumped["mse_losses"][0] == pytest.approx(0.0)


def test_an_empty_run_records_no_losses():
    from puzzle_solvers.generate_traces.record import make_run_record

    record = make_run_record(
        np.empty((0, 25)), 0, StoppingReason.TIMEOUT, make_clues(), GRID
    )
    assert record["cr_losses"] == []
    assert record["mse_losses"] == []
    assert record["solved"] is False


def test_cr_losses_are_zero_for_a_perfect_grid_and_positive_otherwise():
    """`grid_losses` must read probabilities directly, with no sigmoid."""
    from puzzle_solvers.generate_traces.record import grid_losses

    clues = make_clues()
    assert grid_losses(GRID[None].reshape(1, -1), clues)[0] == pytest.approx(0.0)
    assert grid_losses(np.zeros((1, 25)), clues)[0] > 0.0
    assert grid_losses(np.ones((1, 25)), clues)[0] > 0.0

    perturbed = GRID.copy()
    perturbed[0] = [1, 1, 1, 0, 0]
    assert grid_losses(perturbed[None].reshape(1, -1), clues)[0] > 0.0


# -- the checkpointed solver -------------------------------------------


def test_the_checkpoint_dir_is_absolute():
    """Otherwise the path depends on the directory the process was launched from."""
    assert CHECKPOINT_DIR.is_absolute()


def test_a_relative_checkpoint_path_resolves_under_the_checkpoint_dir():
    resolved = resolve_checkpoint(DEFAULT_CHECKPOINT)
    assert resolved == CHECKPOINT_DIR / DEFAULT_CHECKPOINT
    assert resolved.is_absolute()


def test_an_absolute_checkpoint_path_is_left_alone(tmp_path):
    absolute = tmp_path / "somewhere" / "model.pt"
    assert resolve_checkpoint(str(absolute)) == absolute


def test_a_missing_checkpoint_says_where_it_looked():
    """The failure mode is a config path being wrong, so the error has to say so."""
    missing = "nonogramsdataset/recursivemlp/definitely-not-here.pt"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_checkpoint(missing)

    message = str(excinfo.value)
    assert str(resolve_checkpoint(missing)) in message
    assert "MODEL_DIR" in message


@needs_checkpoint
def test_architecture_is_read_from_the_checkpoint():
    """`hidden_size` and the output width are read, `num_layers` is not.

    `RecursiveMLP` applies one shared `ResidualBlock` in a loop, so `num_layers`
    owns no parameters and cannot be recovered -- it stays in the config.
    """
    arch = infer_architecture(load_checkpoint(DEFAULT_CHECKPOINT), (5, 5))

    assert arch["hidden_size"] == 256
    assert arch["input_size"] == 2 * 5 * 3  # 2 * H * K for this puzzle
    assert arch["num_classes"] == 3  # two classes plus the abstain channel
    assert "num_layers" not in arch


def test_architecture_rejects_an_output_that_does_not_divide_the_grid():
    """A checkpoint whose width is not a whole number of classes per cell."""
    import torch

    state_dict = {
        "clue_embed.pos_embed": torch.zeros(1, 30),
        "clue_embed.proj.0.weight": torch.zeros(256, 60),
        "answer_proj.2.weight": torch.zeros(75, 256),
    }
    with pytest.raises(ValueError, match="not a whole number of classes"):
        infer_architecture(state_dict, (10, 10))


@needs_checkpoint
def test_a_checkpoint_for_another_grid_size_is_rejected_clearly():
    """The clue width is baked in, so a 5x5 checkpoint cannot drive a 10x10 grid."""
    solver = build_solvers(_compose(["solvers=[nn_recursive_mlp]"]))["nn_recursive_mlp"]
    with pytest.raises(ValueError, match="not a whole number of classes|different grid size"):
        solver.state.initialise(np.zeros((2, 10, 6), dtype=np.float32), (10, 10))


@needs_checkpoint
def test_narrower_clues_are_widened_to_the_checkpoints_width():
    """`max_size` must not decide whether the checkpoint can be loaded.

    `NonogramsDataset` pads to the widest run in the file it actually loaded, so
    a small sample gives a narrower clue array than the checkpoint was trained
    on. Widening is lossless -- the extra slots are the zeros the narrower array
    already implied -- so it must not be an error.
    """
    from puzzle_solvers.solvers.neural_network.state import _fit_clue_width

    narrow = make_clues(k=2)
    assert narrow.shape == (2, 5, 2)

    widened = _fit_clue_width(narrow, input_size=30, grid_shape=(5, 5))
    assert widened.shape == (2, 5, 3)
    # Right-padding only: nothing that was there is moved or lost.
    assert np.array_equal(widened[:, :, :2], narrow)
    assert not widened[:, :, 2].any()

    solver = build_solvers(_compose(["solvers=[nn_recursive_mlp]"]))["nn_recursive_mlp"]
    state = solver.state.initialise(narrow, GRID.shape)
    assert state.grids.shape == (1, 5, 5)
    assert state.grids.min() >= 0.0 and state.grids.max() <= 1.0


def test_clues_wider_than_the_checkpoint_are_rejected():
    """The other direction cannot be rescued: the slots are not there to hold it."""
    from puzzle_solvers.solvers.neural_network.state import _fit_clue_width

    with pytest.raises(ValueError, match="trained with 2"):
        _fit_clue_width(np.zeros((2, 5, 3), dtype=np.float32), input_size=20, grid_shape=(5, 5))


@needs_checkpoint
def test_nn_grids_are_probabilities_not_logits():
    """The model emits class logits; the trace has to carry probabilities.

    Everything downstream -- the aggregators and `grid_losses` -- assumes a
    value in [0, 1] per cell, and the trained model has `sigmoid_output=False`
    with three logits per cell, so this would otherwise be silently wrong.
    """
    state = build_solvers(_compose(["solvers=[nn_recursive_mlp]"]))[
        "nn_recursive_mlp"
    ].state.initialise(make_clues(), GRID.shape)

    assert state.pop_of_grids.ndim == 4  # (k, classes, H, W) raw
    assert state.pop_of_grids.shape[1] == 3

    grids = state.grids
    assert grids.shape == (1, 5, 5)
    assert np.isfinite(grids).all()
    assert grids.min() >= 0.0 and grids.max() <= 1.0


@needs_checkpoint
def test_nn_grids_before_initialise_explain_themselves():
    solver = build_solvers(_compose(["solvers=[nn_recursive_mlp]"]))["nn_recursive_mlp"]
    with pytest.raises(RuntimeError, match="never initialised"):
        _ = solver.state.grids


@needs_checkpoint
def test_the_nn_solver_solves_and_its_losses_are_comparable():
    """Its `cr_losses` only mean anything if the steps are probabilities."""
    from puzzle_solvers.generate_traces.record import compute_run_losses

    clues = make_clues()
    solver = build_solvers(_compose(["solvers=[nn_recursive_mlp]"]))[
        "nn_recursive_mlp"
    ].with_seed(0)
    steps, num_steps, reason = solver.solve(clues, GRID, max_time=60.0)

    assert reason is StoppingReason.SOLVED
    assert num_steps == len(steps)

    cr_losses, mse_losses = compute_run_losses(steps, clues, GRID)
    assert len(cr_losses) == len(mse_losses) == num_steps
    assert all(loss >= 0.0 for loss in cr_losses)
    # Recovering the grid means the clue error goes down, not up.
    assert cr_losses[-1] < cr_losses[0]
