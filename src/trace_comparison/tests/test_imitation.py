"""Imitation models: record loading, batch packing, GPT-2 forward/backward, generation, saving."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from old.imitation_old import (
    ImitationGPT2,
    ImitationTraceDataset,
    PuzzleRecord,
    collate_imitation,
    load_imitation,
    load_records,
    train_one_solver,
)

CLUES: Any = [[[1, 1], [1], [1]], [[3], [3], [1, 1]]]
#: Same puzzle but with a shorter clue list, so collating the two right-pads the clue block.
SHORT_CLUES: Any = [[[3], [3], [3]], [[3], [3], [3]]]


def make_record(step_counts: dict[str, int], clues=CLUES) -> dict:
    traces = {}
    for solver, num_steps in step_counts.items():
        steps = [
            np.full((3, 3), (i + 1) / 10.0, dtype=np.float32).tolist() for i in range(num_steps)
        ]
        traces[solver] = {
            "steps": steps,
            "num_steps": num_steps,
            "solved": True,
        }
    return {"clues": clues, "grid": [], "meta": {"shape": [3, 3]}, "traces": traces}


def write_records(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def make_puzzle_record(record: dict) -> PuzzleRecord:
    shape = record["meta"]["shape"]
    return PuzzleRecord(
        clues=record["clues"],
        grid_shape=(int(shape[0]), int(shape[1])),
        runs={
            solver: np.asarray(run["steps"], dtype=np.float32)
            for solver, run in record["traces"].items()
        },
    )


def small_cfg(**overrides) -> Any:
    cfg = {
        "seed": 0,
        "max_steps": 4,
        "train_fraction": 0.5,
        "batch_size": 2,
        "epochs": 1,
        "num_workers": 0,
        "pin_memory": False,
        "model": {
            "_target_": "trace_comparison.training.imitation.ImitationGPT2",
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 2,
            "dropout": 0.0,
        },
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 1.0e-3,
            "weight_decay": 0.0,
        },
        "lr_scheduler": {
            "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": 1,
        },
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def small_model(max_steps: int = 4) -> ImitationGPT2:
    return ImitationGPT2(grid_shape=(3, 3), max_steps=max_steps, n_embd=16, n_layer=1, n_head=2)


def test_load_records_filters_and_solvers(tmp_path):
    path = write_records(
        tmp_path / "traces.jsonl",
        [make_record({"a": 2, "b": 3}), make_record({"a": 1, "b": 4})],
    )

    records = load_records(path, puzzle_filter=None)
    assert list(records[0].runs) == ["a", "b"]
    assert records[0].grid_shape == (3, 3)
    assert records[0].runs["a"].shape == (2, 3, 3)
    assert records[0].runs["a"].dtype == np.float32
    assert len(records) == 2

    subset = load_records(path, solvers=["b"], max_size=1, puzzle_filter=None)
    assert len(subset) == 1
    assert list(subset[0].runs) == ["b"]

    with pytest.raises(ValueError, match="no run for solvers"):
        load_records(path, solvers=["nope"], puzzle_filter=None)
    with pytest.raises(ValueError, match="puzzle_filter"):
        load_records(path, puzzle_filter="drop_all")


def test_dataset_and_collate_pad_masks(tmp_path):
    # first puzzle has 8 clue runs, second 6 -> batch right-pads the second
    path = write_records(
        tmp_path / "traces.jsonl",
        [make_record({"a": 4}, clues=CLUES), make_record({"a": 2}, clues=SHORT_CLUES)],
    )
    records = load_records(path, puzzle_filter=None)
    dataset = ImitationTraceDataset(records, "a", max_steps=3)
    assert len(dataset) == 2

    batch = collate_imitation([dataset[0], dataset[1]])
    assert batch["cell_in"].shape == (2, 3 * 9)  # padded to longest trace (3 steps)
    assert batch["clue_runs"].shape == (2, 8)
    # shorter puzzle: 2 steps of cells, 6 clue runs, padded to the batch maxima
    assert batch["cell_mask"].tolist() == [[1] * 27, [1] * 18 + [0] * 9]
    assert batch["clue_mask"].tolist() == [[1] * 8, [1] * 6 + [0, 0]]
    # teacher forcing: first input is BOS, then every true cell shifted right
    # (cells are step-major: 9 cells of 0.1 at step 1, then 9 of 0.2, ...)
    np.testing.assert_allclose(batch["cell_in"][0, :11].numpy(), [0.5] + [0.1] * 9 + [0.2])
    np.testing.assert_allclose(batch["cell_tgt"][0, :10].numpy(), [0.1] * 9 + [0.2])


def test_forward_backward_and_loss():
    torch.manual_seed(0)
    records = [make_puzzle_record(make_record({"a": 3})), make_puzzle_record(make_record({"a": 2}))]
    dataset = ImitationTraceDataset(records, "a", max_steps=4)
    batch = collate_imitation([dataset[0], dataset[1]])

    model = small_model()
    pred = model(batch)
    assert pred.shape == batch["cell_tgt"].shape

    loss = ((pred - batch["cell_tgt"]).pow(2) * batch["cell_mask"]).sum() / batch["cell_mask"].sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad is not None


def test_key_padding_masks_clue_and_cell_pads(tmp_path):
    """The padded sample's predictions must match its unpadded forward pass."""
    path = write_records(
        tmp_path / "traces.jsonl",
        [make_record({"a": 4}, clues=CLUES), make_record({"a": 2}, clues=SHORT_CLUES)],
    )
    records = load_records(path, puzzle_filter=None)
    dataset = ImitationTraceDataset(records, "a", max_steps=4)

    torch.manual_seed(0)
    model = small_model().eval()
    padded_batch = collate_imitation([dataset[0], dataset[1]])
    alone = model(collate_imitation([dataset[1]]))
    batch = model(padded_batch)
    assert not padded_batch["cell_mask"][1].all()  # padded in both the clue and cell blocks
    n_cells = alone.shape[1]
    torch.testing.assert_close(alone[0], batch[1][:n_cells], atol=1e-5, rtol=1e-5)


def test_generate_shape_and_range():
    torch.manual_seed(0)
    model = small_model(max_steps=5)
    trace = model.generate(CLUES, temperature=0.1)
    assert trace.shape == (5, 3, 3)
    assert trace.dtype == np.float32
    assert trace.min() >= 0.0 and trace.max() <= 1.0


def test_train_one_solver_saves_numbered_checkpoints(tmp_path):
    path = write_records(
        tmp_path / "traces.jsonl",
        [make_record({"a": 3}), make_record({"a": 2}), make_record({"a": 4})],
    )
    records = load_records(path, puzzle_filter=None)
    cfg = small_cfg(max_steps=3)  # longest trace is 4 steps, so this caps it

    first = train_one_solver(cfg, records, "a", tmp_path / "checkpoints", torch.device("cpu"))
    assert first == tmp_path / "checkpoints" / "a" / "01.pt"
    payload = torch.load(first, weights_only=True)
    assert payload["solver"] == "a"
    assert payload["model_config"]["grid_shape"] == (3, 3)
    assert payload["model_config"]["max_steps"] == 3  # capped by the longest trace
    assert "val_loss" in payload["metrics"]

    second = train_one_solver(cfg, records, "a", tmp_path / "checkpoints", torch.device("cpu"))
    assert second.name == "02.pt"

    model = load_imitation(first)
    dataset = ImitationTraceDataset(records, "a", max_steps=3)
    batch = collate_imitation([dataset[0]])
    assert model(batch).shape == batch["cell_tgt"].shape
