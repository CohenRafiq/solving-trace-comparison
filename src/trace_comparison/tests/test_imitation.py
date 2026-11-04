"""Smoke tests for the current reversible imitation model and task."""

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from trace_comparison.data.dataset import load_trace_dataset
from trace_comparison.training.imitation import ImitationModel, ImitationTask

CLUES = [[[1], [1]], [[1], [1]]]


def make_record(step_counts: dict[str, int], grid=(2, 2), clues=CLUES) -> dict:
    traces = {}
    for solver, num_steps in step_counts.items():
        steps = [
            np.full(grid, (i + 1) / 10.0, dtype=np.float32).tolist() for i in range(num_steps)
        ]
        traces[solver] = {"steps": steps, "num_steps": num_steps, "solved": True}
    return {"clues": clues, "grid": [], "meta": {"shape": list(grid)}, "traces": traces}


def write_records(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def model_config() -> DictConfig:
    return OmegaConf.create(
        {
            "_target_": "trace_comparison.training.imitation.ImitationModel",
            "hidden_size": 8,
            "n_blocks": 1,
            "embed_dim": 2,
            "dropout": 0.0,
        }
    )


def test_imitation_model_forward_inverse_and_latent():
    torch.manual_seed(0)
    model = ImitationModel(
        in_features=4,
        out_features=12,
        n_solvers=3,
        n_blocks=1,
        hidden_size=8,
        embed_dim=2,
    )
    clues = torch.randn(5, 4)
    solver = torch.randint(0, 3, (5,))

    trace = model(clues, solver)
    assert trace.shape == (5, 12)

    clues_hat, embedding_hat = model.inverse(trace)
    assert clues_hat.shape == (5, 4)
    assert embedding_hat.shape == (5, 2)
    torch.testing.assert_close(clues_hat, clues, atol=1e-5, rtol=1e-5)

    assert model.latent(trace).shape == (5, 12)


def test_imitation_task_builds_data_model_and_loss(tmp_path):
    path = write_records(
        tmp_path / "traces.jsonl",
        [
            make_record({"a": 2, "b": 3}),
            make_record({"a": 3, "b": 2}),
            make_record({"a": 1, "b": 4}),
        ],
    )
    dataset = load_trace_dataset(path, puzzle_filter=None)
    task = ImitationTask(model_config())

    data = task.build_data(dataset, k=3, val_frac=0.34, seed=0)
    model, payload = task.build_model(data)

    clues, solver, traces = data.train
    out = model(clues, solver)
    assert out.shape == traces.flatten(1).shape
    assert payload["model_config"]["out_features"] == data.info["out_features"]

    loss, _ = task.step(model, data.train)
    assert torch.isfinite(loss)
