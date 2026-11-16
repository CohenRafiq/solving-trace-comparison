"""Task-agnostic training engine shared by the classifier and the imitation model.

A *task* supplies three things:

* a data tuple, e.g. ``(x, y)`` or ``(clues, solver, traces)`` (tensors with a common
  first dimension; batches are plain tensor slices),
* a ``step(model, batch) -> (loss, per_sample)`` function, where ``loss`` is the mean
  batch loss and ``per_sample`` is a dict of per-sample tensors (already detached
  is fine) holding everything the task's metrics need,
* a ``metrics(per_sample) -> {name: float}`` function.

``per_sample`` outputs must be self-contained (e.g. include the targets), because in
training mode samples are visited in shuffled order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from hydra.utils import get_class, get_original_cwd
from omegaconf import DictConfig
from torch import nn

Batch = tuple[torch.Tensor, ...]
PerSample = dict[str, torch.Tensor]
StepFn = Callable[[nn.Module, Batch], tuple[torch.Tensor, PerSample]]
MetricsFn = Callable[[PerSample], dict[str, float]]


# --------------------------------------------------------------------- setup


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path against the original cwd (hydra may have changed dir)."""
    try:
        base = Path(get_original_cwd())
    except ValueError:  # not running under hydra
        base = Path.cwd()
    return base / path  # an absolute ``path`` wins


def setup_run(cfg: DictConfig) -> torch.device:
    """Seed torch (if ``cfg.seed`` is set) and pick the device."""
    if cfg.seed is not None:
        torch.manual_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ------------------------------------------------------------------ training


def run_epoch(
    model: nn.Module,
    data: Batch,
    step: StepFn,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> tuple[float, PerSample]:
    """One pass over ``data``; returns ``(mean loss, per-sample outputs)``.

    Trains if ``optimizer`` is given (shuffled order), otherwise evaluates in order.
    """
    n = len(data[0])
    training = optimizer is not None
    model.train(training)
    idx = torch.randperm(n) if training else torch.arange(n)
    loss_sum = torch.zeros((), device=device)
    chunks: dict[str, list[torch.Tensor]] = {}
    with torch.set_grad_enabled(training):
        for i in range(0, n, batch_size):
            batch_idx = idx[i : i + batch_size]
            batch = tuple(t[batch_idx].to(device) for t in data)
            loss, per_sample = step(model, batch)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            loss_sum += loss.detach() * len(batch_idx)
            for key, value in per_sample.items():
                chunks.setdefault(key, []).append(value.detach())
    outputs = {key: torch.cat(v).cpu() for key, v in chunks.items()}
    return loss_sum.item() / n, outputs


@dataclass
class FitResult:
    """Best-val-loss epoch of a ``fit`` run (model weights are *not* restored here)."""

    best_epoch: int
    best_val_loss: float
    best_val_outputs: PerSample
    best_state: dict[str, torch.Tensor]


def format_metrics(loss: float, metrics: dict[str, float]) -> str:
    return " ".join([f"loss {loss:.4f}", *(f"{k} {v:.4f}" for k, v in metrics.items())])


def fit(
    model: nn.Module,
    train_data: Batch,
    val_data: Batch,
    step: StepFn,
    metrics: MetricsFn,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epochs: int,
    batch_size: int,
    device: torch.device,
    grad_clip: float | None = 1.0,
) -> FitResult:
    """Train for ``epochs`` epochs, tracking the best validation loss."""
    best = FitResult(0, float("inf"), {}, {})
    for epoch in range(1, epochs + 1):
        train_loss, train_out = run_epoch(
            model, train_data, step, batch_size, device, optimizer, grad_clip
        )
        val_loss, val_out = run_epoch(model, val_data, step, batch_size, device)
        scheduler.step()
        print(
            f"epoch {epoch}/{epochs} | train {format_metrics(train_loss, metrics(train_out))} "
            f"| val {format_metrics(val_loss, metrics(val_out))}"
        )
        if val_loss < best.best_val_loss:
            best = FitResult(
                epoch,
                val_loss,
                val_out,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
    if not best.best_state:
        raise RuntimeError("no finite validation loss; check lr/data")
    return best


# --------------------------------------------------------------- checkpoints


def save_checkpoint(model: nn.Module, save_root: Path, extra: dict[str, Any]) -> Path:
    """Save state dict + rebuild info under ``save_root/NN.pt``."""
    save_root.mkdir(parents=True, exist_ok=True)
    number = max((int(p.stem) for p in save_root.glob("*.pt") if p.stem.isdigit()), default=0) + 1
    path = save_root / f"{number:02d}.pt"
    torch.save({**extra, "model_state_dict": model.state_dict()}, path)
    return path


def rebuild_model(payload: dict[str, Any], default_target: str | None = None) -> nn.Module:
    """Rebuild a model from a checkpoint payload using the class path stored with it."""
    target = payload.get("model_target", default_target)
    if target is None:
        raise ValueError("checkpoint has no 'model_target' and no default was given")
    model = get_class(target)(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    return model


def load_model(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a checkpoint -> ``(model in eval mode, payload)``."""
    payload = torch.load(path, map_location=device, weights_only=True)
    return rebuild_model(payload).to(device).eval(), payload

def load_latest(
    checkpoint_path: str | Path, model_type: str, device: str | torch.device = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the latest checkpoint for a model type from a directory."""
    checkpoint_dir = Path(checkpoint_path) / model_type
    
    if not checkpoint_dir.exists():
        raise ValueError(f"checkpoint dir {checkpoint_dir} does not exist")
    latest = max((p for p in checkpoint_dir.glob("*.pt") if p.stem.isdigit()), default=None)
    if latest is None:
        raise ValueError(f"no checkpoints found in {checkpoint_dir}")
    return load_model(latest, device=device)