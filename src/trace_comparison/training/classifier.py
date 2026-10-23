"""Classify which solver produced a trace.

Every (solver, puzzle) trace is resampled to ``k`` steps, giving a ``(k, W, H)``
array labelled with its solver.  A 2-layer MLP (flatten -> hidden -> classes) is
trained with cross-entropy.  The train/val split is over *puzzles* (all solvers
of a puzzle land on the same side), so val accuracy measures generalisation to
unseen puzzles.  Models are saved under ``checkpoints/classifier/NN.pt`` together
with the class names and the val confusion matrix.

Run with ``python -m trace_comparison.training.classifier`` (hydra overrides work
as usual, e.g. ``k=30 epochs=50 model.hidden=512``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset, Subset

from trace_comparison.blocks.resample import Resample
from trace_comparison.dataset import TraceDataset, load_trace_dataset

# ------------------------------------------------------------------- dataset
# (your dataset code; if it already lives in the repo, replace this block with an import)


class ClfDataset(Dataset):
    """Pairs a resampled trace `(k, W, H)` with its solver label."""

    def __init__(
        self,
        traces: np.ndarray,  # (M, k, W, H), float32
        labels: np.ndarray,  # (M,), int64
        classes: Sequence[str] | None = None,
        channel_first: bool = False,
    ) -> None:
        assert traces.ndim == 4, f"expected (M, k, W, H), got {traces.shape}"
        assert len(traces) == len(labels)
        self.traces = traces
        self.labels = labels
        self.classes = tuple(classes) if classes is not None else tuple(range(int(labels.max()) + 1))
        self.channel_first = channel_first

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        x = torch.from_numpy(self.traces[i])  # (k, W, H) = (C, W, H)
        if self.channel_first:
            x = x.permute(0, 2, 1).contiguous()  # (C, H, W)
        return x.float(), int(self.labels[i])


def build_clf_datasets(
    ds: TraceDataset,
    k: int,
    val_frac: float = 0.2,
    seed: int = 0,
    channel_first: bool = False,
) -> tuple[Dataset, Dataset]:
    """Resample every (solver, puzzle) trace to `k` steps; split across *puzzles*."""

    # (S, N, k, W, H) — the block handles ragged -> dense for you.
    dense = Resample(k).forward(ds.traces).astype(np.float32, copy=False)
    S, N = dense.shape[:2]

    # Flatten in (solver, puzzle) order so that sample index i corresponds to
    # (s, p) = (i // N, i % N).
    traces = dense.reshape(S * N, *dense.shape[2:])  # (S*N, k, W, H)
    labels = np.repeat(np.arange(S, dtype=np.int64), N)  # solver index
    puzzle = np.tile(np.arange(N, dtype=np.int64), S)  # puzzle id per sample

    # ---- split over puzzles, not samples --------------------------------------
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val_p = max(1, int(round(val_frac * N)))
    n_val_p = min(n_val_p, N - 1)  # keep >=1 train puzzle
    val_puzzles = np.sort(perm[:n_val_p])

    # Boolean mask over flattened samples: include all solvers of a kept puzzle.
    in_val = np.isin(puzzle, val_puzzles)
    train_idx = np.flatnonzero(~in_val)
    val_idx = np.flatnonzero(in_val)

    full = ClfDataset(traces, labels, ds.solvers, channel_first=channel_first)
    return (
        Subset(full, train_idx),
        Subset(full, val_idx),
    )


def to_tensors(subset: Subset) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise a subset as ``(x (M, k, W, H), y (M,))`` so batches are plain tensor slices."""
    full = subset.dataset
    return torch.from_numpy(full.traces[subset.indices]), torch.from_numpy(full.labels[subset.indices])


# --------------------------------------------------------------------- model


class MLPClassifier(nn.Module):
    """Flatten -> Linear(hidden) -> ReLU -> Linear(n_classes); returns logits."""

    def __init__(self, in_features: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------ training


def run_epoch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    n_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict[str, float], torch.Tensor]:
    """One pass over ``(x, y)``; returns ``{"loss", "acc"}`` and the ``(true, pred)`` confusion matrix."""
    training = optimizer is not None
    model.train(training)
    idx = torch.randperm(len(x)) if training else torch.arange(len(x))
    loss_sum = torch.zeros((), device=device)
    confusion = torch.zeros(n_classes**2, dtype=torch.long, device=device)
    with torch.set_grad_enabled(training):
        for i in range(0, len(idx), batch_size):
            batch = idx[i : i + batch_size]
            xb, yb = x[batch].to(device), y[batch].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += loss.detach() * len(batch)
            confusion += torch.bincount(yb * n_classes + logits.argmax(1), minlength=n_classes**2)
    confusion = confusion.view(n_classes, n_classes).cpu()
    return {"loss": loss_sum.item() / len(x), "acc": confusion.diag().sum().item() / len(x)}, confusion


def save_checkpoint(model: nn.Module, save_root: Path, extra: dict[str, Any]) -> Path:
    """Save state dict + rebuild info under ``save_root/NN.pt``."""
    save_root.mkdir(parents=True, exist_ok=True)
    number = max((int(p.stem) for p in save_root.glob("[0-9][0-9].pt")), default=0) + 1
    path = save_root / f"{number:02d}.pt"
    torch.save({**extra, "model_state_dict": model.state_dict()}, path)
    return path


def load_classifier(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[MLPClassifier, tuple[str, ...]]:
    """Rebuild ``(model, class_names)`` from a checkpoint written by ``save_checkpoint``."""
    payload = torch.load(path, map_location=device, weights_only=True)
    model = MLPClassifier(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), tuple(payload["classes"])


def _resolve(path: str | Path) -> Path:
    """Resolve a config path against the original cwd (hydra may have changed dir)."""
    try:
        base = Path(get_original_cwd())
    except ValueError:  # not running under hydra
        base = Path.cwd()
    return base / path  # an absolute ``path`` wins


@hydra.main(version_base=None, config_path="../config", config_name="classifier")
def main(cfg: DictConfig) -> None:
    """Train the solver classifier and save the best-val-loss checkpoint."""
    if cfg.seed is not None:
        torch.manual_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = _resolve(Path(cfg.data_dir) / "traces_dataset" / "nonograms" / cfg.input)
    solvers = list(cfg.solvers) if cfg.solvers is not None else None
    ds = load_trace_dataset(data_path, solvers=solvers, max_size=cfg.max_size, puzzle_filter=cfg.puzzle_filter)
    train_set, val_set = build_clf_datasets(ds, int(cfg.k), float(cfg.val_frac), int(cfg.seed or 0))
    (x_train, y_train), (x_val, y_val) = to_tensors(train_set), to_tensors(val_set)
    classes = [str(c) for c in train_set.dataset.classes]
    n_classes, in_features = len(classes), x_train[0].numel()
    print(
        f"{len(x_train)} train / {len(x_val)} val samples, {n_classes} solvers "
        f"(chance {1 / n_classes:.3f}), input {tuple(x_train.shape[1:])}"
    )

    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config.pop("_target_", None)
    model_config.update(in_features=in_features, n_classes=n_classes)
    model = instantiate(cfg.model, in_features=in_features, n_classes=n_classes).to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer)

    batch_size = int(cfg.batch_size)
    best, best_epoch, best_state, best_confusion = {"loss": float("inf")}, 0, None, None
    for epoch in range(1, int(cfg.epochs) + 1):
        train, _ = run_epoch(model, x_train, y_train, batch_size, device, n_classes, optimizer)
        val, confusion = run_epoch(model, x_val, y_val, batch_size, device, n_classes)
        scheduler.step()
        print(
            f"epoch {epoch}/{cfg.epochs} | train loss {train['loss']:.4f} acc {train['acc']:.4f} | "
            f"val loss {val['loss']:.4f} acc {val['acc']:.4f}"
        )
        if val["loss"] < best["loss"]:
            best, best_epoch, best_confusion = val, epoch, confusion
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    path = save_checkpoint(
        model,
        _resolve(cfg.model_save_path),
        {
            "model_config": model_config,
            "classes": classes,
            "k": int(cfg.k),
            "metrics": {"val_loss": best["loss"], "val_acc": best["acc"]},
            "val_confusion": best_confusion,
        },
    )
    print(f"best val loss {best['loss']:.4f} acc {best['acc']:.4f} (epoch {best_epoch}) -> {path}")

    per_class = (best_confusion.diag() / best_confusion.sum(1).clamp(min=1)).tolist()
    print("per-solver val accuracy:")
    for name, acc in sorted(zip(classes, per_class), key=lambda t: t[1]):
        print(f"  {name:<12} {acc:.3f}")


if __name__ == "__main__":
    main()