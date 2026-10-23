"""Autoregressive imitation of solver traces.

A model takes a puzzle's clues as input and autoregressively predicts a solver's
trace ``(T, H, W)`` one *step* at a time: the input at step ``t`` is the previous
grid ``g_{t-1}`` (all ``BOS_VALUE`` for ``t = 0``) and the output is ``g_t``,
predicted as a residual on its input.

Each step is ``H + W`` tokens (one per row / column, holding that line's cell
values), so a sequence is ``clues + T * (H + W)`` tokens instead of ``T * H * W``.
Attention is block-causal: a step sees the clues, every earlier step and its own
input grid.  Training uses teacher forcing with a masked MSE over steps (variable
trace lengths are padded + masked).  One model is trained per solver run key and
saved under ``checkpoints/imitation/<solver>/NN.pt``.

Run with ``pixi run train-imitation`` (hydra overrides work as usual, e.g.
``solvers=[ac_sgl_1] epochs=5 max_size=50``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from trace_comparison.dataset import _FILTERS

#: ``[row_clues, col_clues]``; each line is a list of run lengths.
Clues = list[list[list[int]]]

#: Input grid fed at step 0, which has no predecessor ("everything unknown").
BOS_VALUE = 0.5


# --------------------------------------------------------------------------- data


@dataclass
class PuzzleRecord:
    """One puzzle's clues plus every solver's (possibly truncated) trace."""

    clues: Clues
    grid_shape: tuple[int, int]
    runs: dict[str, np.ndarray]  # solver -> float32 (T, H, W)


def load_records(
    path: str | Path,
    solvers: list[str] | None = None,
    max_size: int | None = None,
    puzzle_filter: str | None | Callable[[dict[str, dict[str, Any]]], bool] = None,
) -> list[PuzzleRecord]:
    """Read a traces jsonl into per-puzzle records (same semantics as ``load_trace_dataset``)."""
    keep = puzzle_filter if callable(puzzle_filter) else _FILTERS[puzzle_filter]
    records: list[PuzzleRecord] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in filter(str.strip, f):
            rec = json.loads(line)
            solvers = solvers or list(rec["traces"])
            runs = {s: rec["traces"][s] for s in solvers}
            if not keep(runs):
                continue
            h, w = rec["meta"]["shape"]
            steps = {s: np.asarray(run["steps"], dtype=np.float32) for s, run in runs.items()}
            records.append(PuzzleRecord(rec["clues"], (int(h), int(w)), steps))
            if len(records) == max_size:
                break
    return records


def encode_clues(clues: Clues) -> np.ndarray:
    """Flatten clue runs into ``(n_runs, 3)`` tokens: run length, line index (rows then cols), run position."""
    lines = [*clues[0], *clues[1]]
    tokens = [(run, i, j) for i, line in enumerate(lines) for j, run in enumerate(line)]
    return np.asarray(tokens, dtype=np.int64).reshape(-1, 3)


def build_tensors(
    records: list[PuzzleRecord], solver: str, max_steps: int
) -> tuple[torch.Tensor, ...]:
    """Pad everything once: ``clues (N,R,3)``, ``clue_mask (N,R)``, ``trace (N,T,H,W)``, ``step_mask (N,T)``."""
    encoded = [encode_clues(r.clues) for r in records]
    n, n_runs, (h, w) = len(records), max(len(e) for e in encoded), records[0].grid_shape
    clues = torch.zeros(n, n_runs, 3, dtype=torch.long)
    clue_mask = torch.zeros(n, n_runs, dtype=torch.bool)
    trace = torch.zeros(n, max_steps, h, w)
    step_mask = torch.zeros(n, max_steps, dtype=torch.bool)
    for i, (record, enc) in enumerate(zip(records, encoded)):
        steps = torch.from_numpy(record.runs[solver][:max_steps])
        clues[i, : len(enc)], clue_mask[i, : len(enc)] = torch.from_numpy(enc), True
        trace[i, : len(steps)], step_mask[i, : len(steps)] = steps, True
    return clues, clue_mask, trace, step_mask


# -------------------------------------------------------------------------- model


def _init_weights(module: nn.Module) -> None:
    """GPT-2 init (std 0.02) for *all* embeddings/linears, not only the backbone's."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


class Block(nn.Module):
    """Pre-LN transformer block (GPT-2 style) taking an explicit boolean attention mask."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.ln1(x)).view(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.drop(self.proj(att.transpose(1, 2).reshape(b, n, d)))
        return x + self.drop(self.mlp(self.ln2(x)))


class ImitationGPT2(nn.Module):
    """GPT-2-style transformer over ``[clue tokens | (step, line) tokens]`` with a residual grid head."""

    def __init__(
        self,
        grid_shape: tuple[int, int],
        max_steps: int,
        n_embd: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.grid_shape, self.max_steps = grid_shape, max_steps
        h, w = grid_shape
        self.run_emb = nn.Embedding(max(h, w) + 1, n_embd)
        self.run_pos_emb = nn.Embedding(max((h + 1) // 2, (w + 1) // 2), n_embd)
        # Rows then columns. Shared by clue tokens and trace line tokens, so a clue
        # and the line it constrains carry the *same* vector and attend to each other easily.
        self.line_emb = nn.Embedding(h + w, n_embd)
        self.step_emb = nn.Embedding(max_steps, n_embd)
        self.row_in, self.col_in = nn.Linear(w, n_embd), nn.Linear(h, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(n_embd, n_head, dropout) for _ in range(n_layer))
        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, w)
        self.apply(_init_weights)
        nn.init.zeros_(self.head.weight)  # start as "copy the previous grid"

    def forward(self, clues: torch.Tensor, clue_mask: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """``clues (B,R,3)``, ``clue_mask (B,R)``, input grids ``prev (B,T,H,W)`` -> predicted grids ``(B,T,H,W)``."""
        b, t, h, w = prev.shape
        n_runs, n_line, dev = clues.shape[1], h + w, prev.device
        lines = self.line_emb.weight

        clue = self.run_emb(clues[..., 0]) + self.line_emb(clues[..., 1]) + self.run_pos_emb(clues[..., 2])
        rows = self.row_in(prev) + lines[:h]  # (B,T,H,D)
        cols = self.col_in(prev.transpose(2, 3)) + lines[h:]  # (B,T,W,D)
        trace = torch.cat([rows, cols], dim=2) + self.step_emb.weight[:t, None]  # (B,T,H+W,D)
        x = self.drop(torch.cat([clue, trace.flatten(1, 2)], dim=1))

        # Block-causal mask: clues (step -1) see clues only, step s sees clues and steps <= s.
        step_id = torch.cat(
            [torch.full((n_runs,), -1, device=dev), torch.arange(t, device=dev).repeat_interleave(n_line)]
        )
        key_ok = torch.cat([clue_mask, torch.ones(b, t * n_line, dtype=torch.bool, device=dev)], dim=1)
        mask = (step_id[None] <= step_id[:, None])[None, None] & key_ok[:, None, None]

        for block in self.blocks:
            x = block(x, mask)
        rows_out = self.norm(x[:, n_runs:]).view(b, t, n_line, -1)[:, :, :h]
        return prev + self.head(rows_out)

    @torch.no_grad()
    def generate(self, clues: Clues, temperature: float = 0.0) -> np.ndarray:
        """Roll out a full ``(max_steps, H, W)`` trace, feeding each prediction back as the next input."""
        self.eval()
        dev = next(self.parameters()).device
        enc = torch.as_tensor(encode_clues(clues), device=dev)[None]
        mask = torch.ones(enc.shape[:2], dtype=torch.bool, device=dev)
        grids = torch.full((1, 1, *self.grid_shape), BOS_VALUE, device=dev)  # [BOS, p_0, ..., p_{t-1}]
        for _ in range(self.max_steps):
            nxt = self(enc, mask, grids)[:, -1:]
            nxt = (nxt + temperature * torch.randn_like(nxt)).clamp(0.0, 1.0)
            grids = torch.cat([grids, nxt], dim=1)
        return grids[0, 1:].cpu().numpy()


# ----------------------------------------------------------------------- training


def run_epoch(
    model: nn.Module,
    data: tuple[torch.Tensor, ...],
    idx: torch.Tensor,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """One pass over ``idx``. ``base`` is the MSE of the trivial "copy the previous grid" predictor."""
    training = optimizer is not None
    model.train(training)
    if training:
        idx = idx[torch.randperm(len(idx))]
    sums = torch.zeros(4, device=device)  # loss, mae, base, n_steps
    with torch.set_grad_enabled(training):
        for i in range(0, len(idx), batch_size):
            clues, clue_mask, trace, step_mask = (t[idx[i : i + batch_size]].to(device) for t in data)
            prev = torch.cat([torch.full_like(trace[:, :1], BOS_VALUE), trace[:, :-1]], dim=1)
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(clues, clue_mask, prev)
            m = step_mask.float()
            mse = (pred - trace).square().mean((2, 3))  # (B, T) per-step MSE
            loss = (mse * m).sum() / m.sum()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            mae = (pred - trace).abs().mean((2, 3))
            base = (prev - trace).square().mean((2, 3))
            sums += torch.stack([(mse * m).sum(), (mae * m).sum(), (base * m).sum(), m.sum()]).detach()
    loss, mae, base, n = sums.tolist()
    return {"loss": loss / n, "mae": mae / n, "base": base / n}


def save_checkpoint(
    model: ImitationGPT2,
    save_root: Path,
    solver: str,
    model_config: dict[str, Any],
    metrics: dict[str, float],
) -> Path:
    """Save state dict + rebuild config under ``save_root/<solver>/NN.pt``."""
    model_dir = save_root / solver
    model_dir.mkdir(parents=True, exist_ok=True)
    number = max((int(p.stem) for p in model_dir.glob("[0-9][0-9].pt")), default=0) + 1
    path = model_dir / f"{number:02d}.pt"
    payload = {
        "solver": solver,
        "model_config": model_config,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    return path


def load_imitation(path: str | Path, device: str | torch.device = "cpu") -> ImitationGPT2:
    """Rebuild a model from a checkpoint written by ``save_checkpoint``."""
    payload = torch.load(path, map_location=device, weights_only=True)
    model = ImitationGPT2(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


def train_one_solver(
    cfg: DictConfig,
    records: list[PuzzleRecord],
    solver: str,
    save_root: Path,
    device: torch.device,
) -> Path:
    """Train one solver's imitation model and save the best-val-loss checkpoint."""
    max_steps = max(record.runs[solver].shape[0] for record in records)
    if cfg.max_steps is not None:
        max_steps = min(int(cfg.max_steps), max_steps)
    data = build_tensors(records, solver, max_steps)

    n = len(records)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(cfg.seed or 0)))
    n_train = max(1, min(int(float(cfg.train_fraction) * n), n - 1))
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    grid_shape = records[0].grid_shape
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config.pop("_target_", None)
    model_config.update(grid_shape=grid_shape, max_steps=max_steps)
    model = instantiate(cfg.model, grid_shape=grid_shape, max_steps=max_steps).to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer)

    batch_size = int(cfg.batch_size)
    best, best_epoch, best_state = {"loss": float("inf")}, 0, None
    for epoch in range(1, int(cfg.epochs) + 1):
        train = run_epoch(model, data, train_idx, batch_size, device, optimizer)
        val = run_epoch(model, data, val_idx, batch_size, device)
        scheduler.step()
        print(
            f"[{solver}] epoch {epoch}/{cfg.epochs} | "
            f"train loss {train['loss']:.4f} mae {train['mae']:.4f} | "
            f"val loss {val['loss']:.4f} mae {val['mae']:.4f} (copy-prev baseline {val['base']:.4f})"
        )
        if val["loss"] < best["loss"]:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    metrics = {"val_loss": best["loss"], "val_mae": best["mae"], "val_baseline": best["base"]}
    path = save_checkpoint(model, save_root, solver, model_config, metrics)
    print(f"[{solver}] best val loss {best['loss']:.4f} (epoch {best_epoch}) -> {path}")
    return path


def _resolve(path: str | Path) -> Path:
    """Resolve a config path against the original cwd (hydra may have changed dir)."""
    try:
        base = Path(get_original_cwd())
    except ValueError:  # not running under hydra
        base = Path.cwd()
    return base / path  # an absolute ``path`` wins


@hydra.main(version_base=None, config_path="../config", config_name="imitation")
def main(cfg: DictConfig) -> None:
    """Train an imitation model for every requested solver run."""
    if cfg.seed is not None:
        torch.manual_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = _resolve(Path(cfg.data_dir) / "traces_dataset" / "nonograms" / cfg.input)
    records = load_records(data_path, cfg.solvers, cfg.max_size, cfg.puzzle_filter)
    solvers = list(cfg.solvers or records[0].runs)
    save_root = _resolve(cfg.model_save_path)
    print(f"{len(records)} puzzles, {len(solvers)} solvers, saved to {save_root}")

    for solver in solvers:
        train_one_solver(cfg, records, solver, save_root, device)


if __name__ == "__main__":
    main()