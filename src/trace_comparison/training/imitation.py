from __future__ import annotations

from typing import Any

import torch
from torch import nn

from trace_comparison.training.common import Batch, FitResult
from trace_comparison.data.data import (
    clue_shape,
    encode_all_clues,
    flatten_traces,
    select,
    split_by_puzzle,
)
from trace_comparison.training.task import Task, TaskData
from neural_models.models.recursive_mlp import ResidualBlock
from torch.nn import functional as F

# --------------------------------------------------------------------- model
class ResMLP(nn.Module):
    """F / G: in -> hidden -> your ResidualBlock -> out. Needn't be invertible."""
    def __init__(self, in_dim, out_dim, hidden, dropout):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.res = ResidualBlock(hidden, dropout)          # your block, unchanged
        self.out = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # starts as identity

    def forward(self, x):
        return self.out(self.res(F.gelu(self.inp(x))))


class RevBlock(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.d1 = dim // 2
        self.F = ResMLP(dim - self.d1, self.d1, hidden, dropout)
        self.G = ResMLP(self.d1, dim - self.d1, hidden, dropout)

    def forward(self, x):
        x1, x2 = x[:, :self.d1], x[:, self.d1:]
        y1 = x1 + self.F(x2)
        y2 = x2 + self.G(y1)
        return torch.cat([y1, y2], -1)

    def inverse(self, y):
        y1, y2 = y[:, :self.d1], y[:, self.d1:]
        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(x2)
        return torch.cat([x1, x2], -1)


class ImitationModel(nn.Module):
    def __init__(self, in_features, out_features, n_solvers,
                 n_blocks=6, hidden_size=256, embed_dim=16, dropout=0.0):
        super().__init__()
        assert in_features + embed_dim <= out_features, (
            f"need out_features >= in_features + embed_dim "
            f"({out_features} < {in_features} + {embed_dim}); reduce embed_dim"
        )
        self.n, self.embed_dim = in_features, embed_dim
        self.dim = out_features                       # square bijection: no extra dims
        self.solver_embedding = nn.Embedding(n_solvers, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, in_features))
        self.hidden_size = hidden_size
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(RevBlock(self.dim, hidden_size, dropout) for _ in range(n_blocks))
        self.log_scale = nn.Parameter(torch.zeros(1, self.dim))
        self.bias = nn.Parameter(torch.zeros(1, self.dim))

    def forward(self, clues, solver_id, return_hidden=False):
        x = torch.cat([clues.flatten(1) + self.pos_embed, self.solver_embedding(solver_id)], -1)
        x = F.pad(x, (0, self.dim - x.size(-1)))
        hiddens = [x]
        for b in self.blocks:
            x = b(x)
            hiddens.append(x)
        out = x * self.log_scale.exp() + self.bias
        return (out, hiddens) if return_hidden else out

    def inverse(self, trace, return_hidden=False):
        """trace -> (clues, solver_embedding_hat), exactly. No solver_id."""
        x = (trace.flatten(1) - self.bias) * (-self.log_scale).exp()
        states = [x]
        for b in reversed(self.blocks):
            x = b.inverse(x)
            states.append(x)
        states.reverse()                              # states[k] matches forward hiddens[k]
        clues = states[0][:, :self.n] - self.pos_embed
        e_hat = states[0][:, self.n:self.n + self.embed_dim]
        return ((clues, e_hat), states) if return_hidden else (clues, e_hat)

    @torch.no_grad()
    def latent(self, trace, layer=None):
        _, states = self.inverse(trace, return_hidden=True)
        return states[len(self.blocks) // 2 if layer is None else layer]

    @torch.no_grad()
    def predict_solver(self, trace):
        _, e_hat = self.inverse(trace)
        w = F.normalize(self.solver_embedding.weight, dim=-1)
        return (F.normalize(e_hat, dim=-1) @ w.T).argmax(-1)

# ------------------------------------------------------------ step / metrics



class ImitationTask(Task):
    name = "imitation"

    @staticmethod
    def per_solver_mean(values: torch.Tensor, solver: torch.Tensor, n_solvers: int) -> torch.Tensor:
        """Mean of per-sample ``values`` within each solver -> ``(n_solvers,)``."""
        total = torch.bincount(solver, weights=values.double(), minlength=n_solvers)
        count = torch.bincount(solver, minlength=n_solvers).clamp(min=1)
        return (total / count).float()

    @staticmethod
    def mean_trace_baseline(train: Batch, val: Batch, n_solvers: int) -> torch.Tensor:
        """Per-solver val MSE of predicting that solver's mean train trace (ignores clues)."""
        _, s_tr, t_tr = train
        _, s_va, t_va = val
        mses = []
        for s in range(n_solvers):
            mean = t_tr[s_tr == s].mean(0)
            mses.append(((t_va[s_va == s] - mean) ** 2).mean())
        return torch.stack(mses)
    
    def build_data(self, ds, k, val_frac, seed) -> TaskData:
        if ds.clues is None:
            raise ValueError('dataset has no clues: every record needs a "clues" field')
        flat = flatten_traces(ds, k)
        clues = encode_all_clues(ds.clues, ds.grid_shape)  # (N, F)
        data = (clues[flat.puzzle_idx], flat.solver_idx, flat.traces)  # (M, F), (M,), (M, k, W, H)
        train_idx, val_idx = split_by_puzzle(flat.puzzle_idx, flat.n_puzzles, val_frac, seed)
        train, val = select(data, train_idx), select(data, val_idx)
        n_solvers = len(flat.solvers)
        return TaskData(
            train,
            val,
            {
                "solvers": list(flat.solvers),
                "n_solvers": n_solvers,
                "in_features": clues.shape[1],
                "out_features": flat.traces[0].numel(),
                "grid_shape": list(ds.grid_shape),
                "baseline": self.mean_trace_baseline(train, val, n_solvers),
            },
        )

    def model_kwargs(self, data: TaskData) -> dict[str, Any]:
        i = data.info
        return {"in_features": i["in_features"], "out_features": i["out_features"], "n_solvers": i["n_solvers"]}

    def step(self, model, batch):
        clues, solver, traces = batch
        err = (model(clues, solver) - traces.flatten(1)) ** 2
        return err.mean(), {"mse": err.mean(1), "solver": solver}

    def metrics(self, out):
        return {"mse": out["mse"].mean().item()}

    def describe(self, data: TaskData) -> None:
        i = data.info
        print(
            f"{len(data.train[0])} train / {len(data.val[0])} val samples, {i['n_solvers']} solvers, "
            f"clues {i['in_features']} -> trace {tuple(data.train[2].shape[1:])}"
        )
        print(f"baseline (per-solver mean trace) val mse {i['baseline'].mean():.4f}")

    def checkpoint_extras(self, result: FitResult, data: TaskData) -> dict[str, Any]:
        out, i = result.best_val_outputs, data.info
        return {
            "solvers": i["solvers"],
            "grid_shape": i["grid_shape"],
            "clue_shape": list(clue_shape(tuple(i["grid_shape"]))),
            "metrics": {
                "val_loss": result.best_val_loss,
                "val_mse_per_solver": self.per_solver_mean(out["mse"], out["solver"], i["n_solvers"]).tolist(),
                "baseline_mse_per_solver": i["baseline"].tolist(),
            },
        }

    def report(self, result: FitResult, data: TaskData) -> None:
        super().report(result, data)
        out, i = result.best_val_outputs, data.info
        per_solver = self.per_solver_mean(out["mse"], out["solver"], i["n_solvers"]).tolist()
        print("per-solver val mse (model / mean-trace baseline):")
        rows = sorted(zip(i["solvers"], per_solver, i["baseline"].tolist()), key=lambda t: -t[1] / t[2])
        for name, m, b in rows:
            print(f"  {name:<12} {m:.4f} / {b:.4f}  (ratio {m / b:.2f})")