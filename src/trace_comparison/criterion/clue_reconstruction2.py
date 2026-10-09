import math

import torch
import torch.nn.functional as F


class ClueReconstructionLoss(torch.nn.Module):
    clue_indices_template: torch.Tensor
    _step: torch.Tensor

    def __init__(
        self,
        reduction: str = "mean",
        aux_weight_start: float = 0.2,
        aux_weight_end: float = 0.5,
        entropy_weight_start: float = 0.5,
        entropy_weight_end: float = 0.2,
        anneal_steps: int = 1500,
        temperature: float = 1.0,
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        self.aux_weight_start = aux_weight_start
        self.aux_weight_end = aux_weight_end
        self.entropy_weight_start = entropy_weight_start
        self.entropy_weight_end = entropy_weight_end
        self.anneal_steps = anneal_steps
        self.temperature = temperature
        self.register_buffer("clue_indices_template", torch.tensor([], dtype=torch.float))
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long))

    def _current(self, start, end):
        t = min(float(self._step.item()) / max(1, self.anneal_steps), 1.0)
        return start + t * (end - start)

    def _grid_to_row_clues(self, grid: torch.Tensor, K: int):
        shifted = F.pad(grid[..., :-1], (1, 0), value=0.0)
        left_run_ends = grid * (1.0 - shifted)
        cum = torch.cumsum(left_run_ends, dim=-1)

        if (
            self.clue_indices_template.numel() != K
            or self.clue_indices_template.device != grid.device
        ):
            clue_idx = torch.arange(1, K + 1, dtype=grid.dtype, device=grid.device)
            self.clue_indices_template = clue_idx

        # Original triangular kernel: exactly zero at |diff|>=1, so a cell
        # can never leak weight into a neighboring slot. This guarantees
        # loss == 0 at the true binary solution (unlike a Gaussian kernel,
        # which always leaks a bit into neighbors even at perfect alignment).
        weights = torch.relu(1.0 - torch.abs(cum.unsqueeze(-1) - self.clue_indices_template))
        row_lengths = (grid.unsqueeze(-1) * weights).sum(dim=-2)
        return row_lengths, cum

    def _clue_match_percentage(self, pred_row, pred_col, row_clues, col_clues):
        row_match = (pred_row.round() == row_clues).all(dim=2).float()
        col_match = (pred_col.round() == col_clues).all(dim=2).float()
        return (row_match.mean(dim=1) + col_match.mean(dim=1)) / 2.0

    def forward(self, grid: torch.Tensor, clues: torch.Tensor, debug: bool = False):
        if self.training:
            self._step.add_(1)
        aux_weight = self._current(self.aux_weight_start, self.aux_weight_end)
        entropy_weight = self._current(self.entropy_weight_start, self.entropy_weight_end)

        probs = torch.sigmoid(grid / self.temperature)
        B, S = probs.shape
        side = int(math.sqrt(S))
        assert side * side == S, "Grid must contain a perfect square number of cells"
        probs = probs.view(B, side, side)

        N = side
        K = clues.shape[-1] // (2 * N)
        row_clues = clues[:, : N * K].view(B, N, K)
        col_clues = clues[:, N * K :].view(B, N, K)
        row_clues = row_clues.to(probs.device, non_blocking=True)
        col_clues = col_clues.to(probs.device, non_blocking=True)

        max_row_runs = row_clues.shape[-1]
        max_col_runs = col_clues.shape[-1]

        probs_t = probs.transpose(1, 2)

        pred_row, cum_row = self._grid_to_row_clues(probs, max_row_runs)
        pred_col, cum_col = self._grid_to_row_clues(probs_t, max_col_runs)

        row_err = (pred_row - row_clues.float()) ** 2
        col_err = (pred_col - col_clues.float()) ** 2
        per_sample = row_err.mean(dim=[1, 2]) + col_err.mean(dim=[1, 2])

        row_fill_err = ((probs.sum(dim=-1) - row_clues.sum(dim=-1).float()) / max(N, 1)) ** 2
        col_fill_err = ((probs_t.sum(dim=-1) - col_clues.sum(dim=-1).float()) / max(N, 1)) ** 2

        target_row_runs = (row_clues > 0).sum(dim=-1).float()
        target_col_runs = (col_clues > 0).sum(dim=-1).float()
        row_count_err = ((cum_row[..., -1] - target_row_runs) / max(K, 1)) ** 2
        col_count_err = ((cum_col[..., -1] - target_col_runs) / max(K, 1)) ** 2

        aux = (
            row_fill_err.mean(dim=-1)
            + col_fill_err.mean(dim=-1)
            + row_count_err.mean(dim=-1)
            + col_count_err.mean(dim=-1)
        )

        eps = 1e-6
        entropy = -(
            probs * torch.log(probs.clamp_min(eps))
            + (1 - probs) * torch.log((1 - probs).clamp_min(eps))
        ).mean(dim=[1, 2])

        main_term = per_sample.clone()
        aux_term = aux_weight * aux
        entropy_term = -entropy_weight * entropy
        per_sample = main_term + aux_term + entropy_term

        clue_match_pct = self._clue_match_percentage(pred_row, pred_col, row_clues, col_clues)

        if debug:
            return {
                "loss": per_sample.mean(),
                "main_term": main_term.mean().item(),
                "aux_term": aux_term.mean().item(),
                "entropy_term": entropy_term.mean().item(),
                "row_err": row_err.mean().item(),
                "col_err": col_err.mean().item(),
                "clue_match_pct": clue_match_pct.mean().item(),
                "aux_weight": aux_weight,
                "entropy_weight": entropy_weight,
                "mean_saturation": (probs - 0.5).abs().mean().item(),
            }

        if self.reduction == "none":
            return per_sample, row_err.mean(dim=[1, 2]), col_err.mean(dim=[1, 2]), clue_match_pct
        elif self.reduction == "mean":
            return per_sample.mean(), row_err.mean(), col_err.mean(), clue_match_pct.mean()
        else:
            return per_sample.sum(), row_err.sum(), col_err.sum(), clue_match_pct.mean()


def _grid_to_clues_discrete(grid_bin, K):
    B, N, _ = grid_bin.shape
    clues = torch.zeros(B, N, K)
    for b in range(B):
        for r in range(N):
            row = grid_bin[b, r].tolist()
            runs, cnt = [], 0
            for v in row:
                if v == 1:
                    cnt += 1
                else:
                    if cnt > 0:
                        runs.append(cnt)
                    cnt = 0
            if cnt > 0:
                runs.append(cnt)
            runs = runs[:K] + [0] * (K - len(runs))
            clues[b, r] = torch.tensor(runs, dtype=torch.float)
    return clues


def make_synthetic_puzzle(B, side, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    grid_bin = (torch.rand(B, side, side, generator=g) > 0.5).float()
    row_clues = _grid_to_clues_discrete(grid_bin, K)
    col_clues = _grid_to_clues_discrete(grid_bin.transpose(1, 2), K)
    clues = torch.cat([row_clues.view(B, -1), col_clues.view(B, -1)], dim=-1)
    return grid_bin, clues


def check_optimal_loss(criterion, grid_bin, clues):
    print("\n=== Check 1: loss at the true optimum ===")
    logits = (grid_bin * 2 - 1) * 12.0
    B = grid_bin.shape[0]
    logits_flat = logits.view(B, -1)
    criterion._step.fill_(criterion.anneal_steps)
    out = criterion(logits_flat, clues, debug=True)
    for k, v in out.items():
        print(f"  {k}: {v}")
    ok = out["loss"].item() < 1e-3 and out["clue_match_pct"] >= 0.999
    print(f"  -> {'PASS' if ok else 'FAIL'}: optimum loss ~0 and clue_match_pct ~1")
    criterion._step.fill_(0)
    return ok


def run_training_debug(criterion, grid_bin, clues, steps=2000, lr=0.1, log_every=100):
    print("\n=== Check 2/3: training trace (gradient + plateau) ===")
    B = grid_bin.shape[0]
    torch.manual_seed(0)
    logits = torch.randn(B, grid_bin.shape[1] * grid_bin.shape[2], requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)

    criterion.train()
    history = []
    last_loss = None
    plateau_count = 0

    for step in range(steps):
        opt.zero_grad()
        out = criterion(logits, clues, debug=True)
        loss = out["loss"]
        loss.backward()
        grad_norm = logits.grad.norm().item()
        opt.step()

        history.append(
            {**{k: v for k, v in out.items() if isinstance(v, float)}, "grad_norm": grad_norm}
        )

        if last_loss is not None and abs(loss.item() - last_loss) < 1e-6:
            plateau_count += 1
        else:
            plateau_count = 0
        last_loss = loss.item()

        if step % log_every == 0 or step == steps - 1:
            print(
                f"  step {step:5d} | loss={loss.item():.5f} "
                f"(main={out['main_term']:.5f} aux={out['aux_term']:.5f} "
                f"ent={out['entropy_term']:.5f}) "
                f"| grad_norm={grad_norm:.2e} | sat={out['mean_saturation']:.3f} "
                f"| match={out['clue_match_pct']:.3f} aux_w={out['aux_weight']:.2f}"
            )

        if plateau_count > 50 and grad_norm < 1e-5:
            print(f"  -> STOPPED at step {step}: loss flat AND grad_norm ~0 (saturation dead zone)")
            break
        if plateau_count > 50 and grad_norm >= 1e-5:
            print(
                f"  -> loss flat at step {step} but "
                f"grad_norm={grad_norm:.2e} nonzero: REAL local min"
            )
            break

    final = history[-1]
    print("\n  Final state:")
    print(f"    loss={last_loss}")
    print(f"    grad_norm={final['grad_norm']:.2e}")
    print(f"    mean_saturation={final['mean_saturation']:.3f}")
    print(f"    clue_match_pct={final['clue_match_pct']:.3f}")
    return history


if __name__ == "__main__":
    torch.manual_seed(42)
    B, side, K = 4, 6, 3
    grid_bin, clues = make_synthetic_puzzle(B, side, K, seed=1)

    criterion = ClueReconstructionLoss(
        # reduction="mean",
        # aux_weight_start=0.0,
        # aux_weight_end=0.15,
        # entropy_weight_start=0.05,
        # entropy_weight_end=0.0,
        # anneal_steps=1500,
    )

    check_optimal_loss(criterion, grid_bin, clues)
    run_training_debug(criterion, grid_bin, clues, steps=2000, lr=0.15, log_every=100)
