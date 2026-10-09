import math

import torch
import torch.nn.functional as F


class ClueReconstructionLoss(torch.nn.Module):
    """
    Loss that reconstructs nonogram clues from a soft predicted grid.

    The caller drives annealing by passing `anneal_progress` in [0, 1] to
    `forward`. At 0 the loss is soft and exploration-friendly; at 1 it is
    sharp and binarisation-heavy.
    """

    def __init__(
        self,
        aux_weight: float = 0.3,
        binarize_weight: float = 0.1,
        temperature_start: float = 2.0,
        temperature_end: float = 0.5,
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.binarize_weight = binarize_weight
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end

    # ------------------------------------------------------------------ #
    # Soft clue reconstruction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _grid_to_row_clues(grid: torch.Tensor, K: int):
        """
        grid: (B, N, N) probabilities.
        Returns (row_lengths, cum) each of shape (B, N, K) and (B, N, N).
        """
        # Start indicator: p_i * (1 - p_{i-1}), with p_{-1} = 0.
        shifted = F.pad(grid[..., :-1], (1, 0), value=0.0)
        starts = grid * (1.0 - shifted)
        cum = torch.cumsum(starts, dim=-1)  # (B, N, N)

        # Triangular kernel: exactly 1 at |diff| = 0, 0 at |diff| >= 1.
        # Cells with cum == k contribute fully to slot k.
        idx = torch.arange(1, K + 1, dtype=grid.dtype, device=grid.device)
        weights = torch.relu(1.0 - (cum.unsqueeze(-1) - idx).abs())  # (B, N, N, K)
        row_lengths = (grid.unsqueeze(-1) * weights).sum(dim=-2)     # (B, N, K)
        return row_lengths, cum

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        grid: torch.Tensor,
        clues: torch.Tensor,
        anneal_progress: float = 0.0,
        return_metrics: bool = False,
    ):
        """
        grid:  (B, S) logits, S = N*N.
        clues: (B, 2*N*K) flattened row-major then column-major clues.
        anneal_progress: float in [0, 1]; 0 = soft/exploratory, 1 = sharp.
        """
        t = float(min(max(anneal_progress, 0.0), 1.0))

        temperature = self.temperature_start + t * (self.temperature_end - self.temperature_start)
        aux_weight = self.aux_weight
        binarize_weight = self.binarize_weight * t

        probs = torch.sigmoid(grid / temperature)
        B, S = probs.shape
        side = int(math.sqrt(S))
        if side * side != S:
            raise ValueError(f"Grid size {S} is not a perfect square")
        probs = probs.view(B, side, side)
        N = side

        if clues.shape[-1] % (2 * N) != 0:
            raise ValueError(
                f"clues last dim {clues.shape[-1]} not divisible by 2*N={2 * N}"
            )
        K = clues.shape[-1] // (2 * N)
        clues = clues.to(probs.device, non_blocking=True)
        row_clues = clues[:, : N * K].view(B, N, K)
        col_clues = clues[:, N * K :].view(B, N, K)

        probs_t = probs.transpose(1, 2)

        pred_row, cum_row = self._grid_to_row_clues(probs, K)
        pred_col, cum_col = self._grid_to_row_clues(probs_t, K)

        # -------- main term: slot-wise MSE, masked to valid slots --------
        row_mask = (row_clues > 0).any(dim=1, keepdim=True).float()  # (B, 1, K)
        col_mask = (col_clues > 0).any(dim=1, keepdim=True).float()
        # Actually per-row mask: a slot is valid if any row has a clue there.
        row_slot_mask = (row_clues > 0).any(dim=1).float().unsqueeze(-1)  # (B, N, 1)
        col_slot_mask = (col_clues > 0).any(dim=1).float().unsqueeze(-1)
        # Simpler and standard: a slot is valid per (row, slot) if the target is > 0
        # OR the slot index is below the number of actual clues in that row.
        # Use the target itself as the mask source (padded entries are 0).
        row_valid = (row_clues > 0).float()
        col_valid = (col_clues > 0).float()
        # Include the first padding slot to keep the loss from ignoring "no more runs".
        row_valid = torch.clamp(row_valid + (row_clues == 0).cumsum(-1).eq(1), 0, 1)
        col_valid = torch.clamp(col_valid + (col_clues == 0).cumsum(-1).eq(1), 0, 1)

        row_err = (pred_row - row_clues) ** 2 * row_valid
        col_err = (pred_col - col_clues) ** 2 * col_valid
        denom = row_valid.sum() + col_valid.sum() + 1e-8
        main_term = (row_err.sum() + col_err.sum()) / denom

        # -------- auxiliary: fill count and run count --------
        row_fill_err = ((probs.sum(dim=-1) - row_clues.sum(dim=-1)) / max(N, 1)) ** 2
        col_fill_err = ((probs_t.sum(dim=-1) - col_clues.sum(dim=-1)) / max(N, 1)) ** 2

        target_row_runs = (row_clues > 0).sum(dim=-1).float()
        target_col_runs = (col_clues > 0).sum(dim=-1).float()
        row_count_err = ((cum_row[..., -1] - target_row_runs) / max(K, 1)) ** 2
        col_count_err = ((cum_col[..., -1] - target_col_runs) / max(K, 1)) ** 2

        aux_term = (
            row_fill_err.mean()
            + col_fill_err.mean()
            + row_count_err.mean()
            + col_count_err.mean()
        )

        # -------- binarisation: p(1-p), annealed upward --------
        binarize_term = (probs * (1.0 - probs)).mean()

        loss = main_term + aux_weight * aux_term + binarize_weight * binarize_term

        if not return_metrics:
            return loss

        # -------- metrics --------
        row_match = (pred_row.round() == row_clues).all(dim=-1).float()
        col_match = (pred_col.round() == col_clues).all(dim=-1).float()
        clue_match_pct = 0.5 * (row_match.mean() + col_match.mean())

        return loss, {
            "loss": loss.item(),
            "main_term": main_term.item(),
            "aux_term": aux_term.item(),
            "binarize_term": binarize_term.item(),
            "row_err": row_err.sum().item() / (row_valid.sum().item() + 1e-8),
            "col_err": col_err.sum().item() / (col_valid.sum().item() + 1e-8),
            "clue_match_pct": clue_match_pct.item(),
            "temperature": temperature,
            "mean_saturation": (probs - 0.5).abs().mean().item(),
        }


# ---------------------------------------------------------------------- #
# Optional: discrete clue extraction for synthetic data generation only.
# ---------------------------------------------------------------------- #
def _grid_to_clues_discrete(grid_bin: torch.Tensor, K: int) -> torch.Tensor:
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
        aux_weight_start=0.0,
        aux_weight_end=0.15,
        entropy_weight_start=0.05,
        entropy_weight_end=0.0,
        anneal_steps=1500,
    )

    check_optimal_loss(criterion, grid_bin, clues)
    run_training_debug(criterion, grid_bin, clues, steps=2000, lr=0.15, log_every=100)
