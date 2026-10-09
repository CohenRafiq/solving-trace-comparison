"""Clue-only nonogram loss via exact line-automaton semantic loss.

Each row/column clue [c1..cm] describes the regular language

    0*  1^c1  0+  1^c2  0+ ... 0+  1^cm  0*

Cells are modelled as independent Bernoulli(p = sigmoid(logit)).  For one line the exact
probability that a sample satisfies its clue is computed by a forward (HMM-style)
algorithm over the automaton, in log space, straight from the logits.

    line_nll = -log P(line satisfies clue)

Properties (see test_checks.py for numerical verification):
  * exact: matches brute-force enumeration;
  * d(line_nll)/d(logit_j) = p_j - q_j, with q_j the posterior marginal given the clue is
    satisfied, so gradients do NOT vanish for confidently-wrong cells (no sigmoid dead zone);
  * loss >= 0 and == 0 iff every cell is deterministic and consistent with the clues, so no
    entropy / auxiliary / kernel / annealing terms are needed;
  * run order and run count are handled by the automaton (no slot-alignment problem).

Clue layout is the same as the original module: clues[:, :N*K] are row clues and
clues[:, N*K:] are column clues, each right-padded with zeros to K entries.
"""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1.0e9  # finite "-inf": keeps logaddexp / logsumexp gradients NaN-free


class LineAutomata(NamedTuple):
    is_one: torch.Tensor  # [L, S] bool  state emits a filled cell
    valid: torch.Tensor  # [L, S] bool  state exists for this line (rest is padding)
    start: torch.Tensor  # [L, S] bool  allowed initial states
    accept: torch.Tensor  # [L, S] bool  allowed final states


def build_line_automata(line_clues: torch.Tensor) -> LineAutomata:
    """Vectorised construction.  line_clues: [L, K], right-padded with zeros.

    State layout per line: state 0 is the leading blank; every block c contributes c "one"
    states followed by one blank state.  Blank states may self-loop, one states may not.
    Start in state 0 or 1 (leading blank optional); accept in the last two states
    (trailing blank optional).
    """
    c = line_clues.long().clamp_min(0)
    seg = (c + 1) * (c > 0)  # [L, K] states used per block
    ends = 1 + seg.cumsum(-1)  # exclusive end of each block segment
    starts = ends - seg
    n_states = 1 + seg.sum(-1)  # [L]
    smax = int(n_states.max())

    s = torch.arange(smax, device=c.device)
    s3 = s[None, :, None]
    blk = (c > 0)[:, None, :]
    in_seg = blk & (s3 >= starts[:, None, :]) & (s3 < ends[:, None, :])
    is_blank_end = blk & (s3 == ends[:, None, :] - 1)
    is_one = (in_seg & ~is_blank_end).any(-1)

    valid = s[None, :] < n_states[:, None]
    start = valid & (s[None, :] <= 1)
    accept = valid & ((s[None, :] == (n_states - 1)[:, None]) | (s[None, :] == (n_states - 2)[:, None]))
    return LineAutomata(is_one, valid, start, accept)


def line_nll(logits: torch.Tensor, auto: LineAutomata) -> torch.Tensor:
    """-log P(line satisfies its clue) for each line.  logits: [L, N] -> [L]."""
    is_one, valid, start, accept = auto
    lp1 = F.logsigmoid(logits)
    lp0 = F.logsigmoid(-logits)
    N = logits.shape[-1]
    stay_ok = valid & ~is_one

    def emit(t: int) -> torch.Tensor:
        return torch.where(is_one, lp1[:, t : t + 1], lp0[:, t : t + 1])

    alpha = emit(0).masked_fill(~start, NEG)
    for t in range(1, N):
        prev = F.pad(alpha[:, :-1], (1, 0), value=NEG)  # advance one state
        stay = alpha.masked_fill(~stay_ok, NEG)  # self-loop (blank states only)
        alpha = (torch.logaddexp(stay, prev) + emit(t)).masked_fill(~valid, NEG)
    return -torch.logsumexp(alpha.masked_fill(~accept, NEG), dim=-1)


def grid_to_clues_hard(x: torch.Tensor, K: int):
    """Exact clues of binary lines.  x: [L, N] in {0,1} -> (clues [L, K], overflow [L])."""
    x = x.float()
    prev = F.pad(x[:, :-1], (1, 0))
    run_id = ((x * (1 - prev)).cumsum(-1) * x).long()
    out = torch.zeros(x.shape[0], K + 2, device=x.device).scatter_add_(1, run_id.clamp(max=K + 1), x)
    return out[:, 1 : K + 1], out[:, K + 1] > 0


def split_clues(clues: torch.Tensor, N: int):
    B = clues.shape[0]
    K = clues.shape[-1] // (2 * N)
    return clues[:, : N * K].reshape(B, N, K), clues[:, N * K :].reshape(B, N, K), K


@torch.no_grad()
def evaluate(logits: torch.Tensor, clues: torch.Tensor) -> dict:
    """Metrics on the thresholded grid (p > 0.5), independent of any soft relaxation."""
    B = clues.shape[0]
    N = int(round((logits.numel() // B) ** 0.5))
    grid = (logits.reshape(B, N, N) > 0).float()
    rc, cc, K = split_clues(clues, N)
    r_hat, r_of = grid_to_clues_hard(grid.reshape(B * N, N), K)
    c_hat, c_of = grid_to_clues_hard(grid.transpose(1, 2).reshape(B * N, N), K)
    r_ok = ((r_hat == rc.reshape(B * N, K).float()).all(-1) & ~r_of).view(B, N)
    c_ok = ((c_hat == cc.reshape(B * N, K).float()).all(-1) & ~c_of).view(B, N)
    return {
        "solved": (r_ok.all(-1) & c_ok.all(-1)).float(),  # [B] puzzle fully satisfied
        "line_frac": (r_ok.float().mean(-1) + c_ok.float().mean(-1)) / 2,  # [B]
    }


class LineAutomatonLoss(nn.Module):
    """Semantic loss = sum (or mean) over all rows and columns of -log P(line satisfied)."""

    def __init__(self, reduction: str = "mean", line_reduction: str = "sum"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Invalid reduction: {reduction}")
        if line_reduction not in {"mean", "sum"}:
            raise ValueError(f"Invalid line_reduction: {line_reduction}")
        self.reduction = reduction
        self.line_reduction = line_reduction

    def prepare(self, clues: torch.Tensor, N: int) -> LineAutomata:
        """Optional: build once for a fixed batch and pass to forward(automata=...)."""
        rc, cc, K = split_clues(clues, N)
        return build_line_automata(torch.cat([rc, cc], dim=1).reshape(-1, K))

    def forward(self, logits: torch.Tensor, clues: torch.Tensor, automata: LineAutomata = None):
        B = clues.shape[0]
        N = int(round((logits.numel() // B) ** 0.5))
        assert N * N * B == logits.numel(), "grid must be square"
        g = logits.reshape(B, N, N)
        if automata is None:
            automata = self.prepare(clues.to(logits.device), N)
        lines = torch.cat([g, g.transpose(1, 2)], dim=1).reshape(B * 2 * N, N)
        per_line = line_nll(lines, automata).view(B, 2 * N)
        per_sample = per_line.sum(-1) if self.line_reduction == "sum" else per_line.mean(-1)
        if self.reduction == "none":
            return per_sample
        return per_sample.mean() if self.reduction == "mean" else per_sample.sum()