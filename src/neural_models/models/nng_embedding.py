import torch
from torch import nn


class NonogramPosEmbed(nn.Module):
    def __init__(self, shape, d_model):
        super().__init__()
        _, width, max_runs = shape
        self.axis_embed = nn.Embedding(2, d_model)
        self.line_embed = nn.Embedding(width, d_model)
        self.slot_embed = nn.Embedding(max_runs, d_model)

        axis_idx = torch.arange(2).view(2, 1, 1).expand(2, width, max_runs)
        line_idx = torch.arange(width).view(1, width, 1).expand(2, width, max_runs)
        slot_idx = torch.arange(max_runs).view(1, 1, max_runs).expand(2, width, max_runs)
        self.register_buffer('axis_idx', axis_idx)
        self.register_buffer('line_idx', line_idx)
        self.register_buffer('slot_idx', slot_idx)

    def forward(self, batch_size: int) -> torch.Tensor:
        pos = (self.axis_embed(self.axis_idx)
               + self.line_embed(self.line_idx)
               + self.slot_embed(self.slot_idx))          # (2, w, k, d_model)
        pos = pos.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        return pos.flatten(start_dim=1, end_dim=3)        # (B, 2*w*k, d_model)  ← only first 3 dims


class ClueContext(nn.Module):
    """Structure-preserving flat clue encoder.

    Mirrors the architecture that reached 0.80 exact-match: an additive
    positional bias plus a per-slot Linear projection, so each of the
    ``input_size`` clue positions keeps its own weight column (unlike the
    mean-pooled NonogramClueEmbed).

    When a padding mask is supplied it is appended to the (already
    positionally-biased) clue vector before projection, making padding
    explicit without destroying the pad-slot positional signal.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.proj = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.pos_embed
        if mask is not None:
            mask = mask.flatten(start_dim=1).to(x.dtype)
            if mask.size(-1) == x.size(-1):
                x = torch.cat([x, mask], dim=-1)
            else:
                x = torch.cat([x, torch.zeros_like(x)], dim=-1)
        else:
            x = torch.cat([x, torch.zeros_like(x)], dim=-1)
        return self.proj(x)


class NonogramClueEmbed(nn.Module):
    def __init__(self, shape, d_model):
        super().__init__()
        self.pos_embed = NonogramPosEmbed(shape, d_model)
        self.clue_embed = nn.Linear(1, d_model)

    def forward(self, clues, pad_mask):
        B = clues.shape[0]
        pos_embedded = self.pos_embed(B)                                  # (B, P, d_model)
        clue_embedded = self.clue_embed(
            clues.unsqueeze(-1).float()
        ).flatten(start_dim=1, end_dim=-2)                                # (B, P, d_model)

        out = pos_embedded + clue_embedded                                # (B, P, d_model)

        mask_flat = pad_mask.flatten(start_dim=1).unsqueeze(-1)       # (B, P, 1)
        return (out * mask_flat).sum(dim=1) / mask_flat.sum(dim=1).clamp(min=1)
