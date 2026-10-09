import torch
from torch import nn


class CluePositionalEmb(nn.Module):
    """Positional embedding for clues with masking"""

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
