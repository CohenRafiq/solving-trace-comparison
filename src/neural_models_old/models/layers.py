import torch
from torch import Tensor, nn


class CluePositionalEmbedding(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_embed


class GridResidualBlock(nn.Module):
    def __init__(
        self, output_size: int, hidden_size: int, dropout: float, normalize_grid: bool = False
    ):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.combined_size = output_size + hidden_size

        self.thinking_norm = nn.LayerNorm(hidden_size)
        self.grid_norm = nn.LayerNorm(output_size) if normalize_grid else None
        self.fc = nn.Linear(self.combined_size, self.combined_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        grid = x[:, : self.output_size]
        thinking = x[:, self.output_size :]

        thinking = self.thinking_norm(thinking)
        if self.grid_norm is not None:
            grid = self.grid_norm(grid)

        out = torch.cat([grid, thinking], dim=-1)
        out = self.fc(out)
        out = self.activation(out)
        out = self.dropout(out)

        return residual + out
