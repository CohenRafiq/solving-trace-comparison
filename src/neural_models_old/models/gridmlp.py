import torch
from torch import Tensor, nn

from neural_models_old.models.base import BaseModel
from neural_models_old.models.layers import CluePositionalEmbedding, GridResidualBlock


class GridMLP(BaseModel):
    require_flat_input = True
    grid: Tensor

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__(input_size, output_size)
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.combined_size = hidden_size + output_size

        self.register_buffer("grid", torch.zeros(1, output_size))

        self.clue_pos_embed = CluePositionalEmbedding(input_size)

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                GridResidualBlock(
                    output_size=output_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    normalize_grid=False,
                )
            )

    def forward(self, x: Tensor, layer_num: int | None = None) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected 2-D input, got {x.dim()}D")
        if x.size(-1) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {x.size(-1)}")

        x = self.clue_pos_embed(x)

        context = self.input_proj(x)
        batch_grid = self.grid.expand(x.size(0), -1)
        out = torch.cat([batch_grid, context], dim=-1)

        blocks = self.blocks if layer_num is None else self.blocks[:layer_num]
        for block in blocks:
            out = block(out)

        return out[:, : self.output_size]
