import torch
from torch import Tensor, nn

from neural_models_old.models.base import BaseModel
from neural_models_old.models.layers import CluePositionalEmbedding, GridResidualBlock


class RecursiveGridMLP(BaseModel):
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
        self.num_layers = num_layers

        self.register_buffer("grid", torch.zeros(1, output_size))

        self.clue_pos_embed = CluePositionalEmbedding(input_size)

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.residual_block = GridResidualBlock(
            output_size=output_size,
            hidden_size=hidden_size,
            dropout=dropout,
            normalize_grid=True,
        )

        self.fc = nn.Linear(self.combined_size, self.output_size)

    def forward(self, x: Tensor, layer_num: int | None = None) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected 2-D input, got {x.dim()}D")
        if x.size(-1) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {x.size(-1)}")

        x = self.clue_pos_embed(x)

        context = self.input_proj(x)
        batch_grid = self.grid.expand(x.size(0), -1)
        out = torch.cat([batch_grid, context], dim=-1)

        for _ in range(self.num_layers if layer_num is None else layer_num):
            out = self.residual_block(out)

        return self.fc(out)

    def full_forward(self, x: Tensor, layer_num: int | None = None) -> list[Tensor]:
        if x.dim() != 2:
            raise ValueError(f"Expected 2-D input, got {x.dim()}D")
        if x.size(-1) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {x.size(-1)}")

        x = self.clue_pos_embed(x)

        context = self.input_proj(x)
        batch_grid = self.grid.expand(x.size(0), -1)
        out = torch.cat([batch_grid, context], dim=-1)

        layers_outputs = [out]

        for _ in range(self.num_layers if layer_num is None else layer_num):
            out = self.residual_block(out)
            layers_outputs.append(out)

        layer_logits = [self.fc(layer_out) for layer_out in layers_outputs]
        return layer_logits
