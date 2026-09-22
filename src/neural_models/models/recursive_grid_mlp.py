import torch
from torch import Tensor, nn

from neural_models.models.nng_embedding import CluePositionalEmb


class GridResidualBlock(nn.Module):
    """Ported from the old style, activation swapped ReLU -> GELU to match new conventions."""

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
        self.activation = nn.GELU()
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


class RecursiveGridMLP(nn.Module):
    grid: Tensor

    def __init__(
        self,
        input_shape: torch.Size,
        output_shape: torch.Size,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        sigmoid_output: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_size = output_shape.numel()
        self.hidden_size = hidden_size
        self.combined_size = hidden_size + self.output_size
        self.num_layers = num_layers

        self.register_buffer("grid", torch.zeros(1, self.output_size))

        self.clue_embed = CluePositionalEmb(
            input_size=int(torch.prod(torch.as_tensor(input_shape))),
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.residual_block = GridResidualBlock(
            output_size=self.output_size,
            hidden_size=hidden_size,
            dropout=dropout,
            normalize_grid=True,
        )

        answer_proj = [
            nn.Linear(self.combined_size, self.combined_size),
            nn.GELU(),
            nn.Linear(self.combined_size, self.output_size),
        ]
        if sigmoid_output:
            answer_proj.append(nn.Sigmoid())
        self.answer_proj = nn.Sequential(*answer_proj)

    def _embed(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x.flatten(start_dim=1)
        context = self.clue_embed(x, mask)  # (B, hidden_size)
        batch_grid = self.grid.expand(context.size(0), -1)
        return torch.cat([batch_grid, context], dim=-1)

    def forward(self, x: Tensor, mask: Tensor, layer_num: int | None = None) -> Tensor:
        out = self._embed(x, mask)

        for _ in range(self.num_layers if layer_num is None else layer_num):
            out = self.residual_block(out)

        out = self.answer_proj(out)
        return out.view(out.size(0), *self.output_shape)

    def full_forward(
        self, x: Tensor, mask: Tensor, layer_num: int | None = None
    ) -> list[Tensor]:
        out = self._embed(x, mask)

        layers_outputs = [out]
        for _ in range(self.num_layers if layer_num is None else layer_num):
            out = self.residual_block(out)
            layers_outputs.append(out)

        return [
            self.answer_proj(layer_out).view(layer_out.size(0), *self.output_shape)
            for layer_out in layers_outputs
        ]
