import torch
from torch import Tensor, nn

from neural_models.models.nng_embedding import CluePositionalEmb


class ConvResidualBlock(nn.Module):
    """2-D convolutional residual block with pre-norm (GroupNorm), GELU and dropout."""

    def __init__(self, hidden_size: int, dropout: float, kernel_size: int = 3):
        super().__init__()
        self.norm = nn.GroupNorm(1, hidden_size)
        self.conv = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.norm(x)
        out = self.conv(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class RecursiveCNN(nn.Module):
    def __init__(
        self,
        input_shape: torch.Size,
        output_shape: torch.Size,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        sigmoid_output: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.grid_height = output_shape[-2]
        self.grid_width = output_shape[-1]

        self.clue_embed = CluePositionalEmb(
            input_size=int(torch.prod(torch.as_tensor(input_shape))),
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.grid_decode = nn.Linear(hidden_size, self.grid_height * self.grid_width)
        self.pre_conv = nn.Sequential(
            nn.GroupNorm(1, 1),
            nn.Conv2d(1, hidden_size, kernel_size=1),
        )
        self.residual_block = ConvResidualBlock(
            hidden_size=hidden_size, dropout=dropout
        )

        answer_proj = [
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_shape.numel()),
        ]
        if sigmoid_output:
            answer_proj.append(nn.Sigmoid())
        self.answer_proj = nn.Sequential(*answer_proj)

    def _embed(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x.flatten(start_dim=1)
        return self.clue_embed(x, mask)  # (B, hidden_size)

    def forward(self, x: Tensor, mask: Tensor, layer_num: int | None = None) -> Tensor:
        context = self._embed(x, mask)  # (B, hidden_size)

        grid = self.grid_decode(context)
        grid = grid.view(grid.size(0), 1, self.grid_height, self.grid_width)
        out = self.pre_conv(grid)  # (B, hidden_size, H, W)

        for _ in range(self.num_layers if layer_num is None else layer_num):
            out = self.residual_block(out) + out  # Residual connection

        out = out.mean(dim=(2, 3))  # (B, hidden_size)
        out = self.answer_proj(out)
        return out.view(out.size(0), *self.output_shape)
