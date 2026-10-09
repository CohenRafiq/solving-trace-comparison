import torch
from torch import nn

from neural_models.models.nng_embedding import CluePositionalEmb


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x


class RecursiveMLP(nn.Module):
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

        self.clue_embed = CluePositionalEmb(
            input_size=int(torch.prod(torch.as_tensor(input_shape))),
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.core = ResidualBlock(hidden_size, dropout)
        answer_proj = [
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_shape.numel()),
        ]
        if sigmoid_output:
            answer_proj.append(nn.Sigmoid())
        self.answer_proj = nn.Sequential(*answer_proj)

    def embed(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)  # Flatten the clues for embedding
        return self.clue_embed(x, mask)

    def unembed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.answer_proj(x)
        return x.view(x.size(0), *self.output_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, layer_num: int | None = None
                ) -> torch.Tensor:
        out = self.embed(x, mask)

        for _ in range(self.num_layers if layer_num is None else layer_num):
            out = self.core(out)

        return self.unembed(out)
