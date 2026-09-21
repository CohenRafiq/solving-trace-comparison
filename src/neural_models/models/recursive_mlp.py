import torch
from torch import nn

from neural_models.models.nng_embedding import ClueContext


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

        self.clue_embed = ClueContext(
            input_size=int(torch.prod(torch.as_tensor(input_shape))),
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.core = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        answer_proj = [
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_shape.numel()),
        ]
        if sigmoid_output:
            answer_proj.append(nn.Sigmoid())
        self.answer_proj = nn.Sequential(*answer_proj)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)  # Flatten the clues for embedding
        x = self.clue_embed(x, mask)
        for _ in range(self.num_layers):
            x = self.core(x) + x  # Residual connection
        out = self.answer_proj(x)
        out = out.view(out.size(0), *self.output_shape)  # Reshape to (batch, height, width)
        return out
