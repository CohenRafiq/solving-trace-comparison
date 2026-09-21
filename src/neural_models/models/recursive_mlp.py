import torch
from torch import nn

from neural_models.models.nng_embedding import NonogramClueEmbed

class RecursiveMLP(nn.Module):
    def __init__(self, input_shape: torch.Size, output_shape: torch.Size, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.clue_embed = NonogramClueEmbed(input_shape, hidden_size)
        self.core = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.answer_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_shape.numel()),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)  # Flatten the clues for embedding
        x = self.clue_embed(x, mask)
        for _ in range(self.num_layers):
            x = self.core(x)
        out = self.answer_proj(x)
        # out = out.view(out.size(0), self.output_shape[0], -1) # Reshape to (batch, categories, height * width)
        out = out.view(out.size(0), *self.output_shape)  # Reshape to (batch, height, width)
        return out