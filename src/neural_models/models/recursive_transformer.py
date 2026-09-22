import torch
from torch import Tensor, nn


def _num_tokens(input_shape: torch.Size) -> int:
    """Number of clue-line tokens = product of all dims except the last."""
    return int(torch.prod(torch.as_tensor(input_shape[:-1])))


class RecursiveTransformer(nn.Module):
    def __init__(
        self,
        input_shape: torch.Size,
        output_shape: torch.Size,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_heads: int = 4,
        ff_dim: int | None = None,
        sigmoid_output: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_tokens = _num_tokens(input_shape)
        self.token_len = int(input_shape[-1])
        ff_dim = ff_dim or hidden_size * 4

        self.token_embed = nn.Sequential(
            nn.Linear(self.token_len * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_size)

        answer_proj = [
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_shape.numel()),
        ]
        if sigmoid_output:
            answer_proj.append(nn.Sigmoid())
        self.answer_proj = nn.Sequential(*answer_proj)

    def _embed(self, x: Tensor, mask: Tensor) -> Tensor:
        batch = x.size(0)
        x = x.view(batch, self.num_tokens, self.token_len).to(mask.dtype)
        mask = mask.view(batch, self.num_tokens, self.token_len).to(x.dtype)
        tokens = torch.cat([x, mask], dim=-1)  # values + mask, (B, T, token_len*2)
        hidden = self.token_embed(tokens) + self.pos_embed  # (B, T, hidden_size)
        return hidden

    def forward(self, x: Tensor, mask: Tensor, layer_num: int | None = None) -> Tensor:
        hidden = self._embed(x, mask)

        for i in range(self.num_layers if layer_num is None else layer_num):
            hidden = self.layers[i](hidden)
        hidden = self.norm(hidden)
        hidden = hidden.mean(dim=1)  # (B, hidden_size)

        out = self.answer_proj(hidden)
        return out.view(out.size(0), *self.output_shape)
