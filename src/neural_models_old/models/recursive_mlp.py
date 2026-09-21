import torch
from torch import Tensor, nn

from neural_models_old.models.base import BaseModel


class RecursiveMLP(BaseModel):
    """
    A feedforward neural network with shared weights between hidden layers.
    Uses layer norm and depth embeddings for stable training.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(input_size, output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.shared_layer = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.depth_embeddings = nn.Embedding(num_layers, hidden_size)  # one vector per iteration
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        for i in range(self.num_layers - 1):
            depth_emb = self.depth_embeddings(torch.tensor(i, device=x.device))
            out = self.layer_norm(out + depth_emb)  # tell the layer which iteration it's on
            out = out + self.shared_layer(out)
            out = self.relu(out)
            out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
