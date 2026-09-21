from torch import Tensor, nn

from neural_models_old.models.base import BaseModel


class MultiLayerPerceptron(BaseModel):
    require_flat_input = True

    def __init__(
        self, input_size: int, output_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super().__init__(input_size, output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        for layer in self.layers:
            out = out + layer(out)
            out = self.relu(out)
        out = self.fc2(out)
        return out
