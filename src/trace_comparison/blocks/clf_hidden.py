import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from trace_comparison.training.common import load_model
from trace_comparison.blocks.base import Block, DataKind


class ClassifierHidden(Block):
    """
    input_shape: (solvers, n_puzzles, ...)
    output_shape: (solvers, n_puzzles, hidden_dim)

    Runs data through classifier and returns the hidden layer embedding for each solver.
    """

    input_kind = DataKind.DENSE
    output_kind = DataKind.DENSE

    def __init__(self, checkpoint_dir: str, model_version: str, device: str) -> None:
        checkpoint_path = Path(checkpoint_dir) / "classifier" / f"{model_version}.pt"
        self.model, self.classes = load_classifier(checkpoint_path, device=device)
        self.device = device
        self.model.eval()

        # Locate the last Linear layer and its position in the Sequential.
        linears = [(i, m) for i, m in enumerate(self.model.net) if isinstance(m, nn.Linear)]
        if not linears:
            raise ValueError("Classifier net contains no nn.Linear layers")
        self._last_linear_idx, self._last_linear = linears[-1]
        self._first_linear_idx, self._first_linear = linears[0]
        self._hidden_dim = self._first_linear.out_features  # now 256
        self._in_features = self._first_linear.in_features

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) < 2:
            raise ValueError(
                f"Input shape must have at least 2 dimensions, got {input_shape}"
            )
        return (input_shape[0], input_shape[1], self._hidden_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        torch_x = torch.from_numpy(x).to(self.device)
        torch_x = torch_x.reshape(x.shape[0] * x.shape[1], -1)
        if torch_x.shape[1] != self._in_features:
            raise ValueError(
                f"Input shape {tuple(torch_x.shape)} does not match classifier "
                f"input shape {self._in_features}"
            )
        with torch.no_grad():
            hidden = self.model.net[: self._last_linear_idx](torch_x)
        hidden = hidden.reshape(x.shape[0], x.shape[1], -1)
        return hidden.cpu().numpy()