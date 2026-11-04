from pathlib import Path
from typing import cast

import numpy as np
import torch

from trace_comparison.blocks.base import Block, DataKind
from trace_comparison.training.classifier import MLPClassifier
from trace_comparison.training.common import load_model


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
        model, self.payload = load_model(checkpoint_path, device=device)
        self.model = cast(MLPClassifier, model)
        self.classes = self.payload["classes"]
        self.device = device
        self.model.eval()

        self._hidden_dim = self.model.fc1.out_features
        self._in_features = self.model.fc1.in_features

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
            hidden = self.model.latent(torch_x)
        hidden = hidden.reshape(x.shape[0], x.shape[1], -1)
        return hidden.cpu().numpy()
