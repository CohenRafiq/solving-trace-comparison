from pathlib import Path
from typing import cast

import numpy as np
import torch

from trace_comparison.blocks.base import Block, DataKind
from trace_comparison.training.classifier import MLPClassifier
from trace_comparison.training.common import load_model
from trace_comparison.training.imitation import ImitationModel


class LatentBlock(Block):
    """
    input_shape: (solvers, n_puzzles, ...)
    output_shape: (solvers, n_puzzles, hidden_dim)

    Runs data through imitation model and returns the hidden layer embedding for each solver.
    """

    input_kind = DataKind.DENSE
    output_kind = DataKind.DENSE

    def __init__(
        self, checkpoint_dir: str, model_version: str, model_type: str, device: str
    ) -> None:
        checkpoint_path = Path(checkpoint_dir) / model_type / f"{model_version}.pt"
        model, self.payload = load_model(checkpoint_path, device=device)
        self.model = cast(MLPClassifier | ImitationModel, model)
        self.model_type = model_type
        self.hidden_dim = (
            int(self.payload["model_config"]["hidden"])
            if model_type == "classifier"
            else None
        )
        self.device = device
        self.model.eval()

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) < 2:
            raise ValueError(
                f"Input shape must have at least 2 dimensions, got {input_shape}"
            )
        if self.model_type == "classifier":
            assert self.hidden_dim is not None
            return (input_shape[0], input_shape[1], self.hidden_dim)
        if len(input_shape) < 5:
            raise ValueError(
                "Imitation latent input must be "
                f"(solvers, puzzles, steps, height, width), got {input_shape}"
            )
        return (
            input_shape[0],
            input_shape[1],
            input_shape[2] * input_shape[3] * input_shape[4],
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        torch_x = torch.from_numpy(x).to(self.device)
        torch_x = torch_x.reshape(x.shape[0] * x.shape[1], -1)
        with torch.no_grad():
            hidden = self.model.latent(torch_x)
        hidden = hidden.reshape(x.shape[0], x.shape[1], -1)
        hidden = hidden - hidden.mean(dim=0, keepdim=True)  # center across solvers
        return hidden.cpu().numpy()
