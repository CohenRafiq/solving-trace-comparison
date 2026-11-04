from pathlib import Path
from typing import cast

import numpy as np

from trace_comparison.blocks.base import Block, DataKind
from trace_comparison.training.classifier import MLPClassifier
from trace_comparison.training.common import load_model


class ClassifierWeights(Block):
    """
    input_shape: (solvers, ...)
    output_shape: (solvers)

    Gets the final layer weights of a classifier for each solver.
    """

    input_kind = DataKind.JAGGED
    output_kind = DataKind.DENSE

    def __init__(self, checkpoint_dir: str, model_version: str) -> None:
        checkpoint_path = Path(checkpoint_dir) / "classifier" / f"{model_version}.pt"
        model, self.payload = load_model(checkpoint_path, device="cpu")
        self.model = cast(MLPClassifier, model)
        self.classes = self.payload["classes"]
        W = self.model.fc2.weight.detach()
        Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.Wn = Wn.cpu().numpy()

    def output_shape(self, input_shape: tuple[int, ...] | None = None) -> tuple[int, ...]:
        return self.Wn.shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.Wn
