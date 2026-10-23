import numpy as np
import torch
from pathlib import Path

from trace_comparison.training.classifier import load_classifier
from trace_comparison.blocks.base import Block, DataKind


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
        self.model, self.classes = load_classifier(checkpoint_path, device="cpu")
        self.payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        W = self.model.net[-1].weight.detach()
        Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.Wn = Wn.cpu().numpy()


    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return self.Wn.shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.Wn