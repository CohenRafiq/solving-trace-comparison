from pathlib import Path
from typing import cast

import numpy as np
import torch

from trace_comparison.blocks.base import Block, DataKind
from trace_comparison.training.common import load_model
from trace_comparison.training.imitation import ImitationModel


class ImitationWeights(Block):
    """
    input_shape: (solvers, ...)
    output_shape: (solvers, out_features)

    Per-solver fingerprint of an imitation model: its output on shared reference clues,
    averaged over the references and centered across solvers.
    """

    input_kind = DataKind.JAGGED
    output_kind = DataKind.DENSE

    def __init__(self, checkpoint_dir: str, model_version: str,
                 ref_clues: torch.Tensor | None = None, center: bool = True) -> None:
        checkpoint_path = Path(checkpoint_dir) / "imitation" / f"{model_version}.pt"
        model, self.classes = load_model(checkpoint_path, device="cpu")
        self.model = cast(ImitationModel, model)
        self.model.eval()                                   # dropout off

        S = self.model.solver_embedding.num_embeddings
        if ref_clues is None:
            ref_clues = torch.zeros(1, self.model.n)        # better: a sample of real clues
        K = ref_clues.shape[0]

        with torch.no_grad():
            clues = ref_clues.repeat_interleave(S, dim=0)               # (K*S, n)
            sid = torch.arange(S).repeat(K)                             # (K*S,)
            out = self.model(clues, sid).view(K, S, -1).mean(0)         # (S, out_features)
            if center:
                # remove what all solvers share
                out = out - out.mean(dim=0, keepdim=True)
            Wn = out / out.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.Wn = Wn.cpu().numpy()

    def output_shape(self, input_shape):
        return self.Wn.shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.Wn
