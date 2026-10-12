from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint, infer_architecture, resolve
from ..utils.const import DEVICE, empty_torch
from ...types import State

@dataclass(frozen=True)
class NNState(State):
    """
    model_class:   dotted path to the model class, e.g.
                   "neural_models.models.recursive_mlp.RecursiveMLP".
    model_path:    checkpoint file, relative to CHECKPOINT_DIR (or absolute).
    num_layers:    how many times the core block is applied per solver step. Not
                   inferrable: the block is shared, so it owns no parameters.
    dropout:       only used to rebuild the architecture; inert under eval().
    pop_of_grids:  (k, num_classes, H, W) raw logits; `grids` decodes them.
    model:         built in initialise(); shared and never replaced between steps.
    """

    model_class: str = ""
    model_path: str = ""
    num_layers: int = 1
    dropout: float = 0.0
    pop_of_grids: torch.Tensor = field(default_factory=empty_torch, compare=False)
    pop_of_hidden: torch.Tensor = field(default_factory=empty_torch, compare=False)
    model: torch.nn.Module | None = field(default=None, compare=False, repr=False)

    def _init_params(self, padded_clues: np.ndarray, grid_shape: tuple[int, int]) -> dict:
        weights = load_checkpoint(self.model_path)
        arch = infer_architecture(weights, grid_shape)
        # clues = fit_clue_width(padded_clues, arch["input_size"], grid_shape)

        model = resolve(self.model_class)(
            input_shape=torch.Size(padded_clues.shape),
            output_shape=torch.Size([arch["num_classes"], *grid_shape]),
            hidden_size=arch["hidden_size"],
            num_layers=self.num_layers,
            dropout=self.dropout,
            # Decoding is done in `grids`, so the model itself stays linear.
            sigmoid_output=False,
        )
        model.load_state_dict(weights)
        model.to(DEVICE).eval()

        x = torch.from_numpy(padded_clues).float().to(DEVICE).unsqueeze(0)
        mask = x != 0
        with torch.no_grad():
            hidden = model.embed(x, mask).repeat(self.k, 1)
            grids = model.unembed(hidden)

        return {"model": model, "pop_of_hidden": hidden, "pop_of_grids": grids}

    @property
    def grids(self) -> np.ndarray:
        logits = self.pop_of_grids
        if logits.ndim != 4:
            raise RuntimeError(
                f"expected (k, classes, H, W) logits, got shape {tuple(logits.shape)}; "
                "the state was never initialised"
            )

        probs = F.softmax(logits.detach().float(), dim=1)
        if probs.shape[1] >= 3:
            filled = probs[:, 1] / (1.0 - probs[:, -1]).clamp(min=1e-8)
        elif probs.shape[1] == 2:
            filled = probs[:, 1]
        else:
            raise ValueError(
                f"cannot read a filled-probability from {probs.shape[1]} class channels; "
                "expected two classes plus an abstain channel"
            )

        return filled.clamp(0.0, 1.0).cpu().numpy().astype(np.float64)
