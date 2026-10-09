from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
import numpy as np
import torch

from ...types import State
from ..utils.const import DEVICE, CHECKPOINT_DIR, empty_torch


def _resolve(path: str) -> type[torch.nn.Module]:
    module, _, name = path.rpartition(".")
    return getattr(import_module(module), name)


@dataclass(frozen=True)
class NNState(State):
    """
    model_class:   dotted path to the model class, e.g. "neural_models.models.recursive.RecursiveMLP".
    model_path:    checkpoint file, relative to CHECKPOINT_DIR.
    model_kwargs:  remaining constructor args (hidden_size, num_layers, dropout, ...).
                   input_shape and output_shape are derived from the clues and grid shape.
    pop_of_hidden: (k, hidden_size) hidden vectors the core is applied to.
    model:         built in initialise(); shared and never replaced between steps.
    """
    model_class: str = ""
    model_path: str = ""
    model_kwargs: dict = field(default_factory=dict)
    pop_of_hidden: torch.Tensor = field(default_factory=empty_torch, compare=False)
    model: torch.nn.Module | None = field(default=None, compare=False, repr=False)

    def _init_params(self, padded_clues: np.ndarray, grid_shape: tuple[int, int]) -> dict:
        model = _resolve(self.model_class)(
            input_shape=torch.Size(padded_clues.shape),
            output_shape=torch.Size(grid_shape),
            **self.model_kwargs,
        )
        model.load_state_dict(
            torch.load(CHECKPOINT_DIR / self.model_path, map_location=DEVICE, weights_only=True)
        )
        model.to(DEVICE).eval()

        x = torch.from_numpy(padded_clues).float().to(DEVICE).unsqueeze(0)
        mask = x != 0
        with torch.no_grad():
            hidden = model.embed(x, mask).repeat(self.k, 1)
            grids = model.unembed(hidden)

        return {"model": model, "pop_of_hidden": hidden, "pop_of_grids": grids}