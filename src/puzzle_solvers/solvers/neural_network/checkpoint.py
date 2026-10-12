from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from ..utils.const import CHECKPOINT_DIR, DEVICE

def resolve(path: str) -> type[torch.nn.Module]:
    module, _, name = path.rpartition(".")
    return getattr(import_module(module), name)


def resolve_checkpoint(model_path: str) -> Path:
    """The checkpoint's absolute path; a relative one is taken from CHECKPOINT_DIR."""
    candidate = Path(model_path)
    return candidate if candidate.is_absolute() else CHECKPOINT_DIR / candidate


def load_checkpoint(model_path: str) -> dict[str, torch.Tensor]:
    """Load a model checkpoint from path, returning its state dict."""
    path = resolve_checkpoint(model_path)
    if not path.is_file():
        where = f"MODEL_DIR={os.environ['MODEL_DIR']}" if "MODEL_DIR" in os.environ \
            else f"MODEL_DIR unset, using {CHECKPOINT_DIR}"
        siblings = sorted(p.name for p in path.parent.glob("*.pt")) \
            if path.parent.is_dir() else []
        detail = f"Checkpoints in {path.parent}: {', '.join(siblings)}." if siblings \
            else f"{path.parent} holds no .pt files."
        raise FileNotFoundError(f"Model checkpoint not found at {path} ({where}). {detail}")

    return torch.load(path, map_location=DEVICE, weights_only=True)


def infer_architecture(state_dict: dict, grid_shape: tuple[int, int]) -> dict:
    """Infer architecture such as input size, hidden size, etc"""
    cells = int(np.prod(grid_shape))
    hidden_size = int(state_dict["clue_embed.proj.0.weight"].shape[0])
    input_size = int(state_dict["clue_embed.pos_embed"].shape[1])
    out_features = int(state_dict["answer_proj.2.weight"].shape[0])

    if out_features % cells:
        raise ValueError(
            f"checkpoint emits {out_features} values per puzzle, which is not a whole "
            f"number of classes over {cells} cells of a {tuple(grid_shape)} grid"
        )

    return {
        "input_size": input_size,
        "hidden_size": hidden_size,
        # One output channel per cell per class: two classes plus an abstain channel.
        "num_classes": out_features // cells,
    }


# def fit_clue_width(padded_clues: np.ndarray, input_size: int, grid_shape) -> np.ndarray:
#     """Re-pad `padded_clues` to the clue width the checkpoint was trained at.

#     `NonogramsDataset` derives the clue width from the widest run it actually
#     saw, so it depends on `max_size`: two 5x5 puzzles give `(2, 5, 2)` where the
#     full file gives `(2, 5, 3)`. A checkpoint is not a function of how many
#     puzzles were loaded, so the clues are widened to the width it expects rather
#     than the run being rejected for its sample size.
#     """
#     side = int(grid_shape[0])
#     if padded_clues.shape[0] != 2 or padded_clues.shape[1] != side:
#         raise ValueError(
#             f"expected clues of shape (2, {side}, K), got {tuple(padded_clues.shape)}"
#         )

#     width, found = divmod(input_size, 2 * side)
#     if width < 1:
#         raise ValueError(
#             f"checkpoint's clue width {input_size} is too small for a {side}-cell line"
#         )

#     given = padded_clues.shape[2]
#     if given > width:
#         raise ValueError(
#             f"a line needs {given} clue slots but the checkpoint was trained with {width} "
#             f"(input size {input_size} over {side} cells): trained on a different puzzle size"
#         )

#     widened = np.zeros((2, side, width), dtype=np.float32)
#     widened[:, :, :given] = padded_clues
#     return widened
