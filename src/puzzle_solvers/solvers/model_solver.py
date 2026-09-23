import os

import numpy as np
import torch

from neural_models.models.recursive_grid_mlp import RecursiveGridMLP
from puzzle_solvers.solvers.base import BaseSolver

DEFAULT_MODEL_PATH = os.path.join(
    os.getenv("MODEL_DIR", "./checkpoints"), "nonogramsdataset/recursivegridmlp/default.pt"
)


def resolve_default_model_path(model_path: str | os.PathLike | None = None) -> str:
    """Resolve the model checkpoint path, raising a clear error if missing."""
    path = str(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{path}'. "
            f"Train a model and point MODEL_DIR at it, or pass --model-path."
        )
    return path


def _infer_arch(h: int, k: int, state_dict: dict) -> tuple[torch.Size, torch.Size, int]:
    """Infer (input_shape, output_shape, hidden_size) from a checkpoint state dict."""
    input_size = state_dict["clue_embed.pos_embed"].shape[-1]
    expected_input = 2 * h * k
    if input_size != expected_input:
        raise ValueError(
            f"Checkpoint was trained with clue width {input_size} (= {input_size // (2 * h)} runs "
            f"per line on a {h}x{h} grid), but the dataset provides {k} runs per line "
            f"(needs {expected_input}). Regenerate the dataset with matching run width or train "
            f"a model for the dataset's K."
        )

    output_size = state_dict["answer_proj.2.weight"].shape[0]
    cells = h * h
    if output_size % cells != 0:
        raise ValueError(
            f"Checkpoint output size {output_size} is not a multiple of grid cells {cells}; "
            f"cannot determine channel count."
        )
    channels = output_size // cells

    combined = state_dict["answer_proj.0.weight"].shape[0]
    hidden = combined - output_size
    if hidden <= 0:
        raise ValueError(
            f"Checkpoint hidden size of {hidden} is invalid (combined={combined}, "
            f"output_size={output_size})."
        )

    return torch.Size((2, h, k)), torch.Size((channels, h, h)), hidden


class ModelSolver(BaseSolver):
    def __init__(
        self,
        model=None,
        model_path: str | os.PathLike | None = None,
        input_shape: tuple[int, int, int] = (2, 5, 3),
        num_layers: int = 9,
        dropout: float = 0.3,
        **kwargs,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is not None:
            self.model = model
        else:
            path = resolve_default_model_path(model_path)
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
            h, k = input_shape[1], input_shape[2]
            input_shape_res, output_shape, hidden_size = _infer_arch(h, k, state_dict)
            self.model = RecursiveGridMLP(
                input_shape_res,
                output_shape,
                hidden_size,
                num_layers,
                dropout,
            )
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        super().__init__(**kwargs)

    def _logits_to_grid(self, logits: torch.Tensor, grid_shape: tuple) -> np.ndarray:
        x = logits.cpu().detach()[0].reshape(-1, *grid_shape)
        if x.ndim == 3 and x.shape[0] == 3:
            smooth = torch.softmax(x.to(dtype=torch.float32), dim=0)
            return (smooth[1] + 0.5 * smooth[2]).numpy()
        if x.ndim == 3 and x.shape[0] == 1:
            return x[0].numpy()
        return x.numpy()

    def _step(
        self, clues: np.ndarray, prev: np.ndarray, num_steps: int, sampling_ratio: float
    ) -> np.ndarray:
        grid_shape = prev.shape
        if sampling_ratio < 1.0:
            self.model.train()
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 1.0 - sampling_ratio
        else:
            self.model.eval()
        tensor_clues = (
            torch.from_numpy(np.ascontiguousarray(clues, dtype=np.float32))
            .unsqueeze(0)
            .to(self.device)
        )
        mask = torch.ones_like(tensor_clues)
        with torch.no_grad():
            layer_logits = self.model.full_forward(tensor_clues, mask, layer_num=num_steps)
        list_grids = [self._logits_to_grid(logits, grid_shape) for logits in layer_logits]
        list_grids = list_grids[-num_steps:]
        if not list_grids:
            return np.array([np.asarray(prev)])
        return np.array(list_grids)
