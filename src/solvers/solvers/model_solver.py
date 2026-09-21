import os
from pathlib import Path

import numpy as np
import torch

from neural_models_old.models.base import BaseModel
from neural_models_old.models.recursive_gridmlp import RecursiveGridMLP
from solvers.solvers import BaseSolver
from utils.load_model import load_model


class ModelSolver(BaseSolver):
    default_step_ratio: int = 1
    default_model_path: str = os.path.join(
        os.getenv("MODEL_DIR", "./models/"), "jsonldataset/recursivegridmlp/06.pt"
    )

    def __init__(self, model: BaseModel | str | Path | None = None):
        if model is None or isinstance(model, (str, Path)):
            path = model if model is not None else self.default_model_path
            self.model = load_model(
                RecursiveGridMLP,
                path,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                input_size=30,
                output_size=75,
                hidden_size=256,
                num_layers=9,
                dropout=0.3,
            )
        else:
            self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()

    def _logits_to_grid(self, logits: torch.Tensor, grid_shape: tuple) -> np.ndarray:
        reshaped_logits = logits.cpu().detach()[0].reshape(-1, *grid_shape)
        if reshaped_logits.ndim == 3 or reshaped_logits.shape[0] == 3:
            smooth = torch.softmax(reshaped_logits, dim=0)
            compressed = smooth[1] + 0.5 * smooth[2]
            return compressed.cpu().detach().numpy()
        else:
            return reshaped_logits.cpu().detach().numpy()

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
            torch.from_numpy(np.ascontiguousarray(clues)).flatten().unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            layer_logits = self.model.full_forward(tensor_clues, num_steps)
        list_grids = [self._logits_to_grid(logits, grid_shape) for logits in layer_logits]
        return np.array(list_grids)
