import os
from pathlib import Path

import numpy as np
import torch

UNKNOWN, EMPTY, FILLED = -1, 0, 1
UNKNOWN_REPLACE = 0.5
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: Repo root: this file is `<root>/src/puzzle_solvers/solvers/utils/const.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[4]

#: Absolute, so a checkpoint resolves the same whatever directory the process was
#: launched from. `MODEL_DIR` still wins, which is what the notebooks set.
CHECKPOINT_DIR = Path(os.getenv("MODEL_DIR") or PROJECT_ROOT / "checkpoints").resolve()

def empty_int8() -> np.ndarray:
    return np.empty((0, 0, 0), dtype=np.int8)

def empty_bool() -> np.ndarray:
    return np.empty((0, 0, 0), dtype=bool)

def empty_torch() -> torch.Tensor:
    return torch.empty((0, 0, 0), dtype=torch.float32, device=DEVICE)
