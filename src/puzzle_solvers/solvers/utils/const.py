import torch
import numpy as np
import os
from pathlib import Path

UNKNOWN, EMPTY, FILLED = -1, 0, 1
UNKNOWN_REPLACE = 0.5
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path(os.getenv("MODEL_DIR", "./checkpoints"))
    
def empty_int8() -> np.ndarray:
    return np.empty((0, 0, 0), dtype=np.int8)

def empty_bool() -> np.ndarray:
    return np.empty((0, 0, 0), dtype=np.bool)

def empty_torch() -> torch.Tensor:
    return torch.empty((0, 0, 0), dtype=torch.float32, device=DEVICE)