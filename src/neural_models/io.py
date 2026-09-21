import re
from pathlib import Path

import torch


def get_next_model_number(model_dir: Path) -> int:
    """Find the next available model number in directory (01, 02, etc.)"""
    model_dir.mkdir(parents=True, exist_ok=True)

    existing_numbers = []
    for file in model_dir.glob("*.pt"):
        match = re.match(r"(\d{2})\.pt$", file.name)
        if match:
            existing_numbers.append(int(match.group(1)))

    return max(existing_numbers, default=0) + 1


def save_model(cfg, model, logger):
    dataset_name = cfg.train_dataset._target_.split(".")[-1].lower()
    model_name = cfg.model._target_.split(".")[-1].lower()

    model_dir = Path("models") / dataset_name / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    next_num = get_next_model_number(model_dir)
    model_path = model_dir / f"{next_num:02d}.pt"
    torch.save(model.state_dict(), model_path)
    logger.log_model(model_path, name="nonogram-solver")