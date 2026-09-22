import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from omegaconf import DictConfig
from neural_models.logger import Logger

import torch
from torch import nn

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover
    OmegaConf = None


def get_next_model_number(model_dir: Path) -> int:
    """Find the next available model number in directory (01, 02, etc.)"""
    model_dir.mkdir(parents=True, exist_ok=True)

    existing_numbers = []
    for file in model_dir.glob("*.pt"):
        match = re.match(r"(\d{2})\.pt$", file.name)
        if match:
            existing_numbers.append(int(match.group(1)))

    return max(existing_numbers, default=0) + 1


def _index_path(model_save_root: Path) -> Path:
    return model_save_root / "saved_models.json"


def _load_index(index_path: Path) -> list[dict]:
    if index_path.exists():
        return json.loads(index_path.read_text()).get("entries", [])
    return []


def _config_digest(cfg: DictConfig) -> str:
    try:
        if OmegaConf is not None:
            payload = json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True)
        else:
            payload = str(cfg)
    except Exception:
        payload = str(cfg)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_dir() -> str | None:
    try:
        from hydra.core.hydra_config import HydraConfig

        return HydraConfig.get().run.dir
    except Exception:
        return None


def save_model(cfg: DictConfig, model: nn.Module, logger: Logger, metrics: dict | None = None):
    """Save the model state dict and record a metadata entry in the index file."""
    dataset_name = cfg.train_dataset._target_.split(".")[-1].lower()
    model_name = cfg.model._target_.split(".")[-1].lower()

    model_save_root = Path(getattr(cfg, "model_save_path", "checkpoints/"))
    model_dir = model_save_root / dataset_name / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    next_num = get_next_model_number(model_dir)
    model_path = model_dir / f"{next_num:02d}.pt"
    torch.save(model.state_dict(), model_path)
    logger.log_model(model_path, name="nonogram-solver")

    entry = {
        "model": model_name,
        "dataset": dataset_name,
        "path": str(model_path),
        "saved_at": datetime.now(UTC).isoformat(),
        "run_dir": _run_dir(),
        "run_id": getattr(logger, "run_id", None),
        "config_digest": _config_digest(cfg),
        "metrics": metrics or {},
    }
    index_path = _index_path(model_save_root)
    entries = _load_index(index_path)
    entries.append(entry)
    index_path.write_text(json.dumps({"version": 1, "entries": entries}, indent=2))
