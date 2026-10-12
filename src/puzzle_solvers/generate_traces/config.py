import os
from pathlib import Path
from typing import Any, cast

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


def original_cwd() -> Path:
    if HydraConfig.initialized():
        return Path(get_original_cwd())
    return Path.cwd()


def _resolve_path(path: str | Path, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def resolve(cfg: DictConfig) -> DictConfig:
    """Resolve paths and compute worker/chunk parameters."""
    root = original_cwd()
    
    # Resolve data directory with fallback priority: config > env > default
    data_dir = cfg.get("data_dir") or os.environ.get("DATA_DIR") or root / "data"
    data_dir = _resolve_path(data_dir, root)
    
    # Output defaults to a subdirectory under data_dir
    output = _resolve_path(cfg.output, data_dir / "traces_dataset" / "nonograms")
    
    # Compute workers and chunksize with sensible bounds
    max_size = max(int(cfg.max_size), 1)
    cpu_count = os.cpu_count() or 1
    workers = max(1, min(int(cfg.workers or cpu_count), max_size))
    chunksize = max(1, min(int(cfg.chunksize or max_size // workers), 10))
    
    resolved = cfg.copy()
    resolved.data_dir = str(data_dir)
    resolved.output = str(output)
    resolved.workers = workers
    resolved.chunksize = chunksize
    return resolved


def to_container(cfg: DictConfig) -> dict[str, Any]:
    """Convert to a plain dictionary for printing or serialization."""
    container = OmegaConf.to_container(cfg, resolve=True)
    return cast(dict[str, Any], container)