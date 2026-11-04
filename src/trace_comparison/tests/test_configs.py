"""Hydra config instantiation smoke tests for blocks."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

CONFIG_DIR = Path(__file__).parents[1] / "config"
BLOCK_DIR = CONFIG_DIR / "blocks"


@pytest.mark.parametrize("cfg_name", sorted([p.stem for p in BLOCK_DIR.glob("*.yaml")]))
def test_block_configs_instantiate(cfg_name):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="sim_matrix", overrides=[f"blocks={cfg_name}"])
    blocks = instantiate(cfg.blocks)
    if isinstance(blocks, dict):
        blocks = blocks["blocks"]
    assert len(blocks) > 0
