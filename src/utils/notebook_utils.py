import os
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # adjust the index to where this file sits

def reset_cwd_and_env() -> Path:
    """chdir to the repo root and point DATA_DIR / MODEL_DIR at it."""
    os.chdir(ROOT)
    os.environ["DATA_DIR"] = str(ROOT / "data")
    os.environ["MODEL_DIR"] = str(ROOT / "checkpoints")

def load_traces_dataset(dataset_name: str = "test") -> pd.DataFrame:
    path = os.path.join(os.getcwd(), f"data/traces_dataset/nonograms/{dataset_name}.jsonl")
    with open(path, "r") as f:
        json_dataset = [json.loads(line) for line in f.readlines()]
    traces = [
        {"puzzle_idx": i, "method": m, **run}
        for i, p in enumerate(json_dataset)
        for m, d in p["traces"].items()
        for run in d
    ]
    return pd.json_normalize(traces)