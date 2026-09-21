import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


class JsonShapeError(Exception):
    """Raised when the shape of the data in the JSON file is not as expected."""
    pass

class NonogramsDataset(Dataset):
    """PyTorch Dataset for loading nonogram data from a .jsonl file and padding it."""

    def __init__(self, file_name: str | Path, max_size: int | None = None,
                 padding_type: str = "positional"):
        self.raw_clues, self.raw_grids = self.load_data(file_name, max_size)
        if padding_type == "positional":
            self.clues, self.pad_mask = self.pad_positional(self.raw_clues)
        elif padding_type == "sequential":
            self.clues, self.pad_mask = self.pad_sequential(self.raw_clues)
        else:
            raise ValueError("Invalid padding type. Choose 'positional' or 'sequential'.")
        self.grid = torch.tensor(self.raw_grids, dtype=torch.float32)
        self.input_shape = self.clues.shape[1:]
        self.output_shape = self.grid.shape[1:]

    def load_data(self, file_name: str | Path, max_size: int | None = None):
        clues, grids = [], []
        data_dir = Path(os.environ.get("DATA_DIR", "./data")) / "base_dataset" / "nonograms"
        data_path = data_dir / file_name
        with open(data_path) as fh:
            for line in fh:
                sample = json.loads(line)
                if "clues" in sample:
                    clue = sample["clues"]
                elif "rows" in sample and "cols" in sample:
                    clue = [sample["rows"], sample["cols"]]
                else:
                    raise JsonShapeError(
                        "Missing key in JSON data: expected 'clues' or 'rows'/'cols'"
                    )
                clues.append(clue)
                try:
                    grids.append(sample["grid"])
                except KeyError as e:
                    raise JsonShapeError(f"Missing key in JSON data: {e}") from e
                if max_size is not None and len(clues) >= max_size:
                    break
        return clues, grids

    def pad_positional(self, clues):
        max_runs_in_clue = max(
            (len(line) for puzzle in clues for groups in puzzle for line in groups),
            default=0,
        )

        def pad(line):
            return line + [0] * (max_runs_in_clue - len(line))

        padded = torch.tensor(
            [[[pad(line) for line in groups] for groups in puzzle]
            for puzzle in clues],
            dtype=torch.long
        )
        mask = padded != 0
        return padded, mask.float()  # Convert mask to float for compatibility with loss functions

    def pad_sequential(self, clues):
        max_dimension = max(len(clues[0][0]), len(clues[0][1]))
        max_padding = 2 * max_dimension * (
            (max_dimension + 1) // 2
        )  # upper bound on total number of inputs

        padded, masks = [], []
        for puzzle in clues:
            flat = [
                [val, row_or_col, line_idx, run_idx]
                for row_or_col, groups in enumerate(puzzle)
                for line_idx, line in enumerate(groups)
                for run_idx, val in enumerate(line)
            ]
            pad_len = max_padding - len(flat)
            padded.append(flat + [[0, 0, 0, 0]] * pad_len)
            masks.append([1] * len(flat) + [0] * pad_len)

        return (torch.tensor(padded, dtype=torch.long),
                torch.tensor(masks, dtype=torch.float32))

    def __len__(self):
        return len(self.clues)

    def __getitem__(self, idx):
        return (
            self.clues[idx],
            self.pad_mask[idx],
            self.grid[idx]
            )
