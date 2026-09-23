import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import Dataset

import utils.puzzle_types as t
from utils.grid import derive_clues_from_grid


class NonogramDataset(Dataset):
    def __init__(
        self,
        data: str | Path | t.Dataset,
        max_size: int | None = None,
        batch_size: int = 256,
        padding: str = "positional",
    ) -> None:
        self.max_size = max_size
        self.batch_size = batch_size
        self.padding = padding

        self.X_raw, self.y_raw, self.meta = self._materialize(self.parse(data))
        self.meta = [
            m | self._make_metadata(c, g)
            for c, g, m in zip(self.X_raw, self.y_raw, self.meta, strict=True)
        ]
        self.y = self.y_raw.flatten(start_dim=1)
        max_dimension = max(self.meta[0]["shape"])

        if padding == "positional":
            self.X, self.padding_mask = self._pad_positional(self.X_raw, max_dimension)
        elif padding == "sequence":
            self.X, self.padding_mask = self._pad_sequence(self.X_raw, max_dimension)
        else:
            raise ValueError(f"Unsupported padding type: {padding}")

        self._input_shape = tuple(self.X.shape[1:])
        self._target_shape = tuple(self.y_raw[0].shape)

    # ------------------------------------------------------------------ #
    #  Parsing
    # ------------------------------------------------------------------ #

    def parse(
        self, data: str | Path | t.Dataset
    ) -> t.ReformattedData | Iterator[t.ReformattedData]:
        if isinstance(data, (str, Path)):
            return self._parse_file(data)
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            return self._parse_grid_dataset(data)
        elif isinstance(data, tuple):
            raise ValueError(f"Unsupported tuple data (expected 2-element SplitData): {data!r}")
        elif isinstance(data, list):
            if len(data) == 2 and self._is_split_data(data):
                return self._parse_split_data(cast(t.SplitData, data))
            elif len(data) == 0:
                raise ValueError("Cannot parse an empty dataset.")
            elif self._is_grid_list(data):
                return self._parse_grid_dataset(cast(t.GridDataset, data))
            else:
                return self._stream_batches(iter(cast(list[dict], data)))
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _parse_file(self, file_path: str | Path) -> Iterator[t.ReformattedData]:
        data_dir = os.getenv("DATA_DIR", "./data/")
        new_file_path = os.path.join(data_dir, str(file_path))
        if isinstance(file_path, Path):
            file_path = str(file_path)
        if file_path.endswith(".jsonl"):
            return self._stream_batches(self._iter_jsonl(new_file_path))
        elif file_path.endswith(".parquet"):
            return self._stream_batches(self._iter_parquet(new_file_path))
        elif file_path.endswith(".npy"):
            return self._stream_batches(self._iter_npy(new_file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def _iter_jsonl(self, path: str) -> Iterator[dict]:
        with open(path) as f:
            for line in f:
                yield json.loads(line)

    def _iter_parquet(self, path: str) -> Iterator[dict]:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=self.batch_size):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                yield row.to_dict()

    def _iter_npy(self, path: str) -> Iterator[dict]:
        arr = np.load(path, allow_pickle=True)
        for item in arr:
            if isinstance(item, np.void):
                names = item.dtype.names
                if names is not None:
                    entry_dict: dict = {name: item[name] for name in names}
                    yield entry_dict
                else:
                    yield {}
            else:
                yield dict(item)

    def _stream_batches(self, entries: Iterator[dict]) -> Iterator[t.ReformattedData]:
        batch_clues: list[t.Clues] = []
        batch_grids: list[torch.Tensor] = []
        batch_meta: list[dict] = []
        count = 0

        for raw in entries:
            if self.max_size is not None and count >= self.max_size:
                break

            c, g, m = self._parse_dict_entry(raw)
            batch_clues.append(c)
            batch_grids.append(g)
            batch_meta.append(m)
            count += 1

            if len(batch_clues) >= self.batch_size:
                yield self._pack_batch(batch_clues, batch_grids, batch_meta)
                batch_clues, batch_grids, batch_meta = [], [], []

        if batch_clues:
            yield self._pack_batch(batch_clues, batch_grids, batch_meta)

    def _parse_grid_dataset(self, data: t.GridDataset) -> t.ReformattedData:
        grids = self._normalize_grids(data)
        if self.max_size:
            grids = grids[: self.max_size]
        list_grids = [g.tolist() for g in grids]
        clues: list[t.Clues] = [cast(t.Clues, list(derive_clues_from_grid(g))) for g in list_grids]
        meta: list[dict] = [{} for _ in range(len(clues))]
        return self._pack_batch(clues, grids, meta)

    def _parse_split_data(self, data: t.SplitData) -> t.ReformattedData:
        if isinstance(data, tuple):
            clues_list, grids_raw = data
            clues = cast(list[t.Clues], clues_list)
        else:
            clues = cast(list[t.Clues], data[0])
            grids_raw = cast(t.GridDataset, data[1])

        if self.max_size:
            clues = clues[: self.max_size]

        grids = [self._grid_to_tensor(g) for g in grids_raw]
        meta: list[dict] = [{} for _ in range(len(clues))]
        return self._pack_batch(clues, grids, meta)

    def _parse_dict_entry(self, entry: dict) -> tuple[t.Clues, torch.Tensor, dict]:
        meta = dict(entry)
        clues = cast(t.Clues, meta.pop("clues", None))
        if clues:
            rows = cast(list[list[int]], clues[0])
            cols = cast(list[list[int]], clues[1])
        else:
            rows = cast(list[list[int]], meta.pop("rows", meta.pop("row_clues", [])))
            cols = cast(list[list[int]], meta.pop("cols", meta.pop("col_clues", [])))

        grid = self._grid_to_tensor(cast(t.Grid, meta.pop("grid", meta.pop("solution", []))))
        if isinstance(meta.get("meta"), dict):
            meta.update(meta.pop("meta"))
        return [rows, cols], grid, meta

    def _parse_single_entry(self, entry: t.Entry) -> tuple[t.Clues, torch.Tensor, dict]:
        if isinstance(entry, dict):
            return self._parse_dict_entry(dict(entry))

        elif isinstance(entry, tuple) and len(entry) == 2:
            clues = cast(t.Clues, entry[0])
            grid = self._grid_to_tensor(cast(t.Grid, entry[1]))
            return clues, grid, {}

        elif isinstance(entry, list) and len(entry) == 2:
            clues = cast(t.Clues, entry[0])
            grid = self._grid_to_tensor(cast(t.Grid, entry[1]))
            return clues, grid, {}

        else:
            raise ValueError(f"Entry must be EntryDict or (clues, grid) pair. Got: {type(entry)}")

    # ------------------------------------------------------------------ #
    #  Padding
    # ------------------------------------------------------------------ #

    def _pad_positional(
        self, clues: list[t.Clues], max_dimension: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_runs_in_clue = (max_dimension + 1) // 2  # worst case: alternating 1s and 0s

        def pad(line: list[int]) -> list[int]:
            return list(line) + [0] * (max_runs_in_clue - len(line))

        def mask(line: list[int]) -> list[float]:
            return [0.0] * len(line) + [-torch.inf] * (max_runs_in_clue - len(line))

        padded = [
            [[pad(line) for line in (group if group else [[0]])] for group in puzzle]
            for puzzle in clues
        ]
        masks = [
            [[mask(line) for line in (group if group else [[0]])] for group in puzzle]
            for puzzle in clues
        ]
        return (
            torch.tensor(padded, dtype=torch.float32).flatten(start_dim=1),
            torch.tensor(masks, dtype=torch.float32).flatten(start_dim=1),
        )

    def _pad_sequence(
        self, clues: list[t.Clues], max_dimension: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_padding = 2 * max_dimension * ((max_dimension + 1) // 2)
        padded_seqs, masks = [], []
        for puzzle in clues:
            flat_seq = [
                [val, row_or_col, row_or_col_idx, run_idx]
                for row_or_col, groups in enumerate(puzzle)
                for row_or_col_idx, line in enumerate(groups)
                for run_idx, val in enumerate(line)
            ]
            pad_len = max_padding - len(flat_seq)

            padded_seqs.append(flat_seq + [[0, 0, 0, 0]] * pad_len)
            masks.append([0.0] * len(flat_seq) + [-torch.inf] * pad_len)
        return (
            torch.tensor(padded_seqs, dtype=torch.float32).flatten(start_dim=1),
            torch.tensor(masks, dtype=torch.float32).flatten(start_dim=1),
        )

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    def _pack_batch(
        self,
        clues: list[t.Clues],
        grids: list[torch.Tensor],
        meta: list[dict],
    ) -> t.ReformattedData:
        return clues, torch.stack(grids), meta

    def _is_split_data(self, data: list) -> bool:
        if len(data) != 2:
            return False
        first, second = data
        if not (isinstance(first, list) and len(first) > 0):
            return False
        # first should look like list[Clues] == list[[rows_group, cols_group]]
        puzzle0 = first[0]
        is_clues_shaped = (
            isinstance(puzzle0, list)
            and len(puzzle0) == 2
            and all(isinstance(group, list) for group in puzzle0)
        )
        return is_clues_shaped and isinstance(second, (list, np.ndarray, torch.Tensor))

    def _is_grid_list(self, data: list) -> bool:
        first = data[0]
        if isinstance(first, dict):
            return False
        if isinstance(first, (np.ndarray, torch.Tensor)):
            return True
        if isinstance(first, list) and len(first) > 0:
            if len(first) == 2 and self._looks_like_clues(first[0]):
                return False
            return True
        return False

    def _looks_like_clues(self, obj) -> bool:
        return (
            isinstance(obj, list)
            and len(obj) > 0
            and isinstance(obj[0], list)
            and (len(obj[0]) == 0 or isinstance(obj[0][0], list))
        )

    def _normalize_grids(self, data: t.GridDataset) -> list[torch.Tensor]:
        if isinstance(data, torch.Tensor):
            if data.dim() == 2:
                return [self._grid_to_tensor(data)]
            elif data.dim() == 3:
                return [self._grid_to_tensor(g) for g in data.unbind(0)]
            else:
                raise ValueError(f"Expected 2-D or 3-D tensor, got {data.dim()}-D")
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                return [self._grid_to_tensor(data)]
            elif data.ndim == 3:
                return [self._grid_to_tensor(g) for g in data]
            else:
                raise ValueError(f"Expected 2-D or 3-D ndarray, got {data.ndim}-D")
        elif isinstance(data, list):
            return [self._grid_to_tensor(g) for g in data]
        else:
            raise ValueError(f"Unsupported grid type: {type(data)}")

    def _grid_to_tensor(self, grid: t.Grid) -> torch.Tensor:
        if isinstance(grid, torch.Tensor):
            tensor = grid if grid.dim() == 2 else grid.squeeze()
        elif isinstance(grid, np.ndarray):
            if grid.dtype == object:
                tensor = torch.stack([torch.from_numpy(row.copy()) for row in grid])
            else:
                tensor = torch.from_numpy(grid.copy()).squeeze()
        elif isinstance(grid, list):
            tensor = torch.tensor(grid)
        else:
            raise ValueError(f"Cannot convert {type(grid)} to tensor")

        if tensor.dim() != 2:
            raise ValueError(f"Grid must be 2-D after squeeze, got shape {tuple(tensor.shape)}")

        return tensor.to(dtype=torch.float32)

    def _make_metadata(self, clues: t.Clues, grid: torch.Tensor) -> dict:
        grid_density = torch.sum(grid) / (grid.shape[0] * grid.shape[1])
        mean_clue_runs = np.mean([len(row) for row in clues[0]] + [len(col) for col in clues[1]])
        return {"shape": grid.shape, "density": grid_density, "mean_clue_runs": mean_clue_runs}

    def _materialize(
        self, parsed: t.ReformattedData | Iterator[t.ReformattedData]
    ) -> t.ReformattedData:
        if isinstance(parsed, tuple):
            return parsed

        all_clues: list[t.Clues] = []
        all_grids: list[torch.Tensor] = []
        all_meta: list[dict] = []

        for clues_batch, grid_batch, meta_batch in parsed:
            all_clues.extend(clues_batch)
            all_grids.append(grid_batch)
            all_meta.extend(meta_batch)

        if not all_grids:
            raise ValueError("No data parsed: input produced zero entries.")

        return all_clues, torch.cat(all_grids, dim=0), all_meta

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, object]:
        return {
            "X": self.X[idx],
            "y": self.y[idx],
            "padding_mask": self.padding_mask[idx],
            "meta": self.meta[idx],
            "X_raw": self.X_raw[idx],
            "y_raw": self.y_raw[idx],
        }

    @property
    def input_shape(self) -> tuple[int, ...] | None:
        return self._input_shape

    @property
    def target_shape(self) -> tuple[int, ...] | None:
        return self._target_shape

    def flatten(self):
        self.X = self.X.flatten(start_dim=1)
        self._input_shape = (self.X[0].numel(),)
        if self.y is not None:
            self.y = self.y.flatten(start_dim=1)
            self._target_shape = (int(np.prod(self._target_shape)),)
