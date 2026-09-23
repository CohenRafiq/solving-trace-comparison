import os
from dataclasses import dataclass, field, replace

from puzzle_solvers.solvers import SOLVERS

SAMPLING_RATIO_RUNS: list[tuple] = [(1.0, 1)]
DEFAULT_SOLVERS: list[str] = list(SOLVERS)


@dataclass
class TracesConfig:
    input_path: str = "5x5-100_000-val.jsonl"
    max_size: int = 5
    max_time: float = 10.0
    batch_size: int = 10
    data_dir: str | None = None
    output_path: str | None = None
    solvers: list[str] | None = None
    sampling_ratio_runs: list[tuple] | None = None
    n_workers: int | None = None
    chunksize: int | None = None
    model_path: str | None = None
    clues_shape: tuple | None = None
    stats: dict = field(default_factory=dict)

    def resolve(self) -> "TracesConfig":
        data_dir = self.data_dir or os.environ.get("DATA_DIR", os.path.join(os.getcwd(), "data"))
        output_path = self.output_path or os.path.join(
            data_dir, "traces_dataset", "nonograms", "test.jsonl"
        )
        solvers = self.solvers or DEFAULT_SOLVERS
        sampling_ratio_runs = self.sampling_ratio_runs or SAMPLING_RATIO_RUNS
        n_workers = max(1, min(self.n_workers or os.cpu_count() or 1, self.max_size))
        chunksize = self.chunksize or max(1, min(self.max_size // n_workers, 10))
        return replace(
            self,
            data_dir=data_dir,
            output_path=output_path,
            solvers=solvers,
            sampling_ratio_runs=sampling_ratio_runs,
            n_workers=n_workers,
            chunksize=chunksize,
        )
