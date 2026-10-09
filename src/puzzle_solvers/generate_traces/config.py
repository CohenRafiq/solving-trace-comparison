import os
from dataclasses import dataclass, field, replace

from old.solvers import SOLVERS


@dataclass
class TracesConfig:
    input: str = "5x5-100_000-val.jsonl"
    max_size: int = 5
    max_time: float = 10.0
    batch_size: int = 10
    data_dir: str | None = None
    output: str = "test.jsonl"
    solvers: list[str] | None = None
    workers: int | None = None
    chunksize: int | None = None
    model_path: str | None = None
    clues_shape: tuple | None = None
    stats: dict = field(default_factory=dict)

    def resolve(self) -> "TracesConfig":
        data_dir = self.data_dir or os.environ.get("DATA_DIR", os.path.join(os.getcwd(), "data"))
        output = os.path.join(data_dir, "traces_dataset", "nonograms", self.output)
        solvers = self.solvers or list(SOLVERS)
        workers = max(1, min(self.workers or os.cpu_count() or 1, self.max_size))
        chunksize = self.chunksize or max(1, min(self.max_size // workers, 10))
        return replace(
            self,
            data_dir=data_dir,
            output=output,
            solvers=solvers,
            workers=workers,
            chunksize=chunksize,
        )
