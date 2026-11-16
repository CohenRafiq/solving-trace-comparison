import numpy as np

from ..block_types import AlignmentBlock


class Resample(AlignmentBlock):
    def _apply(self, trace: np.ndarray) -> np.ndarray:
        indices = np.linspace(0, trace.shape[0] - 1, self.k).round().astype(int)
        return trace[indices]


class MeanPool(AlignmentBlock):
    def _output_dtype(self, x: np.ndarray) -> np.dtype:
        return np.zeros(1, dtype=super()._output_dtype(x)).mean().dtype

    def _apply(self, trace: np.ndarray) -> np.ndarray:
        t = trace.shape[0]
        i = np.arange(self.k)
        starts = (i * t) // self.k
        ends = -(-((i + 1) * t) // self.k)  # ceil division, so no bin is ever empty
        return np.stack([trace[s:e].mean(axis=0) for s, e in zip(starts, ends)])


class DynamicTimeWarping(AlignmentBlock):
    # TODO
    ...
