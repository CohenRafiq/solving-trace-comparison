from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseModel(nn.Module, ABC):
    require_flat_input: bool = False

    def __init__(self, input_size: int | None = None, output_size: int | None = None):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...
