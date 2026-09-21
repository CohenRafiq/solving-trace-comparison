from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseCriterion(nn.Module, ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor | tuple[Tensor, ...]: ...
