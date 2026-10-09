from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import torch

from ...types import State
from ..utils.const import DEVICE, empty_torch


@dataclass(frozen=True)
class GDBaseState(State, ABC):
    """Base class for gradient descent state."""

    pop_of_grids: torch.Tensor = field(default_factory=empty_torch)
    pop_grads: torch.Tensor = field(default_factory=empty_torch)
    torch_clues: torch.Tensor = field(default_factory=empty_torch)
    optimiser: torch.optim.Optimizer | None = field(
        default=None, compare=False
    )

    @abstractmethod
    def create_optimiser(self, param: torch.nn.Parameter) -> torch.optim.Optimizer:
        ...

    def _init_params(self, padded_clues: np.ndarray, grid_shape: tuple[int, int]) -> dict:
        full_zeros = torch.zeros((self.k, *grid_shape), dtype=torch.float32, device=DEVICE)
        return {
            "pop_of_grids": full_zeros.clone(),
            "pop_grads": full_zeros.clone(),
            "torch_clues": torch.from_numpy(padded_clues)
                             .float()
                             .to(DEVICE)
                             .reshape(1, -1)
                             .expand(self.k, -1),
            "optimiser": self.create_optimiser(torch.nn.Parameter(full_zeros.clone())),
        }


@dataclass(frozen=True)
class GDSGDState(GDBaseState):
    """Stochastic Gradient Descent (SGD) optimiser state."""
    
    lr: float = 0.01
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False

    def create_optimiser(self, param: torch.nn.Parameter) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            [param],
            lr=self.lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )


@dataclass(frozen=True)
class GDAdamState(GDBaseState):
    """Adam optimiser state."""

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False
    
    def create_optimiser(self, param: torch.nn.Parameter) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            [param],
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )