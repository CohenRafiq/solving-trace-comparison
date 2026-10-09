from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import torch

from ...types import Strategy
from ..utils.const import DEVICE
from .state import GDBaseState


@dataclass(frozen=True)
class GDStrategy(Strategy):
    """Perform a single gradient descent step on the state."""

    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = field(default=None)

    def __call__(self, state: GDBaseState) -> GDBaseState:
        """Perform a single gradient descent step on the state and return the new state."""
        param = state.optimiser.param_groups[0]["params"][0]
        param.data = state.grid.detach().clone().to(DEVICE)
        
        with torch.enable_grad():
            loss = self.loss_fn(param.reshape(state.k, -1), state.torch_clues)
            state.optimiser.zero_grad(set_to_none=True)
            loss.backward()
            grad = param.grad.detach().clone()
        
        state.optimiser.step()
        
        return replace(
            state,
            grid=param.detach().clone(),
            grad=grad,
            step=state.step + 1,
        )