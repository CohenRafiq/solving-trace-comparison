from __future__ import annotations

from dataclasses import dataclass, replace
import torch

from ...types import Strategy
from .state import NNState


@dataclass(frozen=True)
class NNStrategy(Strategy):
    """Apply the core to the hidden state, then unembed to a grid."""

    def __call__(self, state: NNState) -> NNState:
        with torch.no_grad():
            hidden = state.pop_of_hidden
            hidden = state.model.core(hidden)
            grids = state.model.unembed(hidden)
        return replace(state, pop_of_hidden=hidden, pop_of_grids=grids, step=state.step + 1)