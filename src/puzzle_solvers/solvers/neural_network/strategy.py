from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

import torch

from ...types import Strategy
from .state import NNState


@dataclass(frozen=True)
class NNStrategy(Strategy):
    """Apply the core to the hidden state, then unembed to a grid."""

    def __call__(self, state: NNState) -> NNState:  # type: ignore[override]
        assert state.model is not None
        model = cast(Any, state.model)
        with torch.no_grad():
            hidden = state.pop_of_hidden
            hidden = model.core(hidden)
            grids = model.unembed(hidden)
        return replace(state, pop_of_hidden=hidden, pop_of_grids=grids, step=state.step + 1)
