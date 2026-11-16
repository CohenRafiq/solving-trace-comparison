"""Smoke tests for trace-comparison blocks against real checkpoints."""

from pathlib import Path

import numpy as np
import pytest

from trace_comparison.blocks.embedding.clf_hidden import ClassifierHidden
from trace_comparison.blocks.embedding.clf_weights import ClassifierWeights
from trace_comparison.blocks.embedding.imt_weights import ImitationWeights
from trace_comparison.blocks.embedding.model_latent import LatentBlock

CKPT = Path(__file__).parents[3] / "checkpoints"


def jagged_input(n_solvers: int = 22, n_puzzles: int = 1) -> np.ndarray:
    x = np.empty((n_solvers, n_puzzles), dtype=object)
    for index in np.ndindex(x.shape):
        x[index] = np.zeros((1, 5, 5), dtype=np.float32)
    return x


@pytest.mark.skipif(not (CKPT / "classifier" / "06.pt").exists(), reason="missing classifier 06")
def test_classifier_weights_runs():
    block = ClassifierWeights(str(CKPT), "06")
    out = block(jagged_input())
    assert out.shape == (22, 256)


@pytest.mark.skipif(not (CKPT / "classifier" / "10.pt").exists(), reason="missing classifier 10")
def test_classifier_hidden_and_latent():
    clf = ClassifierHidden(str(CKPT), "10", "cpu")
    hidden = clf(np.zeros((22, 3, 125), dtype=np.float32))
    assert hidden.shape == (22, 3, 256)

    latent = LatentBlock(str(CKPT), "10", "classifier", "cpu")
    lat = latent(np.zeros((22, 3, 125), dtype=np.float32))
    assert lat.shape == (22, 3, 256)


@pytest.mark.skipif(not (CKPT / "imitation" / "04.pt").exists(), reason="missing imitation 04")
def test_imitation_latent_and_weights():
    latent = LatentBlock(str(CKPT), "04", "imitation", "cpu")
    lat = latent(np.zeros((22, 2, 20, 5, 5), dtype=np.float32))
    assert lat.shape == (22, 2, 500)

    weights = ImitationWeights(str(CKPT), "04")
    out = weights(jagged_input())
    assert out.shape[0] == 22
