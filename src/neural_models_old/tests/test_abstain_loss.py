import math

import pytest
import torch

from neural_models_old.criterion.abstain import AbstainLoss


def test_output_channels():
    assert AbstainLoss.output_channels == 3


def test_perfectly_committed_prediction_has_low_loss():
    loss = AbstainLoss(abstain_penalty=0.0, entropy_weight=0.0)
    preds = torch.zeros(2, 3, 4)
    preds[:, 0, :] = 10.0
    preds[:, 1, :] = -10.0
    preds[:, 2, :] = -10.0
    targets = torch.zeros(2, 4, dtype=torch.long)

    out = loss(preds, targets)
    assert out.item() < 1e-3


def test_abstain_mask_flags_high_abstain_probability():
    preds = torch.zeros(2, 3, 4)
    preds[:, 2, :] = 10.0
    preds[:, 0, :] = -10.0
    preds[:, 1, :] = -10.0

    mask = AbstainLoss().compute_abstain_mask(preds, threshold=0.5)
    assert mask.shape == (2, 4)
    assert mask.all()


def test_abstain_mask_clears_low_abstain_probability():
    preds = torch.zeros(1, 3, 4)
    preds[:, 0, :] = 10.0
    preds[:, 2, :] = -10.0
    preds[:, 1, :] = -10.0

    mask = AbstainLoss().compute_abstain_mask(preds, threshold=0.5)
    assert not mask.any()


def test_invalid_target_raises():
    loss = AbstainLoss()
    preds = torch.zeros(1, 3, 4)
    targets = torch.tensor([[3, 0, 0, 0]])
    with pytest.raises(ValueError):
        loss(preds, targets)


def test_invalid_penalty_raises():
    with pytest.raises(ValueError):
        AbstainLoss(abstain_penalty=1.5)


def test_non_abstain_loss_above_random_baseline():
    loss = AbstainLoss(abstain_penalty=0.5, entropy_weight=0.0)
    preds = torch.zeros(2, 3, 4)
    preds[:, 2, :] = 10.0
    preds[:, 0, :] = -10.0
    preds[:, 1, :] = -10.0
    targets = torch.zeros(2, 4, dtype=torch.long)

    out = loss(preds, targets)
    assert out.item() > math.log(3)
