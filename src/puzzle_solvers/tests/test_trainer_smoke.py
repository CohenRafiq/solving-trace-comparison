"""Smoke tests for the training loop and the abstain metric.

These used to exercise `neural_models_old.trainers.supervised.NNGSupervisedTrainer`,
which no longer exists. The replacement is the `Trainer` in
`neural_models.train` plus `AbstainLoss` from `neural_models.criterions`, whose
API is different: the batch is a `(inputs, mask, targets)` triple and the model
takes `(inputs, mask)`.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from neural_models.criterions import AbstainLoss
from neural_models.train import Trainer

CELLS = 16
CLASSES = 3  # two answers plus the abstain channel


class _PerCellLogits(nn.Module):
    def __init__(self, cells: int, channels: int = CLASSES):
        super().__init__()
        self.cells = cells
        self.channels = channels
        self.fc = nn.Linear(cells, cells * channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.fc(x).reshape(x.shape[0], self.channels, self.cells)


def _make_trainer(n: int, cells: int = CELLS) -> tuple[Trainer, DataLoader]:
    X = torch.randn(n, cells)
    mask = torch.ones(n, cells, dtype=torch.bool)
    y = torch.randint(0, CLASSES - 1, (n, cells))

    loader = DataLoader(TensorDataset(X, mask, y), batch_size=8)

    model = _PerCellLogits(cells)
    criterion = AbstainLoss(abstain_penalty=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
    ), loader


def test_trainer_runs_a_training_epoch():
    trainer, loader = _make_trainer(n=32)
    loss, acc, _ = trainer.run_epoch(loader, train=True)
    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0


def test_trainer_runs_an_eval_epoch_without_gradients():
    trainer, loader = _make_trainer(n=16)
    loss, acc, _ = trainer.run_epoch(loader, train=False)
    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0


def test_abstain_rate_is_reported_for_the_abstain_criterion():
    trainer, loader = _make_trainer(n=16)
    _, _, additional = trainer.run_epoch(loader, train=False)
    assert "abstain_rate" in additional
    assert 0.0 <= additional["abstain_rate"] <= 1.0
