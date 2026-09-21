import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from neural_models_old.criterion.abstain import AbstainLoss
from neural_models_old.trainers.supervised import NNGSupervisedTrainer


class _PerCellLogits(nn.Module):
    def __init__(self, cells: int, channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(cells, cells * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_trainer(cells: int, n: int, epochs: int = 1) -> NNGSupervisedTrainer:
    X = torch.randn(n, cells)
    y = torch.randint(0, 2, (n, cells))

    loader = DataLoader(TensorDataset(X, y), batch_size=8)

    model = _PerCellLogits(cells)
    criterion = AbstainLoss(abstain_penalty=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    return NNGSupervisedTrainer(
        model=model,
        train_loader=loader,
        test_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        epochs=epochs,
        print_every=0,
    )


def test_supervised_trainer_smoke():
    trainer = _make_trainer(cells=16, n=32, epochs=2)
    trainer.train(log_every=10)

    metrics = trainer.evaluate(trainer.train_loader)
    assert "loss" in metrics
    assert metrics["loss"] >= 0


def test_supervised_trainer_abstain_metrics():
    trainer = _make_trainer(cells=8, n=16, epochs=1)
    metrics = trainer.evaluate(trainer.train_loader)
    assert "abstain_rate" in metrics
    assert 0.0 <= metrics["abstain_rate"] <= 1.0
