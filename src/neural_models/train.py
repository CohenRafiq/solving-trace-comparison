import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from neural_models.io import save_model
from neural_models.logger import Logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def run_epoch(self, data_loader: DataLoader, train: bool):
        self.model.train(train)
        total_loss, total_acc, n = 0.0, 0.0, 0
        additional_metrics = {}

        with torch.set_grad_enabled(train):
            for inputs, mask, targets in data_loader:
                inputs, mask, targets = (
                    inputs.to(self.device), mask.to(self.device), targets.to(self.device)
                )

                if train:
                    self.optimizer.zero_grad()

                outputs = self.model(inputs, mask)
                loss = self.criterion(outputs, targets)

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                predictions = self.criterion.predict(outputs)
                total_acc += (predictions == targets).flatten(1).all(dim=1).float().mean()
                if type(self.criterion).__name__ == "AbstainLoss":
                    # Calculate abstain rate
                    abstain_rate = (outputs.argmax(dim=1) == outputs.shape[1] - 1).float().mean().item()
                    additional_metrics["abstain_rate"] = additional_metrics.get("abstain_rate", 0) + abstain_rate
                n += 1
        for key in additional_metrics:
            additional_metrics[key] /= n
        return total_loss / n, total_acc / n, additional_metrics


def train(trainer: Trainer, train_loader: DataLoader, val_loader: DataLoader,
          epochs: int, logger: Logger, scheduler=None) -> dict[str, float]:
    final_metrics = {}
    for epoch in range(epochs):
        train_loss, train_acc, train_additional = trainer.run_epoch(train_loader, train=True)
        val_loss, val_acc, val_additional = trainer.run_epoch(val_loader, train=False)

        if scheduler is not None:
            scheduler.step()

        metrics = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
        }
        if "abstain_rate" in train_additional:
            metrics["train_abstain_rate"] = float(train_additional["abstain_rate"])
        if "abstain_rate" in val_additional:
            metrics["val_abstain_rate"] = float(val_additional["abstain_rate"])
        logger.log_metrics(metrics, step=epoch)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | LOSSES - "
            f"Train: {train_loss:.4f}, Val: {val_loss:.4f} | "
            f"ACCURACY - Train: {train_acc:.4f}, Val: {val_acc:.4f}"
            + (
                f" | Abstain Rate: {val_additional['abstain_rate']:.4f}"
                if "abstain_rate" in val_additional
                else ""
            )
        )
        final_metrics = metrics
    return final_metrics


@hydra.main(version_base=None, config_path="configs", config_name="recmlp")
def main(cfg: DictConfig):
    """Main function to set up and run the training process from hydra configuration."""

    if cfg.seed:
        torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset = instantiate(cfg.train_dataset)
    val_dataset = instantiate(cfg.val_dataset)
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=cfg.get("pin_memory", False),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    criterion = instantiate(cfg.criterion).to(device)
    output_shape = train_dataset.output_shape
    if type(criterion).__name__ == "AbstainLoss":
        num_classes = 3  # 0, 1, abstain (TODO: Make dynamic for different puzzles)
        output_shape = torch.Size([num_classes] + list(output_shape))
    model = instantiate(cfg.model, input_shape=train_dataset.input_shape, output_shape=output_shape)
    model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model.to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = (
        instantiate(cfg.lr_scheduler, optimizer=optimizer) if "lr_scheduler" in cfg else None
    )
    logger = instantiate(cfg.logger)
    logger.watch_model(model)

    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)

    try:
        final_metrics = train(trainer, train_loader, val_loader, cfg.epochs, logger, scheduler)
        save_model(cfg, model, logger, final_metrics)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
