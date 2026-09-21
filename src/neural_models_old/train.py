import importlib

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Subset, random_split

from utils.io import save_model


@hydra.main(version_base=None, config_path="configs", config_name="n5_mlp_s")
def main(cfg: DictConfig):

    if cfg.seed:
        torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    logger = instantiate(cfg.logger)
    logger.setup(cfg)

    dataset = instantiate(cfg.data)

    target_path = cfg.model._target_
    module_path, class_name = target_path.rsplit(".", 1)
    model_cls = getattr(importlib.import_module(module_path), class_name)
    need_flat = getattr(model_cls, "require_flat_input", False)
    if need_flat and len(dataset.input_shape) > 1:
        dataset.flatten()

    criterion_path = cfg.criterion._target_
    crit_module, crit_class = criterion_path.rsplit(".", 1)
    criterion_cls = getattr(importlib.import_module(crit_module), crit_class)
    n_channels = getattr(criterion_cls, "output_channels", 1)

    if need_flat:
        input_size = int(np.prod(dataset.input_shape))
        output_size = n_channels * int(np.prod(dataset.target_shape))
    else:
        input_size = dataset.input_shape
        output_size = (n_channels,) + tuple(dataset.target_shape)

    model = instantiate(
        cfg.model,
        input_size=input_size,
        output_size=output_size,
    ).to(device)

    generator = torch.Generator().manual_seed(cfg.split.seed)
    shuffled_indices = torch.randperm(len(dataset), generator=generator).tolist()

    max_samples = cfg.split.get("max_samples", None)
    if max_samples is not None:
        shuffled_indices = shuffled_indices[:max_samples]
    dataset = Subset(dataset, shuffled_indices)

    train_size = int(cfg.split.train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=generator
    )

    train_loader = instantiate(cfg.dataloader, dataset=train_dataset, shuffle=True)

    test_loader = instantiate(cfg.dataloader, dataset=test_dataset, shuffle=False)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = instantiate(cfg.criterion).to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    scheduler = None
    if cfg.get("scheduler") and cfg.scheduler.get("_target_") is not None:
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    try:
        trainer = instantiate(
            cfg.trainer,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
        )
        trainer.train()

        print(trainer.test())
        save_model(cfg, model, logger)

    finally:
        logger.finish()


if __name__ == "__main__":
    main()
