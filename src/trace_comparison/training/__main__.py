"""Train a model on traces.  The task *and* its model are chosen in yaml:

    python -m trace_comparison.training input=puzzles.jsonl                 # default task
    python -m trace_comparison.training task=classifier input=puzzles.jsonl
    python -m trace_comparison.training task=imitation k=30 epochs=50 task.model.hidden_dim=512

Swap the model class itself with ``task.model._target_=my_pkg.MyModel`` (it receives the
same size arguments as the default model, see each task's ``model_kwargs``).
Config lives in ``config/train.yaml`` and ``config/task/*.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from trace_comparison.data.dataset import load_trace_dataset
from trace_comparison.training.common import fit, format_metrics, resolve_path, save_checkpoint, setup_run
from trace_comparison.training.task import Task


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    device = setup_run(cfg)
    task: Task = instantiate(cfg.task)
    print(f"task: {task.name}")

    data_path = resolve_path(Path(cfg.data_dir) / "traces_dataset" / "nonograms" / cfg.input)
    solvers = list(cfg.solvers) if cfg.solvers is not None else None
    ds = load_trace_dataset(data_path, solvers=solvers, max_size=cfg.max_size, puzzle_filter=cfg.puzzle_filter)
    data = task.build_data(ds, int(cfg.k), float(cfg.val_frac), int(cfg.seed or 0))
    task.describe(data)

    model, model_meta = task.build_model(data)
    model = model.to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer)

    result = fit(
        model, data.train, data.val, task.step, task.metrics,
        optimizer=optimizer, scheduler=scheduler,
        epochs=int(cfg.epochs), batch_size=int(cfg.batch_size), device=device,
    )  # fmt: skip

    model.load_state_dict(result.best_state)
    path = save_checkpoint(
        model,
        resolve_path(cfg.model_save_path),
        {"task": task.name, "k": int(cfg.k), **model_meta, **task.checkpoint_extras(result, data)},
    )
    print(f"best epoch {result.best_epoch} -> {path}")
    task.report(result, data)


if __name__ == "__main__":
    main()