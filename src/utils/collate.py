from torch.utils.data import default_collate


def collate_default(batch):
    return (
        default_collate([b["X"] for b in batch]),
        default_collate([b["y"] for b in batch]),
        default_collate([b["padding_mask"] for b in batch]),
        [b["meta"] for b in batch],
    )


def collate_raw(batch):
    return (
        [b["X_raw"] for b in batch],
        [b["y_raw"] for b in batch],
        [b["meta"] for b in batch],
    )


def collate_combined(batch):
    return (
        default_collate([b["X"] for b in batch]),
        default_collate([b["y"] for b in batch]),
        default_collate([b["padding_mask"] for b in batch]),
        [b["X_raw"] for b in batch],
        [b["y_raw"] for b in batch],
        [b["meta"] for b in batch],
    )
