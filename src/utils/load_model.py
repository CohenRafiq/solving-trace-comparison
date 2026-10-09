import os

import torch
from torch import nn


def load_model(
    model_class: type[nn.Module],
    model_weight_path: str | os.PathLike,
    device: torch.device,
    **model_params,
):
    model = model_class(**model_params)
    state_dict = torch.load(model_weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
