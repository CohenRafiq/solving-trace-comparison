import os

import torch

from neural_models_old.models.base import BaseModel


def load_model(
    model_class: type[BaseModel],
    model_weight_path: str | os.PathLike,
    device: torch.device,
    **model_params,
):
    model = model_class(**model_params)
    state_dict = torch.load(model_weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
