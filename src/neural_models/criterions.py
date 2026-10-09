import math

import torch
import torch.nn.functional as F
from torch import nn


class MSELoss(nn.MSELoss):
    """Mean squared error loss with added prediction."""

    def predict(self, preds: torch.Tensor) -> torch.Tensor:
        return (preds > 0.5).long()

class AbstainLoss(nn.Module):
    """Cross-entropy loss with an abstain option."""

    def __init__(
        self,
        abstain_penalty: float = 0.4,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not (0.0 <= abstain_penalty <= 1.0):
            raise ValueError(f"abstain_penalty must be in [0, 1], got {abstain_penalty}")
        self.abstain_penalty = abstain_penalty
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.flatten(start_dim=2)
        targets = targets.flatten(start_dim=1)
        num_classes = preds.size(1) - 1
        if num_classes < 1:
            raise ValueError(
                f"preds must have at least 2 channels (classes + abstain), got {preds.size(1)}"
            )

        targets_idx = targets.long()
        if targets_idx.max() >= num_classes:
            raise ValueError(
                f"targets must be < num_classes ({num_classes}), got max={targets_idx.max()}"
            )

        l_abs = self.abstain_penalty * math.log(num_classes)

        probs = F.softmax(preds, dim=1)
        p_abs = probs[:, -1, :]

        idx = targets_idx.unsqueeze(1)
        p_y = probs.gather(dim=1, index=idx).squeeze(1)

        committed = (1.0 - p_abs).clamp(min=self.eps)
        p_y_renorm = (p_y / committed).clamp(min=self.eps, max=1.0)

        ce_loss = -torch.log(p_y_renorm)
        abstain_cost = abstain_cost = -l_abs * torch.log(committed)

        loss = committed * ce_loss + abstain_cost

        return loss.mean()

    def predict(self, preds: torch.Tensor) -> torch.Tensor:
        return preds[:, :-1, :].argmax(dim=1)
