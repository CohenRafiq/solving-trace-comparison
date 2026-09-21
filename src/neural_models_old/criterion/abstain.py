import math

import torch
import torch.nn.functional as F

from neural_models_old.criterion.base import BaseCriterion


class AbstainLoss(BaseCriterion):
    """
    Cross-entropy loss with an abstain option.

    The model outputs one logit per class per cell, with the LAST channel being
    the abstain class. Targets are class indices in [0, num_classes - 1].

    Args:
        abstain_penalty: Float in [0, 1]. Cost of fully abstaining relative to
            the loss for a uniform random guess. 0 = free abstention; 1 = abstain
            only when worse than random.
        entropy_weight: Weight for the entropy bonus that prevents collapse.
        eps: Numerical stability constant.
    """

    output_channels = 3

    def __init__(
        self,
        abstain_penalty: float = 0.5,
        entropy_weight: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not (0.0 <= abstain_penalty <= 1.0):
            raise ValueError(f"abstain_penalty must be in [0, 1], got {abstain_penalty}")
        self.abstain_penalty = abstain_penalty
        self.entropy_weight = entropy_weight
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: [batch, num_classes + 1, cells] — raw logits per class, with
                the last channel being abstain.
            targets: [batch, cells] — class indices in [0, num_classes - 1].
        """
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
        p_non_abs = probs[:, :-1, :]

        idx = targets_idx.unsqueeze(1)
        p_y = probs.gather(dim=1, index=idx).squeeze(1)

        committed = (1.0 - p_abs).clamp(min=self.eps)
        p_y_renorm = (p_y / committed).clamp(min=self.eps, max=1.0)

        ce_loss = -torch.log(p_y_renorm)
        abstain_cost = p_abs * l_abs

        entropy_bonus = -(p_non_abs * torch.log(p_non_abs.clamp(min=self.eps))).sum(dim=1)

        loss = ce_loss + abstain_cost - self.entropy_weight * entropy_bonus

        return loss.mean()

    def compute_abstain_mask(self, preds: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Given raw per-class logits [batch, num_classes + 1, cells], return a
        boolean mask of which cells the model would abstain on.

        Returns:
            mask: [batch, cells] boolean tensor (True = abstain).
        """
        probs = F.softmax(preds, dim=1)
        return probs[:, -1, :] > threshold
