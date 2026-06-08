"""Loss functions for binary medical image segmentation."""

from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    """Soft Dice loss from logits.

    This uses probabilities rather than thresholded masks so gradients remain
    useful during training.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = (targets >= 0.5).float()
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCEWithLogits plus soft Dice loss."""

    def __init__(self, dice_weight: float = 1.0, positive_weight: float = 4.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(positive_weight)))
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = (targets >= 0.5).float()
        return self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)
