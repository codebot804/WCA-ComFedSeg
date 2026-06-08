"""Binary segmentation metrics.

Model logits are converted to probabilities with sigmoid and thresholded at 0.5
before Dice and IoU are computed.
"""

from __future__ import annotations

import torch


def binary_predictions(
    outputs: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> torch.Tensor:
    """Convert model outputs or probabilities into binary masks.

    Args:
        outputs: Model logits when `from_logits=True`, otherwise probabilities.
        threshold: Probability threshold for foreground.
        from_logits: Apply sigmoid before thresholding when true.
    """

    probs = torch.sigmoid(outputs) if from_logits else outputs
    return (probs > threshold).float()


def _flatten_binary_masks(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    from_logits: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    preds = binary_predictions(outputs, threshold=threshold, from_logits=from_logits)
    targets = (targets >= 0.5).float()
    if preds.shape != targets.shape:
        raise ValueError(f"Prediction shape {tuple(preds.shape)} does not match target shape {tuple(targets.shape)}.")
    return preds.flatten(start_dim=1), targets.flatten(start_dim=1)


def dice_coefficient(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> torch.Tensor:
    """Compute mean per-sample Dice in [0, 1] for binary masks."""

    preds, targets = _flatten_binary_masks(outputs, targets, threshold=threshold, from_logits=from_logits)
    intersection = (preds * targets).sum(dim=1)
    denominator = preds.sum(dim=1) + targets.sum(dim=1)
    dice = torch.where(
        denominator > 0,
        (2.0 * intersection) / denominator.clamp_min(1e-7),
        torch.ones_like(denominator),
    )
    return dice.mean()


def iou_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> torch.Tensor:
    """Compute mean per-sample IoU in [0, 1] for binary masks."""

    preds, targets = _flatten_binary_masks(outputs, targets, threshold=threshold, from_logits=from_logits)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = torch.where(
        union > 0,
        intersection / union.clamp_min(1e-7),
        torch.ones_like(union),
    )
    return iou.mean()
