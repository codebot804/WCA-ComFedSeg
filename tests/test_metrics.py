"""Sanity checks for binary segmentation metrics."""

import torch

from utils.metrics import dice_coefficient, iou_score


def test_perfect_prediction_has_unit_scores() -> None:
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = torch.where(targets > 0.5, torch.full_like(targets, 20.0), torch.full_like(targets, -20.0))

    assert torch.isclose(dice_coefficient(logits, targets), torch.tensor(1.0))
    assert torch.isclose(iou_score(logits, targets), torch.tensor(1.0))


def test_no_overlap_prediction_has_zero_scores() -> None:
    targets = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    logits = torch.tensor([[[[-20.0, -20.0], [20.0, 20.0]]]])

    assert torch.isclose(dice_coefficient(logits, targets), torch.tensor(0.0))
    assert torch.isclose(iou_score(logits, targets), torch.tensor(0.0))


def test_partial_overlap_dice_and_iou_are_different() -> None:
    targets = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    logits = torch.tensor([[[[20.0, -20.0], [20.0, -20.0]]]])

    dice = dice_coefficient(logits, targets)
    iou = iou_score(logits, targets)

    assert torch.isclose(dice, torch.tensor(0.5))
    assert torch.isclose(iou, torch.tensor(1.0 / 3.0))
    assert not torch.isclose(dice, iou)
