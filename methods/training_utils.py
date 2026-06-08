"""Shared training and evaluation helpers for Phase 3 baselines."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.losses import BCEDiceLoss
from utils.metrics import dice_coefficient, iou_score


def train_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> float:
    """Train a model and return the last observed training loss."""

    criterion = BCEDiceLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    last_loss = 0.0

    for _ in range(epochs):
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

    return last_loss


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a segmentation model with sample-weighted averaging."""

    criterion = BCEDiceLoss().to(device)
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.size(0)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_dice += float(dice_coefficient(logits, masks, from_logits=True).detach().cpu()) * batch_size
        total_iou += float(iou_score(logits, masks, from_logits=True).detach().cpu()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty loader.")

    return {
        "loss": total_loss / total_samples,
        "dice": total_dice / total_samples,
        "iou": total_iou / total_samples,
    }


def get_eval_loader(client, split: str) -> DataLoader:
    if split == "val":
        return client.val_loader
    if split == "test":
        return client.test_loader
    raise ValueError("split must be 'val' or 'test'.")

