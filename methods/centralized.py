"""Centralized training baseline.

This pools all synthetic client training data and evaluates the resulting model
separately on each client's validation/test split. It is an approximate
upper-bound reference, not privacy-preserving FL.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from federated.client import FederatedClient
from methods.training_utils import evaluate_model, get_eval_loader, train_model


def run_centralized_training(
    clients: list[FederatedClient],
    model_fn: Callable[[], nn.Module],
    device: torch.device,
    rounds: int,
    local_epochs: int,
    split_name: str,
    eval_split: str,
    lr: float,
    batch_size: int,
    seed: int,
) -> tuple[list[dict[str, float | int | str]], nn.Module]:
    """Train one pooled model and evaluate it per client each round."""

    train_dataset = ConcatDataset([client.train_loader.dataset for client in clients])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = model_fn().to(device)
    rows: list[dict[str, float | int | str]] = []

    for round_idx in range(1, rounds + 1):
        train_model(model, train_loader, device=device, epochs=local_epochs, lr=lr)
        for client in clients:
            metrics = evaluate_model(model, get_eval_loader(client, eval_split), device=device)
            rows.append(
                {
                    "round": round_idx,
                    "client_id": client.client_id,
                    "method": "centralized",
                    "split": split_name,
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "loss": metrics["loss"],
                }
            )

    return rows, model
