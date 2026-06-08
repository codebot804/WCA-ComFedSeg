"""Local training baseline.

Each client trains an independent model on its own local data. There is no
server aggregation.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from federated.client import FederatedClient
from methods.training_utils import evaluate_model, get_eval_loader, train_model


def run_local_training(
    clients: list[FederatedClient],
    model_fn: Callable[[], nn.Module],
    device: torch.device,
    rounds: int,
    local_epochs: int,
    split_name: str,
    eval_split: str,
    lr: float,
) -> tuple[list[dict[str, float | int | str]], nn.Module]:
    """Run local independent training and return per-client logs."""

    rows: list[dict[str, float | int | str]] = []
    local_models = {client.client_id: model_fn().to(device) for client in clients}

    for round_idx in range(1, rounds + 1):
        for client in clients:
            model = local_models[client.client_id]
            train_model(model, client.train_loader, device=device, epochs=local_epochs, lr=lr)
            metrics = evaluate_model(model, get_eval_loader(client, eval_split), device=device)
            rows.append(
                {
                    "round": round_idx,
                    "client_id": client.client_id,
                    "method": "local",
                    "split": split_name,
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "loss": metrics["loss"],
                }
            )

    first_client_id = clients[0].client_id
    return rows, local_models[first_client_id]
