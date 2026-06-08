"""Federated client logic for synthetic FL baselines."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.losses import BCEDiceLoss
from utils.metrics import dice_coefficient, iou_score


class FederatedClient:
    """A simulated hospital/client with train, validation, and test loaders."""

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        model_fn: Callable[[], nn.Module],
        device: torch.device,
        lr: float = 1e-3,
    ) -> None:
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model_fn = model_fn
        self.device = device
        self.lr = lr
        self.criterion = BCEDiceLoss().to(device)
        self.local_model_state: dict[str, torch.Tensor] | None = None

    @property
    def num_train_samples(self) -> int:
        return len(self.train_loader.dataset)

    def train_from_global(
        self,
        global_state: dict[str, torch.Tensor],
        local_epochs: int,
        proximal_mu: float = 0.0,
        initial_state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], int, float]:
        """Train a local copy of the global model and return its weights."""

        model = self.model_fn().to(self.device)
        state_to_load = initial_state if initial_state is not None else global_state
        model.load_state_dict(deepcopy(state_to_load))
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        global_params = {
            name: tensor.detach().to(self.device)
            for name, tensor in global_state.items()
            if tensor.is_floating_point()
        }
        last_loss = 0.0

        for _ in range(local_epochs):
            for images, masks in self.train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = self.criterion(logits, masks)
                if proximal_mu > 0:
                    prox_term = torch.zeros((), device=self.device)
                    for name, parameter in model.named_parameters():
                        prox_term = prox_term + torch.sum((parameter - global_params[name]) ** 2)
                    loss = loss + 0.5 * proximal_mu * prox_term
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())

        cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        self.local_model_state = cpu_state
        return cpu_state, self.num_train_samples, last_loss

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        split: str = "val",
    ) -> dict[str, float]:
        """Evaluate the current global model on this client's split."""

        if split == "val":
            loader = self.val_loader
        elif split == "test":
            loader = self.test_loader
        else:
            raise ValueError("split must be 'val' or 'test'.")

        model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_samples = 0

        for images, masks in loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            batch_size = images.size(0)
            logits = model(images)
            loss = self.criterion(logits, masks)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_dice += float(dice_coefficient(logits, masks, from_logits=True).detach().cpu()) * batch_size
            total_iou += float(iou_score(logits, masks, from_logits=True).detach().cpu()) * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError(f"Client {self.client_id} has an empty {split} loader.")

        return {
            "loss": total_loss / total_samples,
            "dice": total_dice / total_samples,
            "iou": total_iou / total_samples,
        }

    def evaluate_state(self, model_state: dict[str, torch.Tensor], split: str = "val") -> dict[str, float]:
        model = self.model_fn().to(self.device)
        model.load_state_dict(deepcopy(model_state))
        return self.evaluate(model, split=split)
