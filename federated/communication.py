"""Communication scheduling and cost helpers for Phase 4B."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from federated.client import FederatedClient


@dataclass(frozen=True)
class ClientSelection:
    client_id: int
    selected: bool
    selected_reason: str
    previous_validation_dice: float | None
    performance_deficit: float


def count_uploaded_parameters(state: dict[str, torch.Tensor]) -> int:
    """Count tensors uploaded as a client model state."""

    return int(sum(tensor.numel() for tensor in state.values()))


def estimate_uploaded_mb(uploaded_parameters: int, bytes_per_value: int = 4) -> float:
    """Estimate upload size in MiB using float32-equivalent values."""

    return float(uploaded_parameters * bytes_per_value) / float(1024**2)


def compute_performance_deficits(
    client_ids: list[int],
    previous_validation_dice: dict[int, float] | None,
) -> dict[int, float]:
    """Compute max(avg Dice - client Dice, 0) for available client metrics."""

    if previous_validation_dice is None:
        return {client_id: 0.0 for client_id in client_ids}

    safe_values = []
    for client_id in client_ids:
        value = previous_validation_dice.get(client_id)
        if value is None:
            continue
        value = float(value)
        if isfinite(value):
            safe_values.append(value)

    if not safe_values:
        return {client_id: 0.0 for client_id in client_ids}

    avg_dice = sum(safe_values) / float(len(safe_values))
    deficits: dict[int, float] = {}
    for client_id in client_ids:
        value = previous_validation_dice.get(client_id)
        if value is None or not isfinite(float(value)):
            deficits[client_id] = 0.0
        else:
            deficits[client_id] = max(avg_dice - float(value), 0.0)
    return deficits


def selected_client_count(total_clients: int, client_fraction: float, min_selected_clients: int) -> int:
    """Return bounded selected-client count for a communication round."""

    if total_clients <= 0:
        raise ValueError("total_clients must be positive.")
    requested = int(float(client_fraction) * total_clients)
    requested = max(requested, int(min_selected_clients))
    return min(max(requested, 1), total_clients)


def select_clients_for_communication(
    clients: list[FederatedClient],
    previous_validation_dice: dict[int, float] | None,
    round_idx: int,
    client_fraction: float,
    min_selected_clients: int,
    scheduler: str = "adaptive",
) -> list[ClientSelection]:
    """Select clients for Phase 4B communication-efficient WCA.

    Round 1 selects every client. Later rounds always include the previous
    weakest client, then fill remaining slots by performance deficit. If all
    deficits are zero, the fallback priority is data size.
    """

    if scheduler != "adaptive":
        raise ValueError("Only the adaptive communication scheduler is supported in Phase 4B.")

    client_ids = [client.client_id for client in clients]
    deficits = compute_performance_deficits(client_ids, previous_validation_dice)
    dice_lookup = previous_validation_dice or {}

    if round_idx == 1 or previous_validation_dice is None:
        return [
            ClientSelection(
                client_id=client.client_id,
                selected=True,
                selected_reason="all_clients_round1",
                previous_validation_dice=None,
                performance_deficit=0.0,
            )
            for client in clients
        ]

    select_count = selected_client_count(len(clients), client_fraction, min_selected_clients)
    valid_dice_clients = [
        client
        for client in clients
        if client.client_id in previous_validation_dice and isfinite(float(previous_validation_dice[client.client_id]))
    ]
    if valid_dice_clients:
        worst_client = min(valid_dice_clients, key=lambda client: float(previous_validation_dice[client.client_id]))
    else:
        worst_client = max(clients, key=lambda client: client.num_train_samples)

    selected_ids = {worst_client.client_id}
    any_deficit = any(deficit > 0.0 for deficit in deficits.values())
    if any_deficit:
        ranked_clients = sorted(
            clients,
            key=lambda client: (deficits[client.client_id], client.num_train_samples, -client.client_id),
            reverse=True,
        )
    else:
        ranked_clients = sorted(clients, key=lambda client: (client.num_train_samples, -client.client_id), reverse=True)

    for client in ranked_clients:
        if len(selected_ids) >= select_count:
            break
        selected_ids.add(client.client_id)

    selections = []
    for client in clients:
        selected = client.client_id in selected_ids
        if not selected:
            reason = "skipped"
        elif client.client_id == worst_client.client_id:
            reason = "worst_client"
        elif any_deficit:
            reason = "priority_deficit"
        else:
            reason = "data_size_fallback"
        previous_dice = dice_lookup.get(client.client_id)
        selections.append(
            ClientSelection(
                client_id=client.client_id,
                selected=selected,
                selected_reason=reason,
                previous_validation_dice=None if previous_dice is None else float(previous_dice),
                performance_deficit=deficits[client.client_id],
            )
        )
    return selections


def select_clients_for_conservative_communication(
    clients: list[FederatedClient],
    previous_validation_dice: dict[int, float] | None,
    round_idx: int,
    client_fraction: float,
    min_selected_clients: int,
    skipped_streaks: dict[int, int],
    warmup_rounds: int = 3,
    max_skip_streak: int = 1,
) -> list[ClientSelection]:
    """Select clients with a more conservative Phase 5G scheduler.

    This diagnostic scheduler keeps the previous weakest client in every
    post-warmup round and prevents any client from being skipped for more than
    ``max_skip_streak`` consecutive rounds. The goal is to retain some
    communication reduction while avoiding repeated exclusion of weak or
    unstable clients.
    """

    client_ids = [client.client_id for client in clients]
    deficits = compute_performance_deficits(client_ids, previous_validation_dice)
    dice_lookup = previous_validation_dice or {}

    if round_idx <= warmup_rounds or previous_validation_dice is None:
        return [
            ClientSelection(
                client_id=client.client_id,
                selected=True,
                selected_reason="conservative_warmup_all_clients",
                previous_validation_dice=None if previous_validation_dice is None else dice_lookup.get(client.client_id),
                performance_deficit=deficits[client.client_id],
            )
            for client in clients
        ]

    select_count = selected_client_count(len(clients), client_fraction, min_selected_clients)
    valid_dice_clients = [
        client
        for client in clients
        if client.client_id in previous_validation_dice and isfinite(float(previous_validation_dice[client.client_id]))
    ]
    if valid_dice_clients:
        worst_client = min(valid_dice_clients, key=lambda client: float(previous_validation_dice[client.client_id]))
    else:
        worst_client = max(clients, key=lambda client: client.num_train_samples)

    forced_ids = {
        client.client_id
        for client in clients
        if skipped_streaks.get(client.client_id, 0) >= max_skip_streak
    }
    selected_ids = {worst_client.client_id, *forced_ids}

    any_deficit = any(deficit > 0.0 for deficit in deficits.values())
    ranked_clients = sorted(
        clients,
        key=lambda client: (
            deficits[client.client_id] if any_deficit else 0.0,
            skipped_streaks.get(client.client_id, 0),
            client.num_train_samples,
            -client.client_id,
        ),
        reverse=True,
    )
    target_count = max(select_count, len(selected_ids))
    target_count = min(target_count, len(clients))
    for client in ranked_clients:
        if len(selected_ids) >= target_count:
            break
        selected_ids.add(client.client_id)

    selections = []
    for client in clients:
        selected = client.client_id in selected_ids
        if not selected:
            reason = "conservative_skipped"
        elif client.client_id == worst_client.client_id:
            reason = "conservative_worst_client"
        elif client.client_id in forced_ids:
            reason = "conservative_skip_cooldown"
        elif any_deficit:
            reason = "conservative_priority_deficit"
        else:
            reason = "conservative_data_size_fallback"
        previous_dice = dice_lookup.get(client.client_id)
        selections.append(
            ClientSelection(
                client_id=client.client_id,
                selected=selected,
                selected_reason=reason,
                previous_validation_dice=None if previous_dice is None else float(previous_dice),
                performance_deficit=deficits[client.client_id],
            )
        )
    return selections
