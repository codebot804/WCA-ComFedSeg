"""Worst-client-aware aggregation for Phase 4A.

This module only implements the aggregation component of WCA-ComFedSeg. The
communication-efficient scheduler is intentionally left for a later phase.
"""

from __future__ import annotations

from collections import OrderedDict
from math import isfinite
from typing import Any

import torch


def _safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    return parsed


def compute_wca_aggregation_weights(
    client_num_samples: list[int],
    client_validation_dice: list[float | None] | None,
    alpha: float = 0.5,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Compute data-size and worst-client-aware aggregation weights.

    Round 1 can pass ``client_validation_dice=None`` to recover FedAvg weights.
    Later rounds combine data-size weights with a deficit weight:

    ``final = (1 - alpha) * data_weight + alpha * deficit_weight``

    where ``deficit_i = max(avg_dice - dice_i, 0)``. If deficits are unavailable
    or sum to zero, the function falls back to data-size weights.
    """

    if not client_num_samples:
        raise ValueError("client_num_samples must not be empty.")
    if any(count < 0 for count in client_num_samples):
        raise ValueError("client sample counts must be non-negative.")
    if client_validation_dice is not None and len(client_validation_dice) != len(client_num_samples):
        raise ValueError("client_validation_dice must match client_num_samples length.")

    alpha = min(max(float(alpha), 0.0), 1.0)
    total_samples = float(sum(client_num_samples))
    if total_samples <= 0:
        raise ValueError("Total number of client samples must be positive.")

    data_weights = [float(count) / total_samples for count in client_num_samples]
    safe_dice = (
        [_safe_float(value) for value in client_validation_dice]
        if client_validation_dice is not None
        else [None for _ in client_num_samples]
    )
    valid_dice = [value for value in safe_dice if value is not None]

    deficits = [0.0 for _ in client_num_samples]
    if valid_dice:
        avg_dice = sum(valid_dice) / float(len(valid_dice))
        deficits = [
            max(avg_dice - dice, 0.0) if dice is not None else 0.0
            for dice in safe_dice
        ]

    deficit_sum = sum(deficits)
    if deficit_sum <= 0:
        deficit_weights = [0.0 for _ in client_num_samples]
        final_weights = list(data_weights)
    else:
        deficit_weights = [deficit / deficit_sum for deficit in deficits]
        final_weights = [
            (1.0 - alpha) * data_weight + alpha * deficit_weight
            for data_weight, deficit_weight in zip(data_weights, deficit_weights)
        ]

    final_sum = sum(final_weights)
    if final_sum <= 0 or not isfinite(final_sum):
        final_weights = list(data_weights)
    else:
        final_weights = [weight / final_sum for weight in final_weights]

    rows = []
    for index, count in enumerate(client_num_samples):
        rows.append(
            {
                "client_id": index,
                "data_size": count,
                "data_weight": data_weights[index],
                "validation_dice_used": "" if safe_dice[index] is None else safe_dice[index],
                "performance_deficit": deficits[index],
                "deficit_weight": deficit_weights[index],
                "final_aggregation_weight": final_weights[index],
                "wca_alpha": alpha,
            }
        )

    return final_weights, rows


def wca_aggregate(
    client_states: list[dict[str, torch.Tensor]],
    aggregation_weights: list[float],
    exclude_keys: set[str] | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Aggregate model states using precomputed WCA weights."""

    if not client_states:
        raise ValueError("client_states must not be empty.")
    if len(client_states) != len(aggregation_weights):
        raise ValueError("client_states and aggregation_weights must have the same length.")

    weight_sum = sum(float(weight) for weight in aggregation_weights)
    if weight_sum <= 0:
        raise ValueError("aggregation_weights must have a positive sum.")
    normalized_weights = [float(weight) / weight_sum for weight in aggregation_weights]

    aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
    exclude_keys = exclude_keys or set()
    for key in client_states[0].keys():
        if key in exclude_keys:
            continue
        weighted_sum = None
        for state, weight in zip(client_states, normalized_weights):
            tensor = state[key].detach().cpu()
            weighted_tensor = tensor * weight
            weighted_sum = weighted_tensor if weighted_sum is None else weighted_sum + weighted_tensor
        aggregated[key] = weighted_sum
    return aggregated
