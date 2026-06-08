"""Reliability-gated WCA aggregation for Phase 4F."""

from __future__ import annotations

from math import isfinite
from typing import Any

import torch

from methods.wca_comfedseg import wca_aggregate


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


def _normalize(weights: list[float], fallback: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0.0 or not isfinite(total):
        return list(fallback)
    normalized = [float(weight) / total for weight in weights]
    if any(not isfinite(weight) or weight < 0.0 for weight in normalized):
        return list(fallback)
    return normalized


def compute_data_weights(client_num_samples: list[int]) -> list[float]:
    """Return FedAvg-style data-size weights."""

    if not client_num_samples:
        raise ValueError("client_num_samples must not be empty.")
    if any(count < 0 for count in client_num_samples):
        raise ValueError("client sample counts must be non-negative.")
    total_samples = float(sum(client_num_samples))
    if total_samples <= 0.0:
        raise ValueError("Total number of client samples must be positive.")
    return [float(count) / total_samples for count in client_num_samples]


def compute_performance_deficits(client_validation_dice: list[float | None]) -> list[float]:
    """Compute deficit against the current client-average validation Dice."""

    safe_dice = [_safe_float(value) for value in client_validation_dice]
    valid_dice = [value for value in safe_dice if value is not None]
    if not valid_dice:
        return [0.0 for _ in safe_dice]
    average_dice = sum(valid_dice) / float(len(valid_dice))
    return [
        max(average_dice - dice, 0.0) if dice is not None else 0.0
        for dice in safe_dice
    ]


def compute_reliability_scores(
    previous_validation_dice: list[float | None] | None,
    current_validation_dice: list[float | None],
) -> tuple[list[float], list[float], list[float], bool]:
    """Return reliability scores from positive validation Dice improvement.

    The boolean flag is false when no previous Dice is available, which lets the
    caller preserve first-round FedAvg behavior while still logging current Dice.
    """

    client_count = len(current_validation_dice)
    if client_count == 0:
        raise ValueError("current_validation_dice must not be empty.")
    if previous_validation_dice is not None and len(previous_validation_dice) != client_count:
        raise ValueError("previous_validation_dice must match current_validation_dice length.")

    safe_current = [_safe_float(value) for value in current_validation_dice]
    safe_previous = (
        [_safe_float(value) for value in previous_validation_dice]
        if previous_validation_dice is not None
        else [None for _ in current_validation_dice]
    )
    has_previous = previous_validation_dice is not None and all(value is not None for value in safe_previous)

    improvements = [
        (current - previous) if current is not None and previous is not None else 0.0
        for previous, current in zip(safe_previous, safe_current)
    ]
    positive_improvements = [max(value, 0.0) for value in improvements]
    positive_sum = sum(positive_improvements)
    if positive_sum > 0.0 and isfinite(positive_sum):
        reliability_scores = [value / positive_sum for value in positive_improvements]
    else:
        reliability_scores = [1.0 / float(client_count) for _ in range(client_count)]
    return reliability_scores, improvements, positive_improvements, has_previous


def compute_reliability_gates(reliability_scores: list[float], min_gate: float = 0.2) -> list[float]:
    """Map normalized reliability scores to bounded reliability gates."""

    if not reliability_scores:
        raise ValueError("reliability_scores must not be empty.")
    min_gate = min(max(float(min_gate), 0.0), 1.0)
    return [
        min_gate + (1.0 - min_gate) * min(max(float(score), 0.0), 1.0)
        for score in reliability_scores
    ]


def compute_reliable_deficit_weights(
    performance_deficits: list[float],
    reliability_gates: list[float],
    data_weights: list[float],
) -> tuple[list[float], list[float]]:
    """Apply reliability gates to WCA deficits and normalize."""

    if len(performance_deficits) != len(reliability_gates) or len(data_weights) != len(performance_deficits):
        raise ValueError("deficits, gates, and data weights must have the same length.")
    reliable_deficits = [
        max(float(deficit), 0.0) * min(max(float(gate), 0.0), 1.0)
        for deficit, gate in zip(performance_deficits, reliability_gates)
    ]
    reliable_deficit_weights = _normalize(reliable_deficits, fallback=data_weights)
    return reliable_deficits, reliable_deficit_weights


def compute_final_rg_weights(
    data_weights: list[float],
    reliable_deficit_weights: list[float],
    alpha: float = 0.5,
) -> list[float]:
    """Blend data-size weights with reliability-gated deficit weights."""

    if len(data_weights) != len(reliable_deficit_weights):
        raise ValueError("data_weights and reliable_deficit_weights must have the same length.")
    alpha = min(max(float(alpha), 0.0), 1.0)
    blended = [
        (1.0 - alpha) * data_weight + alpha * deficit_weight
        for data_weight, deficit_weight in zip(data_weights, reliable_deficit_weights)
    ]
    return _normalize(blended, fallback=data_weights)


def compute_reliability_gated_wca_weights(
    client_num_samples: list[int],
    previous_validation_dice: list[float | None] | None,
    current_validation_dice: list[float | None],
    alpha: float = 0.5,
    min_gate: float = 0.2,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Compute Reliability-Gated WCA aggregation weights.

    Round 1 passes ``previous_validation_dice=None`` and returns data-size
    weights. Later rounds use current local validation Dice and previous local
    validation Dice to gate worst-client deficits by positive improvement.
    """

    if len(current_validation_dice) != len(client_num_samples):
        raise ValueError("current_validation_dice must match client_num_samples length.")

    alpha = min(max(float(alpha), 0.0), 1.0)
    min_gate = min(max(float(min_gate), 0.0), 1.0)
    data_weights = compute_data_weights(client_num_samples)
    safe_previous = (
        [_safe_float(value) for value in previous_validation_dice]
        if previous_validation_dice is not None
        else [None for _ in current_validation_dice]
    )
    safe_current = [_safe_float(value) for value in current_validation_dice]
    deficits = compute_performance_deficits(safe_current)
    reliability_scores, improvements, positive_improvements, has_previous = compute_reliability_scores(
        previous_validation_dice=previous_validation_dice,
        current_validation_dice=current_validation_dice,
    )
    reliability_gates = compute_reliability_gates(reliability_scores, min_gate=min_gate)
    reliable_deficits, reliable_deficit_weights = compute_reliable_deficit_weights(
        performance_deficits=deficits,
        reliability_gates=reliability_gates,
        data_weights=data_weights,
    )

    weak_improving_exists = any(
        deficit > 0.0 and positive_improvement > 0.0
        for deficit, positive_improvement in zip(deficits, positive_improvements)
    )

    if not has_previous:
        final_weights = list(data_weights)
    elif not weak_improving_exists:
        reliable_deficit_weights = list(data_weights)
        final_weights = list(data_weights)
    else:
        final_weights = compute_final_rg_weights(
            data_weights=data_weights,
            reliable_deficit_weights=reliable_deficit_weights,
            alpha=alpha,
        )

    rows = []
    for index, count in enumerate(client_num_samples):
        rows.append(
            {
                "client_id": index,
                "data_size": count,
                "data_weight": data_weights[index],
                "previous_validation_dice": "" if safe_previous[index] is None else safe_previous[index],
                "current_validation_dice": "" if safe_current[index] is None else safe_current[index],
                "validation_dice_improvement": improvements[index],
                "positive_improvement": positive_improvements[index],
                "reliability_score": reliability_scores[index],
                "reliability_gate": reliability_gates[index],
                "performance_deficit": deficits[index],
                "reliable_deficit": reliable_deficits[index],
                "reliable_deficit_weight": reliable_deficit_weights[index],
                "final_aggregation_weight": final_weights[index],
                "wca_alpha": alpha,
                "rg_min_gate": min_gate,
            }
        )

    return final_weights, rows


def aggregate_model_states_with_rg_weights(
    client_states: list[dict[str, torch.Tensor]],
    aggregation_weights: list[float],
) -> dict[str, torch.Tensor]:
    """Aggregate client model states with precomputed RG-WCA weights."""

    return wca_aggregate(client_states, aggregation_weights)
