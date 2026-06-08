"""WCA-Smooth aggregation weights for Phase 4D."""

from __future__ import annotations

from math import isfinite
from typing import Any

from methods.wca_comfedseg import compute_wca_aggregation_weights


def _normalize(weights: list[float], fallback: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0.0 or not isfinite(total):
        return list(fallback)
    normalized = [float(weight) / total for weight in weights]
    if any(not isfinite(weight) for weight in normalized):
        return list(fallback)
    return normalized


def compute_wca_smooth_aggregation_weights(
    client_num_samples: list[int],
    client_validation_dice: list[float | None] | None,
    previous_weights: list[float] | None,
    alpha: float = 0.5,
    beta: float = 0.5,
    max_weight: float = 0.5,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Compute WCA weights with max-weight capping and temporal smoothing.

    The raw WCA weights are computed by the Phase 4A formula. The weights are
    then capped, renormalized, smoothed against the previous round's weights,
    and renormalized again. Round 1 naturally falls back to data-size weights.
    """

    raw_weights, raw_rows = compute_wca_aggregation_weights(
        client_num_samples=client_num_samples,
        client_validation_dice=client_validation_dice,
        alpha=alpha,
    )
    beta = min(max(float(beta), 0.0), 1.0)
    max_weight = min(max(float(max_weight), 1e-12), 1.0)

    data_weights = [float(row["data_weight"]) for row in raw_rows]
    if client_validation_dice is None and previous_weights is None:
        rows = []
        for index, raw_row in enumerate(raw_rows):
            rows.append(
                {
                    "client_id": raw_row["client_id"],
                    "data_size": raw_row["data_size"],
                    "data_weight": raw_row["data_weight"],
                    "validation_dice_used": raw_row["validation_dice_used"],
                    "performance_deficit": raw_row["performance_deficit"],
                    "deficit_weight": raw_row["deficit_weight"],
                    "raw_wca_weight": data_weights[index],
                    "capped_weight": data_weights[index],
                    "previous_weight": data_weights[index],
                    "smoothed_weight": data_weights[index],
                    "final_aggregation_weight": data_weights[index],
                    "wca_alpha": min(max(float(alpha), 0.0), 1.0),
                    "wca_beta": beta,
                    "wca_max_weight": max_weight,
                }
            )
        return data_weights, rows

    capped_pre_normalization = [min(weight, max_weight) for weight in raw_weights]
    capped_weights = _normalize(capped_pre_normalization, fallback=data_weights)

    if previous_weights is None or len(previous_weights) != len(capped_weights):
        safe_previous = list(data_weights)
    else:
        safe_previous = _normalize([float(weight) for weight in previous_weights], fallback=data_weights)

    smoothed_pre_normalization = [
        beta * previous_weight + (1.0 - beta) * capped_weight
        for previous_weight, capped_weight in zip(safe_previous, capped_weights)
    ]
    final_weights = _normalize(smoothed_pre_normalization, fallback=capped_weights)

    rows = []
    for index, raw_row in enumerate(raw_rows):
        rows.append(
            {
                "client_id": raw_row["client_id"],
                "data_size": raw_row["data_size"],
                "data_weight": raw_row["data_weight"],
                "validation_dice_used": raw_row["validation_dice_used"],
                "performance_deficit": raw_row["performance_deficit"],
                "deficit_weight": raw_row["deficit_weight"],
                "raw_wca_weight": raw_weights[index],
                "capped_weight": capped_weights[index],
                "previous_weight": safe_previous[index],
                "smoothed_weight": final_weights[index],
                "final_aggregation_weight": final_weights[index],
                "wca_alpha": min(max(float(alpha), 0.0), 1.0),
                "wca_beta": beta,
                "wca_max_weight": max_weight,
            }
        )

    return final_weights, rows
