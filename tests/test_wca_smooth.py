"""Sanity checks for Phase 4D WCA-Smooth weights."""

from methods.wca_smooth import compute_wca_smooth_aggregation_weights


def test_wca_smooth_round_one_uses_data_weights() -> None:
    weights, rows = compute_wca_smooth_aggregation_weights(
        client_num_samples=[10, 30],
        client_validation_dice=None,
        previous_weights=None,
        alpha=0.5,
        beta=0.5,
        max_weight=0.5,
    )

    assert weights == [0.25, 0.75]
    assert rows[0]["validation_dice_used"] == ""
    assert abs(sum(weights) - 1.0) < 1e-9


def test_wca_smooth_weights_are_normalized() -> None:
    weights, rows = compute_wca_smooth_aggregation_weights(
        client_num_samples=[10, 10, 10],
        client_validation_dice=[0.9, 0.2, 0.8],
        previous_weights=[1.0 / 3.0] * 3,
        alpha=0.5,
        beta=0.5,
        max_weight=0.5,
    )

    assert abs(sum(weights) - 1.0) < 1e-9
    assert all(row["final_aggregation_weight"] >= 0.0 for row in rows)
    assert rows[1]["raw_wca_weight"] > rows[0]["raw_wca_weight"]


def test_wca_smooth_handles_equal_dice() -> None:
    weights, rows = compute_wca_smooth_aggregation_weights(
        client_num_samples=[2, 3, 5],
        client_validation_dice=[0.7, 0.7, 0.7],
        previous_weights=None,
        alpha=0.5,
        beta=0.5,
        max_weight=0.5,
    )

    assert weights == [0.2, 0.3, 0.5]
    assert all(row["performance_deficit"] == 0.0 for row in rows)
