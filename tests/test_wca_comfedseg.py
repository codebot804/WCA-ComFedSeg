"""Sanity checks for WCA-ComFedSeg Phase 4A aggregation weights."""

from methods.wca_comfedseg import compute_wca_aggregation_weights


def test_round_one_wca_weights_fallback_to_data_weights() -> None:
    weights, rows = compute_wca_aggregation_weights([10, 30], None, alpha=0.5)

    assert weights == [0.25, 0.75]
    assert rows[0]["validation_dice_used"] == ""
    assert sum(weights) == 1.0


def test_lower_dice_client_gets_higher_deficit_weight() -> None:
    weights, rows = compute_wca_aggregation_weights([10, 10, 10], [0.9, 0.4, 0.8], alpha=0.5)

    assert rows[1]["performance_deficit"] > rows[0]["performance_deficit"]
    assert rows[1]["final_aggregation_weight"] > rows[0]["final_aggregation_weight"]
    assert abs(sum(weights) - 1.0) < 1e-9


def test_equal_dice_falls_back_to_data_weights() -> None:
    weights, rows = compute_wca_aggregation_weights([2, 3, 5], [0.7, 0.7, 0.7], alpha=0.5)

    assert weights == [0.2, 0.3, 0.5]
    assert all(row["performance_deficit"] == 0.0 for row in rows)
