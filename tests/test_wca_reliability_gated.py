"""Sanity checks for Phase 4F Reliability-Gated WCA weights."""

from methods.wca_reliability_gated import (
    compute_reliability_gated_wca_weights,
    compute_reliability_scores,
)


def test_wca_rg_round_one_uses_data_weights() -> None:
    weights, rows = compute_reliability_gated_wca_weights(
        client_num_samples=[10, 30],
        previous_validation_dice=None,
        current_validation_dice=[0.4, 0.8],
        alpha=0.5,
        min_gate=0.2,
    )

    assert weights == [0.25, 0.75]
    assert rows[0]["previous_validation_dice"] == ""
    assert rows[0]["current_validation_dice"] == 0.4
    assert abs(sum(weights) - 1.0) < 1e-9


def test_wca_rg_rewards_weak_improving_client_more_than_non_improving_weak_client() -> None:
    weights, rows = compute_reliability_gated_wca_weights(
        client_num_samples=[10, 10, 10],
        previous_validation_dice=[0.8, 0.2, 0.45],
        current_validation_dice=[0.82, 0.25, 0.43],
        alpha=0.5,
        min_gate=0.2,
    )

    assert rows[1]["performance_deficit"] > 0.0
    assert rows[2]["performance_deficit"] > 0.0
    assert rows[1]["positive_improvement"] > 0.0
    assert rows[2]["positive_improvement"] == 0.0
    assert rows[1]["reliability_gate"] > rows[2]["reliability_gate"]
    assert weights[1] > weights[2]
    assert abs(sum(weights) - 1.0) < 1e-9


def test_wca_rg_all_nonpositive_improvements_use_neutral_reliability_scores() -> None:
    scores, improvements, positives, has_previous = compute_reliability_scores(
        previous_validation_dice=[0.6, 0.5, 0.4],
        current_validation_dice=[0.55, 0.5, 0.39],
    )

    assert has_previous
    assert improvements == [-0.04999999999999993, 0.0, -0.010000000000000009]
    assert positives == [0.0, 0.0, 0.0]
    assert all(abs(score - (1.0 / 3.0)) < 1e-9 for score in scores)


def test_wca_rg_uses_data_weights_when_no_weak_client_is_improving() -> None:
    weights, rows = compute_reliability_gated_wca_weights(
        client_num_samples=[10, 10, 10],
        previous_validation_dice=[0.8, 0.6, 0.4],
        current_validation_dice=[0.82, 0.62, 0.38],
        alpha=0.5,
        min_gate=0.2,
    )

    assert rows[2]["performance_deficit"] > 0.0
    assert rows[2]["positive_improvement"] == 0.0
    assert weights == [1.0 / 3.0] * 3


def test_wca_rg_handles_invalid_dice_without_nan_weights() -> None:
    weights, rows = compute_reliability_gated_wca_weights(
        client_num_samples=[5, 5, 10],
        previous_validation_dice=[0.2, None, 0.7],
        current_validation_dice=[0.3, float("nan"), 0.65],
        alpha=0.5,
        min_gate=0.2,
    )

    assert abs(sum(weights) - 1.0) < 1e-9
    assert all(weight >= 0.0 for weight in weights)
    assert rows[1]["current_validation_dice"] == ""
