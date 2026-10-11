import pandas as pd
import pytest

from fraud_detection.decisioning.costs import (
    BusinessCostConfig,
    expected_action_costs,
    scenario_profiles,
)
from fraud_detection.decisioning.evaluation import evaluate_policy
from fraud_detection.decisioning.policy import decide

CFG = BusinessCostConfig(1, 3, 0.8, 1, 25, 0.5)


def sample():
    return pd.DataFrame(
        {
            "TransactionID": [3, 1, 2, 4],
            "calibrated_probability": [0, 0.1, 0.8, 1],
            "TransactionAmt": [100.0, 100.0, 100.0, 100.0],
        }
    )


def test_cost_formulas():
    a, r, b = expected_action_costs([0.2], [100], CFG)
    assert a[0] == pytest.approx(20)
    assert r[0] == pytest.approx(3 + 0.2 * 0.2 * 100 + 0.8)
    assert b[0] == pytest.approx(0.8 * 25 + 0.5)


def test_decisions_monotonic_exposure_and_inputs():
    frame = sample()
    original = frame.copy(deep=True)
    result = decide(frame, CFG)
    assert result.decision.tolist() == ["APPROVE", "REVIEW", "BLOCK", "BLOCK"]
    pd.testing.assert_frame_equal(frame, original)
    a, _, _ = expected_action_costs([0.2, 0.2], [100, 200], CFG)
    assert a[1] > a[0]
    with pytest.raises(ValueError, match="calibrated probability"):
        decide(frame.rename(columns={"calibrated_probability": "raw_score"}), CFG)


@pytest.mark.parametrize("probability,amount", [(1.1, 10), (-0.1, 10), (0.5, -1)])
def test_invalid_inputs(probability, amount):
    with pytest.raises(ValueError):
        expected_action_costs([probability], [amount], CFG)


def test_invalid_costs():
    with pytest.raises(ValueError):
        BusinessCostConfig(1, -1, 0.8, 1, 25)
    with pytest.raises(ValueError):
        BusinessCostConfig(1, 1, 1.1, 1, 25)
    with pytest.raises(ValueError, match="manual_review_cost"):
        BusinessCostConfig(1, "invalid", 0.8, 1, 25)
    assert len(scenario_profiles()) == 3


def test_capacity_and_determinism():
    frame = sample()
    a = decide(frame, CFG, maximum_review_rate=0, overflow="cheapest")
    b = decide(frame, CFG, maximum_review_rate=0, overflow="cheapest")
    pd.testing.assert_frame_equal(a, b)
    assert a.review_overflowed.sum() == 1
    assert (a.decision == "REVIEW").sum() == 0
    assert a.loc[1, "decision"] == "APPROVE"
    assert decide(frame, CFG, maximum_review_rate=0.25).decision.eq("REVIEW").sum() <= 1


def test_evaluation_reconciles():
    result = decide(sample(), CFG)
    metrics = evaluate_policy(result, [0, 1, 1, 0], CFG)
    assert sum(metrics[f"{x}_count"] for x in ("approve", "review", "block")) == 4
    assert sum(metrics[f"fraud_transactions_{x}"] for x in ("approved", "reviewed", "blocked")) == 2
    assert (
        sum(metrics[f"fraudulent_amount_{x}"] for x in ("approved", "reviewed", "blocked")) == 200
    )
    assert sum(
        metrics[f"legitimate_{x}_rate"] for x in ("approve", "review", "block")
    ) == pytest.approx(1)
    assert metrics["false_decline_count"] == 1
    assert metrics["approve_all_fraud_transactions_missed"] == 2
    assert metrics["modeled_total_cost"] <= metrics["approve_all_modeled_cost"]


def test_capacity_keeps_highest_review_benefit():
    frame = pd.DataFrame(
        {
            "TransactionID": [20, 10],
            "calibrated_probability": [0.1, 0.2],
            "TransactionAmt": [100.0, 100.0],
        }
    )
    result = decide(frame, CFG, maximum_review_rate=0.5)
    assert result.loc[1, "decision"] == "REVIEW"
    assert result.loc[0, "review_overflowed"]
    assert result.loc[0, "decision"] == "APPROVE"
