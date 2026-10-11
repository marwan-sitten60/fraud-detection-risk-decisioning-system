"""Small synthetic checks for deterministic operational rule behavior."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.decisioning.rules import (
    Rule,
    RulesEngine,
    RuleThresholds,
    apply_review_capacity,
    derive_thresholds,
)


@pytest.fixture
def thresholds():
    return RuleThresholds(100, 500, 0.9, 0.1, 0.2, 0.01, 25)


def frame(**updates):
    values = {
        "TransactionID": [1],
        "calibrated_probability": [0.95],
        "TransactionAmt": [150.0],
        "lightgbm_score": [0.4],
        "xgboost_score": [0.9],
        "composite_frequency": [0.0],
        "base_decision": ["APPROVE"],
    }
    values.update({key: [value] for key, value in updates.items()})
    return pd.DataFrame(values)


def costs(ids):
    return pd.DataFrame(
        {
            "TransactionID": ids,
            "expected_cost_approve": [10.0] * len(ids),
            "expected_cost_review": [1.0] * len(ids),
            "expected_cost_block": [15.0] * len(ids),
            "review_priority_benefit": [9.0] * len(ids),
        }
    )


@pytest.mark.parametrize("base", ["APPROVE", "REVIEW"])
def test_hard_block_overrides_base_and_records_all_reasons(thresholds, base):
    result = RulesEngine(thresholds).evaluate(frame(base_decision=base)).iloc[0]
    assert result.final_decision == "BLOCK"
    assert result.highest_priority_rule == "R01"
    assert "R01" in result.triggered_rule_ids
    assert "EXTREME_RISK_HIGH_AMOUNT" in result.rule_reasons


def test_soft_review_escalates_only_approve(thresholds):
    engine = RulesEngine(thresholds)
    approved = engine.evaluate(frame(calibrated_probability=0.2, TransactionAmt=30)).iloc[0]
    blocked = engine.evaluate(
        frame(calibrated_probability=0.2, TransactionAmt=30, base_decision="BLOCK")
    ).iloc[0]
    assert approved.final_decision == "REVIEW"
    assert set(approved.triggered_rule_ids) == {"R02", "R03"}
    assert blocked.final_decision == "BLOCK"


def test_priority_tie_break_and_serializable_config(thresholds):
    a = Rule("B", "B", "description", "SOFT", 7, "REVIEW", "B_REASON", "rare_identity_review")
    b = Rule("A", "A", "description", "SOFT", 7, "REVIEW", "A_REASON", "rare_identity_review")
    engine = RulesEngine(thresholds, (a, b))
    result = engine.evaluate(frame(calibrated_probability=0.2, TransactionAmt=30)).iloc[0]
    assert result.highest_priority_rule == "A"
    assert result.triggered_rule_ids == ("A", "B")
    assert engine.configuration()["thresholds"] == asdict(thresholds)


def test_hard_precedence_beats_soft_even_with_lower_numeric_priority(thresholds):
    hard = Rule(
        "H", "Hard", "description", "HARD", 1, "BLOCK", "HARD_REASON", "extreme_risk_amount"
    )
    soft = Rule(
        "S", "Soft", "description", "SOFT", 999, "REVIEW", "SOFT_REASON", "disagreement_review"
    )
    result = RulesEngine(thresholds, (soft, hard)).evaluate(frame()).iloc[0]
    assert result.highest_priority_rule == "H"
    assert result.final_decision == "BLOCK"


def test_probability_and_source_unchanged_and_labels_not_required(thresholds):
    source = frame()
    copy = source.copy(deep=True)
    result = RulesEngine(thresholds).evaluate(source)
    pd.testing.assert_frame_equal(source, copy)
    assert result.calibrated_probability.iloc[0] == source.calibrated_probability.iloc[0]
    assert "isFraud" not in source
    pd.testing.assert_frame_equal(result, RulesEngine(thresholds).evaluate(source))


def test_disagreement_uses_absolute_component_gap(thresholds):
    source = frame(
        calibrated_probability=0.2,
        TransactionAmt=10,
        lightgbm_score=0.7,
        xgboost_score=0.1,
        composite_frequency=1.0,
    )
    assert RulesEngine(thresholds).evaluate(source).triggered_rule_ids.iloc[0] == ("R02",)


def test_unseen_composite_identity_triggers_review(thresholds):
    result = (
        RulesEngine(thresholds)
        .evaluate(
            frame(
                calibrated_probability=0.2,
                TransactionAmt=30,
                lightgbm_score=0.2,
                xgboost_score=0.2,
                composite_frequency=0.0,
            )
        )
        .iloc[0]
    )
    assert result.triggered_rule_ids == ("R03",)
    assert result.final_decision == "REVIEW"


def test_thresholds_use_only_given_train_signals():
    oof = pd.DataFrame(
        {
            "calibrated_probability": [0.01, 0.02, 0.03, 0.04],
            "lightgbm_score": [0.1, 0.2, 0.3, 0.4],
            "xgboost_score": [0.2, 0.1, 0.3, 0.5],
            "composite_frequency": [0, 0.01, 0.02, 0.03],
            "isFraud": [1, 0, 0, 1],
        }
    )
    first = derive_thresholds([10, 20, 30, 40], oof)
    oof["isFraud"] = 1 - oof.isFraud
    assert first == derive_thresholds([10, 20, 30, 40], oof)
    assert first.high_amount == np.quantile([10, 20, 30, 40], 0.95)


def test_capacity_never_downgrades_hard_block(thresholds):
    engine = RulesEngine(thresholds)
    source = pd.concat(
        [frame(), frame(TransactionID=2, calibrated_probability=0.2, TransactionAmt=30)],
        ignore_index=True,
    )
    results = engine.evaluate(source)
    capped = apply_review_capacity(results, costs([1, 2]), 0.0)
    assert capped.final_decision.tolist() == ["BLOCK", "APPROVE"]
    assert capped.review_capacity_overflowed.tolist() == [False, True]
    pd.testing.assert_frame_equal(results, engine.evaluate(source))


def test_module_has_no_holdout_dependency():
    import fraud_detection.decisioning.rules as module

    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    assert "holdout" not in source


def test_external_predicate_can_be_injected_without_model_changes(thresholds):
    rule = Rule(
        "EXT",
        "External policy",
        "verified upstream signal",
        "HARD",
        200,
        "BLOCK",
        "EXTERNAL_MATCH",
        "verified_match",
    )
    engine = RulesEngine(
        thresholds,
        (rule,),
        {"verified_match": lambda data, _thresholds: data.TransactionID.eq(1).to_numpy()},
    )
    result = engine.evaluate(frame(calibrated_probability=0.01, TransactionAmt=10)).iloc[0]
    assert result.final_decision == "BLOCK"
    assert result.rule_reasons == ("EXTERNAL_MATCH",)
