import numpy as np
import pandas as pd
import pytest

from fraud_detection.evaluation.error_analysis import (
    amount_band_summary,
    amount_bands,
    categorical_slices,
    disagreement_summary,
    error_classes,
    join_decisions,
    representative_cases,
    risk_band_summary,
    risk_bands,
)
from fraud_detection.evaluation.explainability import (
    aggregate_shap,
    feature_family,
    top_local_drivers,
)


def frame():
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "isFraud": [1, 0, 1, 0],
            "TransactionAmt": [10.0, 20.0, 100.0, 200.0],
            "calibrated_probability": [0.9, 0.8, 0.1, 0.01],
            "decision": ["BLOCK", "REVIEW", "APPROVE", "APPROVE"],
            "ProductCD": ["W", "W", "C", "C"],
        }
    )


def test_error_classes_and_bands():
    f = frame()
    assert error_classes(f.isFraud, f.calibrated_probability).tolist() == ["TP", "FP", "FN", "TN"]
    assert amount_bands(f.TransactionAmt, [0, 50, np.inf]).tolist() == [0, 0, 1, 1]
    assert list(risk_bands(f.calibrated_probability)) == ["40%+", "40%+", "10–20%", "1–2%"]


def test_feature_families_and_shap():
    names = [
        "TransactionAmt",
        "id_01",
        "card1_frequency",
        "card1_previous_avg_amount",
        "card1_card2_addr1_previous_tx_count",
        "card1_card2_addr1_target_encoded_fraud_rate",
    ]
    assert [feature_family(name) for name in names] == [
        "raw_transaction",
        "raw_identity",
        "frequency_encoding",
        "historical_amount",
        "composite_identity",
        "composite_temporal_te",
    ]
    values = np.array([[1, -2, 3, 4, 5, 6], [-1, 2, -3, 4, -5, 6.0]])
    original = values.copy()
    feature, family = aggregate_shap(values, names)
    assert feature.iloc[0]["feature"] == names[-1]
    assert family.mean_abs_shap.sum() == pytest.approx(np.abs(values).mean(axis=0).sum())
    assert sum(family.share) == pytest.approx(1)
    assert top_local_drivers(values[0], names, count=1)[0][0] == names[-1]
    np.testing.assert_array_equal(values, original)


def test_amount_and_risk_summaries_reconcile_without_mutation():
    f = frame()
    original = f.copy(deep=True)
    summary = amount_band_summary(f, [0, 50, np.inf])
    assert summary["FN"].sum() == 1
    assert summary.missed_fraud_amount.sum() == 100
    assert summary.fraud_amount.sum() == 110
    risk = risk_band_summary(f)
    assert risk["count"].sum() == 4
    assert risk.fraud_amount.sum() == 110
    pd.testing.assert_frame_equal(f, original)


def test_support_filter_and_disagreement():
    f = frame()
    slices = categorical_slices(f, "ProductCD", min_support=2)
    assert set(slices.category) == {"W", "C"}
    assert categorical_slices(f, "ProductCD", min_support=3).empty
    metrics = disagreement_summary(
        f.isFraud,
        f.TransactionAmt,
        [0.9, 0.7, 0.2, 0.01],
        [0.8, 0.9, 0.1, 0.02],
        f.calibrated_probability,
    )
    assert metrics["high_gap_count"] >= 1
    assert 0 <= metrics["high_gap_fraud_rate"] <= 1


def test_case_selection_and_join_deterministic():
    f = frame()
    a = representative_cases(f, high_value_amount=50)
    b = representative_cases(f, high_value_amount=50)
    assert a == b
    assert a["high_value_fraud_miss"] == 2
    decisions = f[["TransactionID", "decision"]].copy()
    source = f.drop(columns="decision")
    original = source.copy(deep=True)
    joined = join_decisions(source, decisions)
    assert joined.decision.tolist() == f.decision.tolist()
    pd.testing.assert_frame_equal(source, original)
    with pytest.raises(ValueError, match="align"):
        join_decisions(source, decisions.iloc[:-1])
