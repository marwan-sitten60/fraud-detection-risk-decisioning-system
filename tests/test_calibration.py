import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from fraud_detection.models.calibration import (
    Calibrator,
    evaluate,
    probabilities,
    reliability_bins,
    select_method,
    temporal_selection,
    validate_oof,
)


def sample_oof():
    p = np.array([0.01, 0.2, 0.8, 0.9] * 3)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(12),
            "fold": np.repeat([1, 2, 3], 4),
            "y_true": [0, 0, 1, 1] * 3,
            "lightgbm_score": p,
            "xgboost_score": p,
        }
    )


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrators_deterministic_bounded_and_input_unchanged(method):
    p = np.array([0.01, 0.2, 0.8, 0.9])
    original = p.copy()
    y = np.array([0, 0, 1, 1])
    a = Calibrator(method).fit(p, y).transform(p)
    b = Calibrator(method).fit(p, y).transform(p)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(p, original)
    assert ((a >= 0) & (a <= 1)).all()


def test_future_labels_do_not_influence_earlier_fit():
    frame = sample_oof()
    first, _ = temporal_selection(frame)
    changed = frame.copy(deep=True)
    changed.loc[changed.fold == 3, "y_true"] = 1 - changed.loc[changed.fold == 3, "y_true"]
    second, _ = temporal_selection(changed)
    for method in ("platt", "isotonic"):
        a = first.loc[(first.method == method) & (first.evaluation_fold == 2), "brier"].iloc[0]
        b = second.loc[(second.method == method) & (second.evaluation_fold == 2), "brier"].iloc[0]
        assert a == b


def test_order_and_predictor_guard():
    frame = sample_oof()
    assert len(validate_oof(frame)) == 12
    with pytest.raises(ValueError, match="ordered"):
        validate_oof(frame.iloc[::-1])
    with pytest.raises(ValueError, match="Missing"):
        validate_oof(frame.drop(columns="xgboost_score"))
    frame["isFraud"] = 1 - frame.y_true
    np.testing.assert_array_equal(validate_oof(frame), validate_oof(frame.drop(columns="isFraud")))


def test_invalid_probabilities_and_reference_metrics():
    with pytest.raises(ValueError, match="\[0, 1\]"):
        probabilities([0.2, 1.1])
    y = [0, 1, 0, 1]
    p = [0.1, 0.9, 0.3, 0.7]
    result = evaluate(y, p, n_bins=2)
    assert result["brier"] == pytest.approx(brier_score_loss(y, p))
    assert result["ece"] == pytest.approx(0.2)
    assert reliability_bins(y, p, n_bins=3)["count"].sum() == 4


def test_selection_requires_only_oof():
    results, predictions = temporal_selection(sample_oof())
    assert set(predictions) == {"raw", "platt", "isotonic"}
    assert select_method(results) in predictions
    assert len(results) == 9
