from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fraud_detection.decisioning.costs import scenario_profiles
from fraud_detection.decisioning.rules import RulesEngine, RuleThresholds
from fraud_detection.inference.runtime import FrozenFraudInferenceRuntime


class FakeFeatureState:
    raw_columns = (
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "card1",
        "addr1",
        "P_emaildomain",
    )

    def __init__(self) -> None:
        self.preprocessor = SimpleNamespace(categorical_features_={"P_emaildomain"})
        self.composite_engineer = SimpleNamespace(frequency_maps_={"card1_addr1_p_emaildomain": {}})

    def transform(self, rows: pd.DataFrame, *, update_state: bool = True) -> np.ndarray:
        return rows.TransactionAmt.to_numpy(dtype=np.float32).reshape(-1, 1)


class FakeLightGBM:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return features[:, 0].astype(float) / 1000


class FakeXGBoost:
    def predict(self, dmatrix) -> np.ndarray:
        return dmatrix.get_data().toarray()[:, 0].astype(np.float32) / 2000


class FakeCalibrator:
    def transform(self, scores: np.ndarray) -> np.ndarray:
        return scores / 2


def runtime() -> FrozenFraudInferenceRuntime:
    thresholds = RuleThresholds(500, 1000, 0.9, 0.1, 0.5, 0, 100)
    return FrozenFraudInferenceRuntime(
        bundle_path=None,
        manifest={"bundle_version": "frozen_v1"},
        feature_state=FakeFeatureState(),
        lightgbm=FakeLightGBM(),
        xgboost=FakeXGBoost(),
        calibrator=FakeCalibrator(),
        costs=scenario_profiles()["Balanced"],
        rules=RulesEngine(thresholds),
    )


def rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [10, 11],
            "TransactionAmt": [100.0, 200.0],
            "card1": [10, 10],
            "addr1": [20, 20],
            "P_emaildomain": ["a", "a"],
        }
    )


def test_runtime_scores_are_label_free_frozen_blend_and_rules_default_off() -> None:
    source = rows()
    original = source.copy(deep=True)
    output = runtime().predict_batch(source)
    pd.testing.assert_frame_equal(source, original)
    assert "isFraud" not in output
    assert "final_decision" not in output
    np.testing.assert_array_equal(
        output.ensemble_score,
        0.8 * output.lightgbm_score + 0.2 * output.xgboost_score,
    )
    assert output.calibrated_probability.between(0, 1).all()
    assert {"expected_cost_approve", "expected_cost_review", "expected_cost_block"} <= set(output)
    with pytest.raises(ValueError, match="target fields"):
        runtime().predict_batch(source.assign(isFraud=[0, 1]))


def test_optional_rules_preserve_probability_and_one_matches_batch() -> None:
    source = rows()
    primary = runtime().predict_batch(source)
    optional = runtime().predict_batch(source, include_rules=True)
    np.testing.assert_array_equal(primary.calibrated_probability, optional.calibrated_probability)
    assert "final_decision" in optional
    independent = runtime().predict_batch(source, include_rules=True)
    pd.testing.assert_frame_equal(optional, independent)
    sequential = [runtime().predict_one(row) for row in source.to_dict(orient="records")]
    np.testing.assert_array_equal(
        np.asarray([row["ensemble_score"] for row in sequential]), primary.ensemble_score
    )
