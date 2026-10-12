"""Synthetic terminal-evaluation plumbing checks."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.decisioning.costs import scenario_profiles
from fraud_detection.decisioning.policy import decide
from fraud_detection.evaluation.holdout import diagnostic_cutoff_metrics, prediction_artifact


def test_prediction_artifact_requires_no_labels_and_preserves_inputs():
    features = pd.DataFrame(
        {"TransactionID": [1, 2], "TransactionDT": [10, 20], "TransactionAmt": [20.0, 200.0]}
    )
    copy = features.copy(deep=True)
    decisions = decide(
        pd.DataFrame(
            {
                "TransactionID": [1, 2],
                "TransactionAmt": [20.0, 200.0],
                "calibrated_probability": [0.1, 0.8],
            }
        ),
        scenario_profiles()["Balanced"],
    )
    artifact = prediction_artifact(
        features, [0.2, 0.7], [0.3, 0.6], [0.22, 0.68], [0.1, 0.8], decisions
    )
    assert "isFraud" not in artifact
    assert artifact.base_decision.tolist() == decisions.decision.tolist()
    pd.testing.assert_frame_equal(features, copy)
    pd.testing.assert_frame_equal(
        artifact,
        prediction_artifact(features, [0.2, 0.7], [0.3, 0.6], [0.22, 0.68], [0.1, 0.8], decisions),
    )


def test_prediction_artifact_rejects_misaligned_decisions():
    features = pd.DataFrame({"TransactionID": [1], "TransactionDT": [10], "TransactionAmt": [20.0]})
    decisions = pd.DataFrame({"TransactionID": [2], "decision": ["APPROVE"]})
    with pytest.raises(ValueError, match="align"):
        prediction_artifact(features, [0.1], [0.1], [0.1], [0.1], decisions)


def test_diagnostic_cutoff_known_counts():
    result = diagnostic_cutoff_metrics([1, 0, 1, 0], [0.8, 0.6, 0.2, 0.1])
    assert {key: result[key] for key in ("tp", "fp", "fn", "tn")} == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
    }
    assert np.isclose(result["precision"], 0.5)
    assert np.isclose(result["recall"], 0.5)
    assert np.isclose(result["f1"], 0.5)
