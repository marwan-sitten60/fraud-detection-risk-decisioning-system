"""Tests for leakage-safe baseline comparison helpers."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.models import TrainingOnlyPreprocessor, evaluate_probabilities


def test_training_only_preprocessor_handles_missing_and_unseen_categories() -> None:
    """Validation-only categories should map safely without changing feature count."""
    train = pd.DataFrame({"amount": [1.0, None, 3.0], "category": ["a", "a", None]})
    validation = pd.DataFrame({"amount": [None, 5.0], "category": ["new", None]})
    preprocessor = TrainingOnlyPreprocessor()

    train_matrix = preprocessor.fit_transform(train)
    validation_matrix = preprocessor.transform(validation)

    assert train_matrix.shape == (3, 2)
    assert validation_matrix.shape == (2, 2)
    assert np.isfinite(train_matrix).all()
    assert np.isfinite(validation_matrix).all()
    assert validation_matrix[0, 1] == pytest.approx(0.0)


def test_evaluate_probabilities_reports_ranking_and_threshold_metrics() -> None:
    """Metric helper should evaluate probabilities at the documented threshold."""
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.4, 0.9])

    metrics = evaluate_probabilities(target, probabilities)

    assert metrics.pr_auc == pytest.approx(5 / 6)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.confusion_matrix == ((1, 1), (1, 1))
