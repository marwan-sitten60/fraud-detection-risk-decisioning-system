import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import (
    TARGET_ENCODING_FEATURES,
    TemporalTargetEncoder,
    create_composite_keys,
)


def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [100, 100, 200, 300, 400, 500],
            "card1": ["A", "B", "A", "A", "C", None],
            "card2": [1, 2, 1, 1, 3, None],
            "addr1": [10, 20, 10, 10, 30, None],
            "P_emaildomain": ["p1", "p2", "p1", "p1", "p3", None],
            "R_emaildomain": ["r1", "r2", "r1", "r1", "r3", None],
            "DeviceInfo": ["mobile", "desktop", "mobile", "mobile", "tablet", None],
            "ProductCD": ["W", "C", "W", "W", "R", None],
            "isFraud": [1, 0, 0, 1, 0, 1],
        }
    )


def test_own_target_does_not_change_own_encoding() -> None:
    original = transactions()
    changed = original.copy()
    changed.loc[2, "isFraud"] = 1

    first = TemporalTargetEncoder().fit_transform_features(original)
    second = TemporalTargetEncoder().fit_transform_features(changed)

    pd.testing.assert_series_equal(first.loc[2], second.loc[2])


def test_future_target_changes_do_not_affect_past_encodings() -> None:
    original = transactions()
    changed = original.copy()
    changed.loc[5, "isFraud"] = 0

    first = TemporalTargetEncoder().fit_transform_features(original)
    second = TemporalTargetEncoder().fit_transform_features(changed)

    pd.testing.assert_frame_equal(first.iloc[:5], second.iloc[:5])


def test_evaluation_labels_never_affect_frozen_features() -> None:
    frame = transactions()
    fit = frame.iloc[:4]
    evaluation = frame.iloc[4:].copy()
    changed = evaluation.copy()
    changed["isFraud"] = 1 - changed["isFraud"]
    encoder = TemporalTargetEncoder().fit(fit)

    first = encoder.transform_features(evaluation)
    second = encoder.transform_features(changed)

    pd.testing.assert_frame_equal(first, second)


def test_same_timestamp_rows_cannot_use_each_others_labels() -> None:
    original = transactions().iloc[:3].copy()
    entity_columns = [
        "card1",
        "card2",
        "addr1",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceInfo",
        "ProductCD",
    ]
    original.loc[1, entity_columns] = original.loc[0, entity_columns].to_numpy()
    changed = original.copy()
    changed.loc[0, "isFraud"] = 0

    first = TemporalTargetEncoder().fit_transform_features(original)
    second = TemporalTargetEncoder().fit_transform_features(changed)

    assert first.loc[1, "card1_target_encoded_fraud_rate"] == pytest.approx(0.5)
    pd.testing.assert_series_equal(first.loc[1], second.loc[1])


def test_smoothing_formula_is_correct() -> None:
    frame = transactions().iloc[:3]

    transformed = TemporalTargetEncoder(alpha=50).fit_transform_features(frame)
    expected = (1 + 50 * 0.5) / (1 + 50)

    assert transformed.loc[2, "card1_target_encoded_fraud_rate"] == pytest.approx(expected)


def test_unseen_evaluation_values_use_fit_global_prior() -> None:
    frame = transactions()
    encoder = TemporalTargetEncoder().fit(frame.iloc[:4])
    unseen = frame.iloc[[4]].copy()
    unseen.loc[:, ["card1", "card2", "addr1"]] = ["NEW", 99, 999]

    transformed = encoder.transform_features(unseen)

    assert transformed.loc[unseen.index[0], "card1_target_encoded_fraud_rate"] == pytest.approx(
        encoder.global_fraud_rate_
    )


def test_missing_values_are_deterministic() -> None:
    frame = transactions()
    first = TemporalTargetEncoder().fit_transform_features(frame)
    second = TemporalTargetEncoder().fit_transform_features(frame.copy())

    pd.testing.assert_series_equal(first.iloc[-1], second.iloc[-1])
    assert np.isfinite(first.iloc[-1]).all()


def test_composite_key_generation_is_unchanged() -> None:
    frame = transactions()

    before = create_composite_keys(frame)
    TemporalTargetEncoder().fit_transform_features(frame)
    after = create_composite_keys(frame)

    pd.testing.assert_frame_equal(before, after)


def test_input_is_not_mutated_and_target_is_excluded() -> None:
    frame = transactions()
    original = frame.copy(deep=True)

    transformed = TemporalTargetEncoder().fit_transform_features(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert "isFraud" not in transformed.columns
    assert tuple(transformed.columns) == TARGET_ENCODING_FEATURES
    assert transformed.shape[1] == 10


def test_appended_future_rows_do_not_change_existing_encodings() -> None:
    original = transactions().iloc[:5]
    appended = transactions()
    changed = appended.copy()
    changed.loc[5, "isFraud"] = 0

    original_features = TemporalTargetEncoder().fit_transform_features(original)
    appended_features = TemporalTargetEncoder().fit_transform_features(appended)
    changed_features = TemporalTargetEncoder().fit_transform_features(changed)

    pd.testing.assert_frame_equal(original_features, appended_features.iloc[:5])
    pd.testing.assert_frame_equal(original_features, changed_features.iloc[:5])


def test_earliest_timestamp_uses_fixed_initial_prior() -> None:
    transformed = TemporalTargetEncoder(initial_prior=0.5).fit_transform_features(transactions())

    assert transformed.iloc[:2].eq(0.5).all().all()
