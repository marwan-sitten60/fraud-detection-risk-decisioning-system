import json
from pathlib import Path

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


@pytest.mark.parametrize("delay", [0.0, 7.0])
def test_projected_encoder_matches_saved_legacy_behavior_exactly(delay: float) -> None:
    """The reference was captured from the unmodified encoder on a wide frame."""
    frame = pd.DataFrame(
        {
            "TransactionID": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "TransactionDT": [
                0,
                0,
                86400,
                86400,
                172800,
                604800,
                604801,
                1209600,
                1209600,
                1296000,
            ],
            "card1": ["A", "A", "A", "B", "A", None, "C", "A", "A", "Z"],
            "card2": [1, 1, 1, 2, 1, None, 3, 1, 1, 9],
            "addr1": [10, 10, 10, 20, 10, None, 30, 10, 10, 99],
            "P_emaildomain": ["a", "a", "a", "b", "a", None, "c", "a", "a", "z"],
            "R_emaildomain": ["r", "r", "r", "s", "r", None, "t", "r", "r", "u"],
            "DeviceInfo": ["d", "d", "d", "e", "d", None, "f", "d", "d", "g"],
            "ProductCD": ["W", "W", "W", "C", "W", None, "R", "W", "W", "H"],
            "isFraud": [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        },
        index=[90, 70, 80, 30, 60, 40, 20, 50, 10, 0],
    )
    unused = pd.DataFrame(
        {f"unused_{j}": np.arange(len(frame), dtype=np.float64) + j for j in range(300)},
        index=frame.index,
    )
    wide = pd.concat([frame, unused], axis=1)
    reference = json.loads(
        (Path(__file__).parent / "fixtures/target_encoding_legacy_reference.json").read_text()
    )[str(delay)]
    encoder = TemporalTargetEncoder(label_delay_days=delay)
    train = encoder.fit_transform_features(wide.iloc[:7])
    future = encoder.transform_features(wide.iloc[[9, 7, 8]].drop(columns="isFraud"))
    for actual, expected in ((train, reference["train"]), (future, reference["future"])):
        expected_frame = pd.DataFrame(
            expected["values"], index=expected["index"], columns=expected["columns"]
        ).astype(dict(zip(expected["columns"], expected["dtypes"], strict=True)))
        pd.testing.assert_frame_equal(actual, expected_frame, check_exact=True)
    assert str(encoder.training_statistics_) == reference["fit_statistics"]
    assert str(encoder.last_transform_statistics_) == reference["transform_statistics"]


def test_encoder_copy_excludes_unrelated_wide_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    wide = pd.concat(
        [transactions(), pd.DataFrame({f"unused_{j}": np.zeros(6) for j in range(300)})],
        axis=1,
    )
    original_copy = pd.DataFrame.copy
    copied_widths = []

    def checked_copy(self: pd.DataFrame, *args: object, **kwargs: object) -> pd.DataFrame:
        copied_widths.append(len(self.columns))
        assert not any(column.startswith("unused_") for column in self.columns)
        return original_copy(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "copy", checked_copy)
    TemporalTargetEncoder().fit_transform_features(wide)
    assert copied_widths
