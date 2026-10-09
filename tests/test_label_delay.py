import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import TemporalTargetEncoder

DAY = 86_400


def delayed_transactions(
    times: list[int] | None = None,
    targets: list[int] | None = None,
) -> pd.DataFrame:
    times = times or [0, 3 * DAY, 7 * DAY, 8 * DAY]
    targets = targets or [1, 0, 0, 1]
    row_count = len(times)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, row_count + 1),
            "TransactionDT": times,
            "card1": ["A"] * row_count,
            "card2": [1] * row_count,
            "addr1": [10] * row_count,
            "P_emaildomain": ["p1"] * row_count,
            "R_emaildomain": ["r1"] * row_count,
            "DeviceInfo": ["mobile"] * row_count,
            "ProductCD": ["W"] * row_count,
            "isFraud": targets,
        }
    )


def test_zero_day_delay_is_backward_compatible() -> None:
    frame = delayed_transactions()

    existing = TemporalTargetEncoder().fit_transform_features(frame)
    explicit_zero = TemporalTargetEncoder(label_delay_days=0).fit_transform_features(frame)

    pd.testing.assert_frame_equal(existing, explicit_zero)


def test_immature_historical_label_does_not_affect_current_encoding() -> None:
    original = delayed_transactions(times=[0, 6 * DAY], targets=[1, 0])
    changed = original.copy()
    changed.loc[0, "isFraud"] = 0

    first = TemporalTargetEncoder(label_delay_days=7).fit_transform_features(original)
    second = TemporalTargetEncoder(label_delay_days=7).fit_transform_features(changed)

    pd.testing.assert_series_equal(first.iloc[1], second.iloc[1])
    assert first.iloc[1].eq(0.5).all()


def test_label_becomes_available_at_exact_delay_boundary() -> None:
    frame = delayed_transactions(times=[0, 7 * DAY], targets=[1, 0])

    transformed = TemporalTargetEncoder(label_delay_days=7).fit_transform_features(frame)

    assert transformed.iloc[1].eq(1.0).all()


def test_own_target_independence_with_delay() -> None:
    original = delayed_transactions()
    changed = original.copy()
    changed.loc[2, "isFraud"] = 1

    first = TemporalTargetEncoder(label_delay_days=7).fit_transform_features(original)
    second = TemporalTargetEncoder(label_delay_days=7).fit_transform_features(changed)

    pd.testing.assert_series_equal(first.iloc[2], second.iloc[2])


def test_future_target_invariance_with_delay() -> None:
    original = delayed_transactions()
    changed = original.copy()
    changed.loc[3, "isFraud"] = 0

    first = TemporalTargetEncoder(label_delay_days=7).fit_transform_features(original)
    second = TemporalTargetEncoder(label_delay_days=7).fit_transform_features(changed)

    pd.testing.assert_frame_equal(first.iloc[:3], second.iloc[:3])


def test_same_timestamp_isolation_with_zero_delay() -> None:
    original = delayed_transactions(times=[0, 0, DAY], targets=[1, 0, 0])
    changed = original.copy()
    changed.loc[0, "isFraud"] = 0

    first = TemporalTargetEncoder(label_delay_days=0).fit_transform_features(original)
    second = TemporalTargetEncoder(label_delay_days=0).fit_transform_features(changed)

    pd.testing.assert_series_equal(first.iloc[1], second.iloc[1])
    assert first.iloc[:2].eq(0.5).all().all()


def test_evaluation_labels_never_enter_progressive_maturity_state() -> None:
    fit = delayed_transactions(times=[0, 10 * DAY], targets=[1, 0])
    evaluation = delayed_transactions(times=[12 * DAY, 17 * DAY], targets=[0, 1])
    evaluation["TransactionID"] += 100
    changed = evaluation.copy()
    changed["isFraud"] = 1 - changed["isFraud"]
    encoder = TemporalTargetEncoder(label_delay_days=7).fit(fit)

    first = encoder.transform_features(evaluation)
    second = encoder.transform_features(changed)

    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0].eq(1.0).all()
    assert first.iloc[1].eq(0.5).all()


def test_changing_immature_fit_label_does_not_affect_evaluation_encoding() -> None:
    original = delayed_transactions(times=[0, 6 * DAY], targets=[1, 0])
    changed = original.copy()
    changed.loc[1, "isFraud"] = 1
    evaluation = delayed_transactions(times=[7 * DAY], targets=[0])
    evaluation["TransactionID"] += 100

    first = TemporalTargetEncoder(label_delay_days=7).fit(original).transform_features(evaluation)
    second = TemporalTargetEncoder(label_delay_days=7).fit(changed).transform_features(evaluation)

    pd.testing.assert_frame_equal(first, second)


def test_changing_matured_fit_label_affects_later_encoding() -> None:
    original = delayed_transactions(times=[0], targets=[1])
    changed = original.copy()
    changed.loc[0, "isFraud"] = 0
    evaluation = delayed_transactions(times=[7 * DAY], targets=[0])
    evaluation["TransactionID"] += 100

    first = TemporalTargetEncoder(label_delay_days=7).fit(original).transform_features(evaluation)
    second = TemporalTargetEncoder(label_delay_days=7).fit(changed).transform_features(evaluation)

    assert first.iloc[0, 0] == pytest.approx(1.0)
    assert second.iloc[0, 0] == pytest.approx(0.0)


def test_delay_days_convert_to_transaction_seconds() -> None:
    encoder = TemporalTargetEncoder(label_delay_days=14)

    assert encoder.label_delay_seconds == 14 * DAY


@pytest.mark.parametrize("delay", [-1, -0.001, np.nan])
def test_invalid_delay_raises_clear_error(delay: float) -> None:
    with pytest.raises(ValueError, match="non-negative finite"):
        TemporalTargetEncoder(label_delay_days=delay)


def test_inputs_remain_unmodified() -> None:
    fit = delayed_transactions(times=[0, 7 * DAY], targets=[1, 0])
    evaluation = delayed_transactions(times=[14 * DAY], targets=[0])
    evaluation["TransactionID"] += 100
    fit_original = fit.copy(deep=True)
    evaluation_original = evaluation.copy(deep=True)
    encoder = TemporalTargetEncoder(label_delay_days=7)

    encoder.fit_transform_features(fit)
    encoder.transform_features(evaluation)

    pd.testing.assert_frame_equal(fit, fit_original)
    pd.testing.assert_frame_equal(evaluation, evaluation_original)


def test_fallback_statistics_are_reported() -> None:
    frame = delayed_transactions(times=[0, 6 * DAY, 7 * DAY], targets=[1, 0, 0])
    encoder = TemporalTargetEncoder(label_delay_days=7)

    encoder.fit_transform_features(frame)

    statistics = encoder.training_statistics_
    assert statistics is not None
    assert statistics.fallback_row_count == 2
    assert statistics.fallback_fraction == pytest.approx(2 / 3)
