import numpy as np
import pandas as pd
import pytest

from fraud_detection.models import exponential_time_decay_weights

SECONDS_PER_DAY = 86_400


def test_newest_row_receives_largest_weight() -> None:
    timestamps = pd.Series([0, 10 * SECONDS_PER_DAY, 20 * SECONDS_PER_DAY])

    weights = exponential_time_decay_weights(timestamps, half_life_days=30)

    assert weights[0] < weights[1] < weights[2]


def test_one_half_life_has_half_the_relative_weight() -> None:
    timestamps = pd.Series([0, 30 * SECONDS_PER_DAY])

    weights = exponential_time_decay_weights(timestamps, half_life_days=30)

    assert weights[0] / weights[1] == pytest.approx(0.5)


def test_weights_are_normalized_to_mean_one() -> None:
    timestamps = pd.Series([0, 5, 10, 15]) * SECONDS_PER_DAY

    weights = exponential_time_decay_weights(timestamps, half_life_days=60)

    assert weights.mean() == pytest.approx(1.0)


def test_future_evaluation_changes_do_not_affect_fit_weights() -> None:
    fit_timestamps = pd.Series([0, 10, 20]) * SECONDS_PER_DAY
    first_evaluation = pd.Series([30, 40]) * SECONDS_PER_DAY
    changed_evaluation = pd.Series([300, 400]) * SECONDS_PER_DAY

    first = exponential_time_decay_weights(fit_timestamps, half_life_days=30)
    assert not first_evaluation.equals(changed_evaluation)
    second = exponential_time_decay_weights(fit_timestamps, half_life_days=30)

    np.testing.assert_array_equal(first, second)


def test_only_fit_timestamps_define_the_reference_time() -> None:
    fit_timestamps = pd.Series([0, 30]) * SECONDS_PER_DAY
    later_evaluation_timestamp = 10_000 * SECONDS_PER_DAY

    weights = exponential_time_decay_weights(fit_timestamps, half_life_days=30)

    assert later_evaluation_timestamp > fit_timestamps.max()
    assert weights[-1] == weights.max()
    assert weights[0] / weights[-1] == pytest.approx(0.5)


@pytest.mark.parametrize("half_life_days", [0, -1, np.nan, np.inf])
def test_invalid_half_life_raises(half_life_days: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        exponential_time_decay_weights(pd.Series([0, 1]), half_life_days=half_life_days)


def test_input_is_not_mutated() -> None:
    timestamps = pd.Series([0, 1, 2], name="TransactionDT")
    original = timestamps.copy(deep=True)

    exponential_time_decay_weights(timestamps, half_life_days=30)

    pd.testing.assert_series_equal(timestamps, original)
