"""Tests for target-independent Phase 1 feature engineering."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import Phase1FeatureEngineer


def _training_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionAmt": [0.0, 9.0, 99.0],
            "TransactionDT": [100, 3_700, 90_100],
            "card1": [10, 10, 20],
            "DeviceType": ["desktop", None, "mobile"],
            "isFraud": [0, 1, 0],
        }
    )


def test_log_amount_and_relative_time_features() -> None:
    """Amount and relative-time calculations should use the fitted training origin."""
    result = Phase1FeatureEngineer().fit_transform(_training_data())

    assert result["log_transaction_amount"].tolist() == pytest.approx(np.log1p([0, 9, 99]))
    assert result["elapsed_day"].tolist() == [0, 0, 1]
    assert result["relative_hour"].tolist() == [0, 1, 1]
    assert "TransactionAmt" in result.columns
    assert "TransactionDT" in result.columns


def test_negative_amount_raises_clear_error() -> None:
    """Negative amounts are invalid for the documented log1p feature."""
    dataframe = _training_data()
    dataframe.loc[0, "TransactionAmt"] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        Phase1FeatureEngineer().fit(dataframe)


def test_missing_indicators_are_added_for_available_columns_only() -> None:
    """Optional missingness candidates should not be required."""
    result = Phase1FeatureEngineer().fit_transform(_training_data())

    assert result["DeviceType_is_missing"].tolist() == [0, 1, 0]
    assert "DeviceInfo_is_missing" not in result.columns


def test_frequency_encoding_is_fitted_on_training_only() -> None:
    """Training frequencies should be reused unchanged for validation."""
    train = _training_data()
    validation = pd.DataFrame(
        {
            "TransactionAmt": [4.0, 8.0],
            "TransactionDT": [100_000, 110_000],
            "card1": [10, 999],
            "DeviceType": ["desktop", "tablet"],
            "isFraud": [1, 0],
        }
    )
    transformer = Phase1FeatureEngineer().fit(train)

    result = transformer.transform(validation)

    assert result["card1_frequency"].tolist() == pytest.approx([2 / 3, 0.0])
    assert "999" not in transformer.frequency_maps_["card1"]


def test_target_is_preserved_but_never_used_as_a_feature_source() -> None:
    """Changing labels must not alter target-independent engineered values."""
    first = _training_data()
    second = first.copy()
    second["isFraud"] = 1 - second["isFraud"]

    first_features = Phase1FeatureEngineer().fit_transform(first).drop(columns="isFraud")
    second_features = Phase1FeatureEngineer().fit_transform(second).drop(columns="isFraud")

    pd.testing.assert_frame_equal(first_features, second_features)
    assert "isFraud_frequency" not in first_features.columns


def test_input_dataframe_is_not_mutated() -> None:
    """Feature transformation should return a copy and leave inputs unchanged."""
    dataframe = _training_data()
    original = dataframe.copy(deep=True)

    Phase1FeatureEngineer().fit_transform(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_transform_without_optional_sources_does_not_crash() -> None:
    """Only core amount/time sources are mandatory with every group enabled."""
    dataframe = pd.DataFrame({"TransactionAmt": [1.0], "TransactionDT": [10]})

    result = Phase1FeatureEngineer().fit_transform(dataframe)

    assert set(result.columns) == {
        "TransactionAmt",
        "TransactionDT",
        "log_transaction_amount",
        "elapsed_day",
        "relative_hour",
    }
