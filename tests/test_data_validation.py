"""Tests for merged IEEE-CIS training-data validation."""

import pandas as pd
import pytest

from fraud_detection.data import validate_training_data


def _valid_dataframe() -> pd.DataFrame:
    """Create a minimal structurally valid training dataset."""
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [100, 200, 300, 400],
            "isFraud": [0, 1, 0, 0],
            "feature": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_validate_training_data_returns_summary_for_valid_data() -> None:
    """A valid dataset should return accessible counts and rates."""
    result = validate_training_data(_valid_dataframe())

    assert result.row_count == 4
    assert result.column_count == 4
    assert result.fraud_count == 1
    assert result.non_fraud_count == 3
    assert result.fraud_rate == pytest.approx(0.25)
    assert result.overall_missing_rate == pytest.approx(0.0)
    assert result.duplicate_transaction_ids == 0
    assert result.validation_passed is True


def test_validate_training_data_rejects_empty_dataframe() -> None:
    """An empty dataframe should fail validation clearly."""
    with pytest.raises(ValueError, match="must not be empty"):
        validate_training_data(pd.DataFrame())


@pytest.mark.parametrize("column", ["TransactionID", "TransactionDT", "isFraud"])
def test_validate_training_data_requires_structural_columns(column: str) -> None:
    """Each required structural column should be enforced."""
    dataframe = _valid_dataframe().drop(columns=column)

    with pytest.raises(ValueError, match=column):
        validate_training_data(dataframe)


def test_validate_training_data_rejects_duplicate_transaction_ids() -> None:
    """Duplicate transaction keys should fail validation."""
    dataframe = _valid_dataframe()
    dataframe.loc[1, "TransactionID"] = dataframe.loc[0, "TransactionID"]

    with pytest.raises(ValueError, match="TransactionID must be unique"):
        validate_training_data(dataframe)


def test_validate_training_data_rejects_invalid_target_values() -> None:
    """The target should be strictly binary."""
    dataframe = _valid_dataframe()
    dataframe.loc[1, "isFraud"] = 2

    with pytest.raises(ValueError, match="isFraud must contain only 0 and 1"):
        validate_training_data(dataframe)


def test_validate_training_data_allows_missing_feature_values() -> None:
    """Expected feature missingness should be reported without causing failure."""
    dataframe = _valid_dataframe()
    dataframe.loc[[0, 1], "feature"] = float("nan")

    result = validate_training_data(dataframe)

    assert result.validation_passed is True
    assert result.overall_missing_rate == pytest.approx(0.125)
    assert result.top_missing_columns["feature"] == pytest.approx(0.5)
