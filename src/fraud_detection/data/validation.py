"""Structural validation and quality summaries for merged IEEE-CIS training data."""

import logging
from dataclasses import dataclass

import pandas as pd

LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = ("TransactionID", "TransactionDT", "isFraud")
VALID_TARGET_VALUES = {0, 1}
TOP_MISSING_COLUMN_COUNT = 10


@dataclass(frozen=True, slots=True)
class TrainingDataValidationResult:
    """Summary metrics produced by successful training-data validation."""

    row_count: int
    column_count: int
    fraud_count: int
    non_fraud_count: int
    fraud_rate: float
    overall_missing_rate: float
    duplicate_transaction_ids: int
    top_missing_columns: dict[str, float]
    validation_passed: bool


def validate_training_data(dataframe: pd.DataFrame) -> TrainingDataValidationResult:
    """Validate critical structure and summarize merged IEEE-CIS training data."""
    if dataframe.empty:
        raise ValueError("Merged training DataFrame must not be empty")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            f"Merged training DataFrame is missing required columns: {missing_columns}"
        )

    duplicate_transaction_ids = int(dataframe["TransactionID"].duplicated().sum())
    if duplicate_transaction_ids:
        raise ValueError(
            "TransactionID must be unique in the merged training DataFrame; "
            f"found {duplicate_transaction_ids} duplicate rows"
        )

    invalid_target_mask = ~dataframe["isFraud"].isin(VALID_TARGET_VALUES)
    if invalid_target_mask.any():
        invalid_values = dataframe.loc[invalid_target_mask, "isFraud"].drop_duplicates().tolist()
        raise ValueError(
            f"isFraud must contain only 0 and 1; found invalid values: {invalid_values}"
        )

    row_count, column_count = dataframe.shape
    fraud_count = int((dataframe["isFraud"] == 1).sum())
    non_fraud_count = int((dataframe["isFraud"] == 0).sum())
    fraud_rate = fraud_count / row_count

    missing_rates = dataframe.isna().mean()
    overall_missing_rate = float(missing_rates.mean())
    top_missing_columns = {
        str(column): float(rate)
        for column, rate in missing_rates.nlargest(TOP_MISSING_COLUMN_COUNT).items()
    }

    result = TrainingDataValidationResult(
        row_count=row_count,
        column_count=column_count,
        fraud_count=fraud_count,
        non_fraud_count=non_fraud_count,
        fraud_rate=fraud_rate,
        overall_missing_rate=overall_missing_rate,
        duplicate_transaction_ids=duplicate_transaction_ids,
        top_missing_columns=top_missing_columns,
        validation_passed=True,
    )

    LOGGER.info("Validated merged training data with shape %s", dataframe.shape)
    LOGGER.info(
        "Fraud: %d, non-fraud: %d, fraud rate: %.2f%%",
        fraud_count,
        non_fraud_count,
        fraud_rate * 100,
    )
    LOGGER.info("Overall missing-value rate: %.2f%%", overall_missing_rate * 100)
    LOGGER.info("Top columns by missing rate: %s", top_missing_columns)
    return result
