"""Chronological splitting utilities for merged IEEE-CIS training data."""

import logging
import math
from dataclasses import dataclass

import pandas as pd

LOGGER = logging.getLogger(__name__)
TRANSACTION_ID = "TransactionID"
TRANSACTION_DT = "TransactionDT"
TARGET = "isFraud"


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    """Chronologically ordered train, validation, and holdout datasets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    holdout: pd.DataFrame

    @property
    def transaction_id_overlap(self) -> bool:
        """Return whether any transaction ID occurs in multiple splits."""
        train_ids = set(self.train[TRANSACTION_ID])
        validation_ids = set(self.validation[TRANSACTION_ID])
        holdout_ids = set(self.holdout[TRANSACTION_ID])
        return bool(
            train_ids & validation_ids or train_ids & holdout_ids or validation_ids & holdout_ids
        )

    @property
    def temporal_boundaries_valid(self) -> bool:
        """Return whether split time ranges are ordered and non-overlapping in time order."""
        return bool(
            self.train[TRANSACTION_DT].max() <= self.validation[TRANSACTION_DT].min()
            and self.validation[TRANSACTION_DT].max() <= self.holdout[TRANSACTION_DT].min()
        )


@dataclass(frozen=True, slots=True)
class SplitSummary:
    """Compact metrics for one temporal split."""

    row_count: int
    percentage: float
    fraud_count: int
    fraud_rate: float
    transaction_dt_min: object
    transaction_dt_max: object


def _validate_fractions(
    train_fraction: float,
    validation_fraction: float,
    holdout_fraction: float,
) -> None:
    """Validate positive fractions that form a complete partition."""
    fractions = (train_fraction, validation_fraction, holdout_fraction)
    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("Train, validation, and holdout fractions must all be greater than zero")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "Train, validation, and holdout fractions must sum to 1.0; "
            f"received {sum(fractions):.12g}"
        )


def split_data_chronologically(
    dataframe: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    holdout_fraction: float = 0.15,
) -> TemporalSplit:
    """Split data from oldest to newest without shuffling or dropping rows."""
    _validate_fractions(train_fraction, validation_fraction, holdout_fraction)

    if dataframe.empty:
        raise ValueError("DataFrame must not be empty")

    missing_columns = [
        column for column in (TRANSACTION_ID, TRANSACTION_DT) if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")

    row_count = len(dataframe)
    train_end = int(row_count * train_fraction)
    validation_end = train_end + int(row_count * validation_fraction)
    if train_end == 0 or validation_end in (train_end, row_count):
        raise ValueError(
            "Split fractions produce an empty split; provide more rows or adjust the fractions"
        )

    if dataframe[TRANSACTION_DT].is_monotonic_increasing:
        sorted_dataframe = dataframe
    else:
        sorted_dataframe = dataframe.sort_values(TRANSACTION_DT, kind="mergesort")
    split = TemporalSplit(
        train=sorted_dataframe.iloc[:train_end],
        validation=sorted_dataframe.iloc[train_end:validation_end],
        holdout=sorted_dataframe.iloc[validation_end:],
    )

    if split.transaction_id_overlap:
        raise ValueError("TransactionID overlap detected across temporal splits")
    if not split.temporal_boundaries_valid:
        raise ValueError("Temporal boundaries are invalid after chronological sorting")

    LOGGER.info(
        "Created chronological split with train=%d, validation=%d, holdout=%d rows",
        len(split.train),
        len(split.validation),
        len(split.holdout),
    )
    return split


def summarize_split(dataframe: pd.DataFrame, total_rows: int) -> SplitSummary:
    """Return row, fraud, and temporal-range metrics for one split."""
    if total_rows <= 0:
        raise ValueError("total_rows must be greater than zero")
    required_columns = (TARGET, TRANSACTION_DT)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Split is missing required summary columns: {missing_columns}")
    if dataframe.empty:
        raise ValueError("Cannot summarize an empty split")

    fraud_count = int((dataframe[TARGET] == 1).sum())
    return SplitSummary(
        row_count=len(dataframe),
        percentage=len(dataframe) / total_rows,
        fraud_count=fraud_count,
        fraud_rate=fraud_count / len(dataframe),
        transaction_dt_min=dataframe[TRANSACTION_DT].min(),
        transaction_dt_max=dataframe[TRANSACTION_DT].max(),
    )
