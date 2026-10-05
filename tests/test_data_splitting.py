"""Tests for chronological IEEE-CIS dataset splitting."""

import pandas as pd
import pytest

from fraud_detection.data import split_data_chronologically, summarize_split


def _dataframe(row_count: int = 20) -> pd.DataFrame:
    """Create a small valid dataset in reverse chronological order."""
    return pd.DataFrame(
        {
            "TransactionID": list(range(row_count, 0, -1)),
            "TransactionDT": list(range(row_count, 0, -1)),
            "isFraud": [index % 2 for index in range(row_count)],
            "feature": [float(index) for index in range(row_count)],
        }
    )


def test_split_data_chronologically_orders_oldest_to_newest() -> None:
    """Earlier records should always precede later records across splits."""
    split = split_data_chronologically(_dataframe())

    assert split.train["TransactionDT"].is_monotonic_increasing
    assert split.validation["TransactionDT"].is_monotonic_increasing
    assert split.holdout["TransactionDT"].is_monotonic_increasing
    assert split.train["TransactionDT"].max() <= split.validation["TransactionDT"].min()
    assert split.validation["TransactionDT"].max() <= split.holdout["TransactionDT"].min()


def test_split_data_chronologically_uses_expected_sizes() -> None:
    """Default fractions should create a 70/15/15 partition."""
    split = split_data_chronologically(_dataframe())

    assert (len(split.train), len(split.validation), len(split.holdout)) == (14, 3, 3)
    assert sum(map(len, (split.train, split.validation, split.holdout))) == 20


def test_split_data_chronologically_has_no_transaction_id_overlap() -> None:
    """Transaction IDs should belong to exactly one split."""
    split = split_data_chronologically(_dataframe())

    assert split.transaction_id_overlap is False


def test_split_data_chronologically_has_valid_temporal_boundaries() -> None:
    """The result should expose successful temporal-boundary verification."""
    split = split_data_chronologically(_dataframe())

    assert split.temporal_boundaries_valid is True


@pytest.mark.parametrize(
    ("train", "validation", "holdout"),
    [
        (0.0, 0.5, 0.5),
        (-0.1, 0.5, 0.6),
        (0.7, 0.2, 0.2),
    ],
)
def test_split_data_chronologically_rejects_invalid_fractions(
    train: float, validation: float, holdout: float
) -> None:
    """Fractions must be positive and sum to one."""
    with pytest.raises(ValueError, match="fractions"):
        split_data_chronologically(
            _dataframe(),
            train_fraction=train,
            validation_fraction=validation,
            holdout_fraction=holdout,
        )


@pytest.mark.parametrize("column", ["TransactionDT", "TransactionID"])
def test_split_data_chronologically_requires_key_columns(column: str) -> None:
    """Time and transaction identifiers are required."""
    dataframe = _dataframe().drop(columns=column)

    with pytest.raises(ValueError, match=column):
        split_data_chronologically(dataframe)


def test_split_data_chronologically_rejects_empty_dataframe() -> None:
    """An empty input should fail clearly."""
    with pytest.raises(ValueError, match="must not be empty"):
        split_data_chronologically(pd.DataFrame())


def test_split_data_chronologically_preserves_target() -> None:
    """The target should remain available in every split."""
    split = split_data_chronologically(_dataframe())

    assert all(
        "isFraud" in dataframe.columns
        for dataframe in (split.train, split.validation, split.holdout)
    )


def test_split_data_chronologically_does_not_shuffle_ordered_data() -> None:
    """Already chronological input should retain its record order."""
    dataframe = _dataframe().sort_values("TransactionDT")

    split = split_data_chronologically(dataframe)

    combined_ids = pd.concat(
        [
            split.train["TransactionID"],
            split.validation["TransactionID"],
            split.holdout["TransactionID"],
        ]
    ).tolist()
    assert combined_ids == dataframe["TransactionID"].tolist()


def test_summarize_split_reports_expected_metrics() -> None:
    """Split summaries should report size, fraud, and time range."""
    split = split_data_chronologically(_dataframe())

    summary = summarize_split(split.train, total_rows=20)

    assert summary.row_count == 14
    assert summary.percentage == pytest.approx(0.7)
    assert summary.fraud_count == 7
    assert summary.fraud_rate == pytest.approx(0.5)
    assert summary.transaction_dt_min == 1
    assert summary.transaction_dt_max == 14
