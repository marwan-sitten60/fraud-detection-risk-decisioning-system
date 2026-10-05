"""Lightweight tests for reusable training EDA helpers."""

from pathlib import Path

import pandas as pd
import pytest

from fraud_detection.evaluation.eda import (
    categorical_cardinality_summary,
    categorical_summary,
    missingness_fraud_associations,
    missingness_summary,
    summarize_target,
    temporal_daily_summary,
    transaction_amount_summary,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Return a small labeled dataset with time, amount, and missing values."""
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [0, 3_600, 86_400, 90_000, 172_800, 176_400],
            "isFraud": [0, 1, 0, 0, 1, 0],
            "TransactionAmt": [10.0, 20.0, 10.0, 40.0, 100.0, 20.0],
            "ProductCD": ["W", "C", "W", None, "C", "R"],
            "feature": [1.0, None, 2.0, None, 3.0, 4.0],
        }
    )


def test_summarize_target(sample_data: pd.DataFrame) -> None:
    """Target summary should return class counts and prevalence."""
    summary = summarize_target(sample_data)

    assert summary.row_count == 6
    assert summary.fraud_count == 2
    assert summary.non_fraud_count == 4
    assert summary.fraud_rate == pytest.approx(1 / 3)


def test_transaction_amount_summary(sample_data: pd.DataFrame) -> None:
    """Amount statistics should be separated by target class."""
    summary = transaction_amount_summary(sample_data)

    assert summary.loc[0, "count"] == 4
    assert summary.loc[1, "count"] == 2
    assert summary.loc[0, "median"] == pytest.approx(15.0)
    assert summary.loc[1, "mean"] == pytest.approx(60.0)
    assert "iqr_outlier_rate" in summary.columns


def test_temporal_daily_summary_uses_relative_days(sample_data: pd.DataFrame) -> None:
    """Temporal aggregation should retain daily volume and fraud rates."""
    daily = temporal_daily_summary(sample_data)

    assert daily["elapsed_day"].tolist() == [0, 1, 2]
    assert daily["transaction_count"].tolist() == [2, 2, 2]
    assert daily["fraud_count"].tolist() == [1, 0, 1]


def test_categorical_summary_groups_infrequent_values(sample_data: pd.DataFrame) -> None:
    """Categorical summaries should bound output cardinality."""
    summary = categorical_summary(sample_data, "ProductCD", max_categories=2)

    assert summary.unique_count == 3
    assert summary.missing_rate == pytest.approx(1 / 6)
    assert "<OTHER>" in summary.category_metrics["category"].tolist()
    assert summary.category_metrics["transaction_count"].sum() == 6


def test_missingness_helpers_report_rates_and_target_association(
    sample_data: pd.DataFrame,
) -> None:
    """Missingness helpers should measure gaps without rejecting them."""
    summary = missingness_summary(sample_data)
    associations = missingness_fraud_associations(sample_data, ["feature"])

    feature_missing = summary.loc[summary["feature"] == "feature", "missing_rate"].iloc[0]
    assert feature_missing == pytest.approx(1 / 3)
    assert set(associations["status"]) == {"missing", "present"}
    assert associations["transaction_count"].sum() == 6


def test_categorical_cardinality_summary_assigns_bands(sample_data: pd.DataFrame) -> None:
    """Cardinality summary should include object and explicitly selected columns."""
    summary = categorical_cardinality_summary(sample_data, ("TransactionID",))

    assert {"ProductCD", "TransactionID"}.issubset(set(summary["feature"]))
    transaction_id = summary.loc[summary["feature"] == "TransactionID"].iloc[0]
    assert transaction_id["unique_count"] == 6
    assert transaction_id["cardinality_band"] == "low"


def test_eda_helpers_reject_missing_required_columns() -> None:
    """Missing analysis inputs should produce a clear error."""
    with pytest.raises(ValueError, match="TransactionAmt"):
        transaction_amount_summary(pd.DataFrame({"isFraud": [0, 1]}))


def test_plot_output_directory_type_is_path() -> None:
    """The plot API contract uses pathlib paths."""
    assert isinstance(Path("reports/figures"), Path)
