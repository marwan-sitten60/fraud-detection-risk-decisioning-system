"""Reusable exploratory analysis utilities for the chronological training split."""

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "fraud-detection-matplotlib"),
)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)
TARGET = "isFraud"


@dataclass(frozen=True, slots=True)
class TargetSummary:
    """Class distribution for a fraud-labeled dataset."""

    row_count: int
    fraud_count: int
    non_fraud_count: int
    fraud_rate: float


@dataclass(frozen=True, slots=True)
class CategoricalSummary:
    """Cardinality, missingness, and target rates for one categorical feature."""

    feature: str
    unique_count: int
    missing_rate: float
    category_metrics: pd.DataFrame


def _require_columns(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Raise a clear error when an EDA input column is unavailable."""
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"EDA requires columns that are missing: {missing}")


def summarize_target(dataframe: pd.DataFrame) -> TargetSummary:
    """Summarize fraud and non-fraud counts."""
    _require_columns(dataframe, (TARGET,))
    if dataframe.empty:
        raise ValueError("EDA DataFrame must not be empty")

    fraud_count = int((dataframe[TARGET] == 1).sum())
    non_fraud_count = int((dataframe[TARGET] == 0).sum())
    return TargetSummary(
        row_count=len(dataframe),
        fraud_count=fraud_count,
        non_fraud_count=non_fraud_count,
        fraud_rate=fraud_count / len(dataframe),
    )


def transaction_amount_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Compare transaction amount statistics and IQR outliers by target class."""
    _require_columns(dataframe, (TARGET, "TransactionAmt"))
    grouped = dataframe.groupby(TARGET, observed=True)["TransactionAmt"]
    summary = grouped.agg(["count", "mean", "median", "std", "min", "max"])
    quantiles = grouped.quantile([0.01, 0.25, 0.75, 0.95, 0.99]).unstack()
    quantiles.columns = ["q01", "q25", "q75", "q95", "q99"]
    summary = summary.join(quantiles)

    outlier_counts: dict[object, int] = {}
    for target_value, amounts in grouped:
        first_quartile = amounts.quantile(0.25)
        third_quartile = amounts.quantile(0.75)
        iqr = third_quartile - first_quartile
        outlier_counts[target_value] = int(
            ((amounts < first_quartile - 1.5 * iqr) | (amounts > third_quartile + 1.5 * iqr)).sum()
        )
    summary["iqr_outlier_count"] = pd.Series(outlier_counts)
    summary["iqr_outlier_rate"] = summary["iqr_outlier_count"] / summary["count"]
    return summary


def temporal_daily_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate volume and fraud rate by elapsed day within the supplied data."""
    _require_columns(dataframe, (TARGET, "TransactionDT"))
    elapsed_day = (
        (dataframe["TransactionDT"] - dataframe["TransactionDT"].min()) // 86_400
    ).astype(int)
    daily = (
        dataframe.assign(elapsed_day=elapsed_day)
        .groupby("elapsed_day", observed=True)[TARGET]
        .agg(transaction_count="size", fraud_count="sum", fraud_rate="mean")
        .reset_index()
    )
    return daily


def categorical_summary(
    dataframe: pd.DataFrame,
    feature: str,
    max_categories: int = 10,
) -> CategoricalSummary:
    """Summarize a categorical feature, grouping infrequent values as other."""
    _require_columns(dataframe, (TARGET, feature))
    if max_categories <= 0:
        raise ValueError("max_categories must be greater than zero")

    raw_values = dataframe[feature]
    display_values = raw_values.astype("object").where(raw_values.notna(), "<MISSING>")
    most_frequent = display_values.value_counts().head(max_categories).index
    grouped_values = display_values.where(display_values.isin(most_frequent), "<OTHER>")
    category_metrics = (
        pd.DataFrame({"category": grouped_values, TARGET: dataframe[TARGET]})
        .groupby("category", observed=True)[TARGET]
        .agg(transaction_count="size", fraud_count="sum", fraud_rate="mean")
        .sort_values("transaction_count", ascending=False)
        .reset_index()
    )
    return CategoricalSummary(
        feature=feature,
        unique_count=int(raw_values.nunique(dropna=True)),
        missing_rate=float(raw_values.isna().mean()),
        category_metrics=category_metrics,
    )


def missingness_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing counts and rates from highest to lowest."""
    missing_count = dataframe.isna().sum()
    return (
        pd.DataFrame(
            {
                "missing_count": missing_count,
                "missing_rate": missing_count / len(dataframe),
            }
        )
        .sort_values("missing_rate", ascending=False)
        .rename_axis("feature")
        .reset_index()
    )


def missingness_fraud_associations(
    dataframe: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Compare fraud rates when selected feature values are missing versus present."""
    _require_columns(dataframe, (TARGET, *features))
    rows: list[dict[str, object]] = []
    for feature in features:
        missing_mask = dataframe[feature].isna()
        for status, mask in (("missing", missing_mask), ("present", ~missing_mask)):
            subset_target = dataframe.loc[mask, TARGET]
            rows.append(
                {
                    "feature": feature,
                    "status": status,
                    "transaction_count": len(subset_target),
                    "fraud_count": int((subset_target == 1).sum()),
                    "fraud_rate": float(subset_target.mean()) if len(subset_target) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def categorical_cardinality_summary(
    dataframe: pd.DataFrame,
    additional_features: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Summarize cardinality and dominant values for object-like and selected features."""
    object_features = dataframe.select_dtypes(include=["object", "string", "category"]).columns
    features = list(dict.fromkeys([*object_features, *additional_features]))
    rows: list[dict[str, object]] = []
    for feature in features:
        if feature not in dataframe.columns:
            continue
        values = dataframe[feature]
        unique_count = int(values.nunique(dropna=True))
        counts = values.value_counts(dropna=False)
        top_value = counts.index[0] if not counts.empty else None
        top_count = int(counts.iloc[0]) if not counts.empty else 0
        if unique_count <= 10:
            cardinality_band = "low"
        elif unique_count <= 100:
            cardinality_band = "medium"
        else:
            cardinality_band = "high"
        rows.append(
            {
                "feature": feature,
                "unique_count": unique_count,
                "missing_rate": float(values.isna().mean()),
                "top_value": top_value,
                "top_count": top_count,
                "top_frequency_rate": top_count / len(dataframe),
                "cardinality_band": cardinality_band,
            }
        )
    return pd.DataFrame(rows).sort_values("unique_count", ascending=False).reset_index(drop=True)


def save_eda_plots(
    dataframe: pd.DataFrame,
    figures_dir: Path,
    categorical_features: tuple[str, ...] = ("ProductCD", "card4", "card6", "DeviceType"),
) -> list[Path]:
    """Save a focused set of training-only EDA figures."""
    _require_columns(dataframe, (TARGET, "TransactionAmt", "TransactionDT"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    paths = [
        _plot_class_distribution(dataframe, figures_dir),
        _plot_transaction_amount(dataframe, figures_dir),
        _plot_temporal_patterns(dataframe, figures_dir),
        _plot_missingness(dataframe, figures_dir),
    ]
    available_categories = tuple(
        feature for feature in categorical_features if feature in dataframe.columns
    )
    if available_categories:
        paths.append(_plot_categorical_fraud_rates(dataframe, figures_dir, available_categories))
    LOGGER.info("Saved %d EDA figures under %s", len(paths), figures_dir)
    return paths


def _plot_class_distribution(dataframe: pd.DataFrame, figures_dir: Path) -> Path:
    summary = summarize_target(dataframe)
    path = figures_dir / "train_class_distribution.png"
    fig, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(
        ["Non-fraud", "Fraud"],
        [summary.non_fraud_count, summary.fraud_count],
        color=["#4C78A8", "#E45756"],
    )
    axis.bar_label(bars, labels=[f"{value:,}" for value in bars.datavalues], padding=3)
    axis.set(title="Training class distribution", ylabel="Transactions")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_transaction_amount(dataframe: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "train_transaction_amount_by_class.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = ((0, "Non-fraud", "#4C78A8"), (1, "Fraud", "#E45756"))
    log_amounts: list[pd.Series] = []
    for target_value, label, color in labels:
        amounts = dataframe.loc[dataframe[TARGET] == target_value, "TransactionAmt"].dropna()
        transformed = np.log1p(amounts)
        log_amounts.append(transformed)
        axes[0].hist(transformed, bins=60, density=True, alpha=0.55, label=label, color=color)
    axes[0].set(
        title="Transaction amount distribution",
        xlabel="log(1 + TransactionAmt)",
        ylabel="Density",
    )
    axes[0].legend()
    axes[1].boxplot(
        log_amounts,
        tick_labels=["Non-fraud", "Fraud"],
        showfliers=True,
        flierprops={"markersize": 2, "alpha": 0.15},
    )
    axes[1].set(title="Transaction amount by class", ylabel="log(1 + TransactionAmt)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_temporal_patterns(dataframe: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "train_daily_volume_and_fraud_rate.png"
    daily = temporal_daily_summary(dataframe)
    fig, volume_axis = plt.subplots(figsize=(11, 4.5))
    fraud_axis = volume_axis.twinx()
    volume_axis.plot(
        daily["elapsed_day"], daily["transaction_count"], color="#4C78A8", label="Volume"
    )
    fraud_axis.plot(
        daily["elapsed_day"], daily["fraud_rate"] * 100, color="#E45756", label="Fraud rate"
    )
    volume_axis.set(
        title="Training activity over relative time",
        xlabel="Elapsed day within training period",
        ylabel="Transactions",
    )
    fraud_axis.set_ylabel("Fraud rate (%)")
    lines = volume_axis.lines + fraud_axis.lines
    volume_axis.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_missingness(dataframe: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "train_top_missing_columns.png"
    top_missing = missingness_summary(dataframe).head(20).sort_values("missing_rate")
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top_missing["feature"], top_missing["missing_rate"] * 100, color="#72B7B2")
    axis.set(
        title="Training columns with highest missingness",
        xlabel="Missing values (%)",
        ylabel="Feature",
        xlim=(0, 100),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_categorical_fraud_rates(
    dataframe: pd.DataFrame,
    figures_dir: Path,
    features: tuple[str, ...],
) -> Path:
    path = figures_dir / "train_categorical_fraud_rates.png"
    column_count = 2
    row_count = int(np.ceil(len(features) / column_count))
    fig, axes = plt.subplots(row_count, column_count, figsize=(12, 4 * row_count), squeeze=False)
    for axis, feature in zip(axes.flat, features, strict=False):
        metrics = categorical_summary(dataframe, feature, max_categories=8).category_metrics
        metrics = metrics.sort_values("fraud_rate")
        axis.barh(metrics["category"].astype(str), metrics["fraud_rate"] * 100, color="#F58518")
        axis.set(title=feature, xlabel="Fraud rate (%)")
    for axis in axes.flat[len(features) :]:
        axis.set_visible(False)
    fig.suptitle("Training fraud rate across frequent categories", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
