"""Run target-aware EDA using only the chronological IEEE-CIS training split."""

import logging
from pathlib import Path

import pandas as pd

from fraud_detection.data import (
    load_train_identity,
    load_train_transaction,
    merge_train_data,
    split_data_chronologically,
    validate_training_data,
)
from fraud_detection.evaluation.eda import (
    CategoricalSummary,
    categorical_cardinality_summary,
    categorical_summary,
    missingness_fraud_associations,
    missingness_summary,
    save_eda_plots,
    summarize_target,
    temporal_daily_summary,
    transaction_amount_summary,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_PATH = PROJECT_ROOT / "reports" / "train_eda_summary.md"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
CATEGORICAL_FEATURES = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
)
CARD_AND_IDENTITY_FEATURES = (
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "DeviceType",
    "DeviceInfo",
)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a small Markdown table without an optional formatting dependency."""
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _amount_table(amount_summary: pd.DataFrame) -> str:
    headers = [
        "Class",
        "Count",
        "Mean",
        "Median",
        "Std",
        "Q95",
        "Q99",
        "Max",
        "IQR outliers",
    ]
    rows: list[list[str]] = []
    for target_value, label in ((0, "Non-fraud"), (1, "Fraud")):
        values = amount_summary.loc[target_value]
        rows.append(
            [
                label,
                f"{int(values['count']):,}",
                f"{values['mean']:.2f}",
                f"{values['median']:.2f}",
                f"{values['std']:.2f}",
                f"{values['q95']:.2f}",
                f"{values['q99']:.2f}",
                f"{values['max']:.2f}",
                f"{int(values['iqr_outlier_count']):,} ({values['iqr_outlier_rate']:.2%})",
            ]
        )
    return _markdown_table(headers, rows)


def _categorical_table(summaries: list[CategoricalSummary]) -> str:
    headers = ["Feature", "Unique", "Missing", "Highest supported fraud-rate category"]
    rows: list[list[str]] = []
    for summary in summaries:
        supported = summary.category_metrics.loc[
            summary.category_metrics["transaction_count"] >= 500
        ]
        if supported.empty:
            strongest = "No category with at least 500 rows"
        else:
            row = supported.sort_values("fraud_rate", ascending=False).iloc[0]
            strongest = (
                f"{row['category']} - {row['fraud_rate']:.2%} "
                f"({int(row['transaction_count']):,} rows)"
            )
        rows.append(
            [
                summary.feature,
                f"{summary.unique_count:,}",
                f"{summary.missing_rate:.2%}",
                strongest,
            ]
        )
    return _markdown_table(headers, rows)


def _missingness_table(missingness: pd.DataFrame, limit: int = 15) -> str:
    rows = [
        [
            str(row.feature),
            f"{int(row.missing_count):,}",
            f"{row.missing_rate:.2%}",
        ]
        for row in missingness.head(limit).itertuples(index=False)
    ]
    return _markdown_table(["Feature", "Missing rows", "Missing rate"], rows)


def _missing_association_table(associations: pd.DataFrame) -> str:
    rows = [
        [
            str(row.feature),
            str(row.status),
            f"{int(row.transaction_count):,}",
            f"{row.fraud_rate:.2%}" if pd.notna(row.fraud_rate) else "n/a",
        ]
        for row in associations.itertuples(index=False)
    ]
    return _markdown_table(["Feature", "Status", "Rows", "Fraud rate"], rows)


def _cardinality_table(cardinality: pd.DataFrame, limit: int = 20) -> str:
    rows = [
        [
            str(row.feature),
            f"{int(row.unique_count):,}",
            str(row.cardinality_band),
            f"{row.missing_rate:.2%}",
            str(row.top_value),
            f"{row.top_frequency_rate:.2%}",
        ]
        for row in cardinality.head(limit).itertuples(index=False)
    ]
    return _markdown_table(
        ["Feature", "Unique", "Band", "Missing", "Most common", "Top frequency"], rows
    )


def _write_report(
    train: pd.DataFrame, excluded_rows: dict[str, int], figure_paths: list[Path]
) -> None:
    """Build an observed-facts-first Markdown report from the training split."""
    target = summarize_target(train)
    amounts = transaction_amount_summary(train)
    daily = temporal_daily_summary(train)
    categorical_summaries = [
        categorical_summary(train, feature)
        for feature in CATEGORICAL_FEATURES
        if feature in train.columns
    ]
    missingness = missingness_summary(train)
    selected_missing_features = list(
        dict.fromkeys(
            [
                *missingness.head(5)["feature"].tolist(),
                "P_emaildomain",
                "R_emaildomain",
                "DeviceType",
                "DeviceInfo",
                "dist1",
            ]
        )
    )
    selected_missing_features = [
        feature for feature in selected_missing_features if feature in train.columns
    ]
    missing_associations = missingness_fraud_associations(train, selected_missing_features)
    cardinality = categorical_cardinality_summary(train, CARD_AND_IDENTITY_FEATURES)

    first_window = train.iloc[: len(train) // 5]
    last_window = train.iloc[-(len(train) // 5) :]
    first_window_rate = float(first_window["isFraud"].mean())
    last_window_rate = float(last_window["isFraud"].mean())
    missing_rates = missingness["missing_rate"]
    missing_bands = {
        "none": int((missing_rates == 0).sum()),
        "low": int(((missing_rates > 0) & (missing_rates <= 0.20)).sum()),
        "moderate": int(((missing_rates > 0.20) & (missing_rates <= 0.60)).sum()),
        "high": int((missing_rates > 0.60).sum()),
    }
    high_cardinality = cardinality.loc[cardinality["cardinality_band"] == "high"]

    report = f"""# IEEE-CIS Training-Split EDA Summary

## Scope and leakage guardrail

**Observed fact:** This report uses only the oldest chronological training partition: 
{target.row_count:,} rows. The validation ({excluded_rows["validation"]:,} rows) and holdout 
({excluded_rows["holdout"]:,} rows) partitions were excluded from every EDA calculation and plot.

`TransactionDT` is treated only as relative time. No calendar-date interpretation was made. No
values were imputed, encoded, removed, or transformed in the persisted data.

## Target distribution

**Observed facts**

- Fraud: {target.fraud_count:,} transactions.
- Non-fraud: {target.non_fraud_count:,} transactions.
- Training fraud rate: **{target.fraud_rate:.2%}**.
- The target is strongly imbalanced.

![Training class distribution](figures/train_class_distribution.png)

## Transaction amount

**Observed facts**

{_amount_table(amounts)}

The distribution is right-skewed, so the figure uses `log(1 + TransactionAmt)` for readability.
IQR outliers are reported, not removed. Fraud median amount is 
{amounts.loc[1, "median"]:.2f}, compared with {amounts.loc[0, "median"]:.2f} for non-fraud.

![Transaction amount by class](figures/train_transaction_amount_by_class.png)

**Feature hypothesis:** A log amount representation and past-only entity-relative amount features
may be useful later. This has not been tested as model signal.

## Relative-time behavior

**Observed facts**

- The training period spans elapsed days {int(daily["elapsed_day"].min())} through
  {int(daily["elapsed_day"].max())}.
- Fraud rate in the first chronological 20% is **{first_window_rate:.2%}**; in the last 20% it is
  **{last_window_rate:.2%}**.
- Daily transaction volume and fraud rate vary through the training period; this supports retaining
  chronological validation rather than random splitting.

![Daily activity](figures/train_daily_volume_and_fraud_rate.png)

**Feature hypothesis:** Past-only velocity and relative-hour/day patterns deserve later testing.
No temporal feature is created here.

## Interpretable categorical features

**Observed facts**

{_categorical_table(categorical_summaries)}

Only frequent categories are shown; remaining values are grouped into `<OTHER>` for reporting.
Category-level target differences are associations, not causal evidence.

![Categorical fraud rates](figures/train_categorical_fraud_rates.png)

## Card and identity overview

**Observed facts**

{_cardinality_table(cardinality)}

There are {len(high_cardinality):,} high-cardinality object/card/address/device features in this
reviewed set. These identifiers may support behavior aggregation or graph connectivity later, but
raw target encoding would require strict out-of-fold and time-aware controls.

## Missingness

**Observed facts**

- Columns with no missing values: {missing_bands["none"]}.
- Low missingness (0-20%): {missing_bands["low"]}.
- Moderate missingness (>20-60%): {missing_bands["moderate"]}.
- High missingness (>60%): {missing_bands["high"]}.

{_missingness_table(missingness)}

![Highest missingness](figures/train_top_missing_columns.png)

Selected missing-versus-present target comparisons:

{_missing_association_table(missing_associations)}

**Feature hypothesis:** Missing indicators may contain signal for selected fields. They must be
evaluated temporally before any imputation or column-removal policy is chosen.

## Cardinality and later encoding considerations

**Observed facts:** Cardinality bands are low (<=10), medium (11-100), and high (>100). The table
above reports the most common value and its frequency, helping distinguish concentrated categories
from identifier-like fields.

**Feature hypotheses**

1. Log-transform `TransactionAmt` while retaining the raw amount.
2. Add missing-value indicators only for fields whose missingness remains useful out of time.
3. Add past-only transaction velocity by card, address, email, and device identifiers.
4. Add past-only entity amount deviations and rolling aggregates.
5. Test frequency encoding for high-cardinality card, email, and device fields.
6. Test time-safe categorical encodings for medium-cardinality fields.
7. Test interactions among `ProductCD`, card network/type, email domain, and device type.
8. Build graph features from shared cards, addresses, emails, and devices in a later dedicated step.
9. Test relative time-of-day and day progression without treating `TransactionDT` as a real date.

## Potential leakage and suspicious-feature review

- **`TransactionID` - review required, not confirmed leakage.** It is an identifier and is likely
  correlated with collection order. It should not be treated as an ordinary magnitude feature.
- **`TransactionDT` - temporal proxy, not confirmed leakage.** It is available at transaction time,
  but can let a model memorize changing base rates. Evaluation must remain chronological.
- **Anonymized `V*`, `D*`, and identity fields - provenance review required.** Their definitions are
  unavailable here. Strong associations alone would not prove leakage; feature timing must be
  confirmed before production use.
- **`isFraud` - confirmed target.** It must never enter predictors. The future production simulator
  must withhold holdout labels until delayed feedback is released.

No non-target column is declared leaked based on this EDA alone.

## Generated figures

{chr(10).join(f"- `{path.relative_to(PROJECT_ROOT).as_posix()}`" for path in figure_paths)}
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    """Execute training-only EDA and persist the report and figures."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    transaction = load_train_transaction(RAW_DATA_DIR)
    identity = load_train_identity(RAW_DATA_DIR)
    merged = merge_train_data(transaction, identity)
    del transaction, identity

    validate_training_data(merged)
    split = split_data_chronologically(merged)
    train = split.train
    excluded_rows = {
        "validation": len(split.validation),
        "holdout": len(split.holdout),
    }
    LOGGER.info(
        "EDA scope: train=%d rows; validation=%d and holdout=%d rows excluded",
        len(train),
        excluded_rows["validation"],
        excluded_rows["holdout"],
    )

    figure_paths = save_eda_plots(train, FIGURES_DIR)
    _write_report(train, excluded_rows, figure_paths)
    target = summarize_target(train)
    print(f"Training rows analyzed: {target.row_count:,}")
    print(f"Training fraud rate: {target.fraud_rate:.2%}")
    print(f"Validation rows excluded: {excluded_rows['validation']:,}")
    print(f"Holdout rows excluded: {excluded_rows['holdout']:,}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figures generated: {len(figure_paths)}")


if __name__ == "__main__":
    main()
