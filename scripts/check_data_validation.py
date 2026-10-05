"""Validate the local merged IEEE-CIS training dataset."""

import logging
from pathlib import Path

from fraud_detection.data import (
    load_train_identity,
    load_train_transaction,
    merge_train_data,
    validate_training_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    """Load, merge, validate, and display a concise data-quality summary."""
    logging.basicConfig(level=logging.WARNING)

    try:
        transaction = load_train_transaction(RAW_DATA_DIR)
        identity = load_train_identity(RAW_DATA_DIR)
        merged = merge_train_data(transaction, identity)
        result = validate_training_data(merged)
    except ValueError as error:
        print(f"Validation passed: no ({error})")
        raise SystemExit(1) from error

    print("Validation summary")
    print(f"Rows: {result.row_count:,}")
    print(f"Columns: {result.column_count:,}")
    print(f"Fraud count: {result.fraud_count:,}")
    print(f"Non-fraud count: {result.non_fraud_count:,}")
    print(f"Fraud percentage: {result.fraud_rate:.2%}")
    print(f"Overall missing percentage: {result.overall_missing_rate:.2%}")
    print("Top 10 columns by missing percentage:")
    for column, rate in result.top_missing_columns.items():
        print(f"  {column}: {rate:.2%}")
    print(f"Validation passed: {'yes' if result.validation_passed else 'no'}")


if __name__ == "__main__":
    main()
