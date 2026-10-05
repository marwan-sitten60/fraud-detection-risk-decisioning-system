"""Create and summarize a chronological split of local IEEE-CIS training data."""

import logging
from pathlib import Path

from fraud_detection.data import (
    load_train_identity,
    load_train_transaction,
    merge_train_data,
    split_data_chronologically,
    summarize_split,
    validate_training_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    """Run the complete data preparation path through temporal splitting."""
    logging.basicConfig(level=logging.WARNING)

    transaction = load_train_transaction(RAW_DATA_DIR)
    identity = load_train_identity(RAW_DATA_DIR)
    merged = merge_train_data(transaction, identity)
    del transaction, identity

    validate_training_data(merged)
    split = split_data_chronologically(merged)
    total_rows = len(merged)

    print("Temporal split summary")
    for name, dataframe in (
        ("TRAIN", split.train),
        ("VALIDATION", split.validation),
        ("HOLDOUT", split.holdout),
    ):
        summary = summarize_split(dataframe, total_rows)
        print(
            f"{name}: rows={summary.row_count:,}, percentage={summary.percentage:.2%}, "
            f"fraud_count={summary.fraud_count:,}, fraud_percentage={summary.fraud_rate:.2%}, "
            f"TransactionDT=[{summary.transaction_dt_min}, {summary.transaction_dt_max}]"
        )

    print(f"TransactionID overlap exists: {'yes' if split.transaction_id_overlap else 'no'}")
    print(f"Temporal boundaries valid: {'yes' if split.temporal_boundaries_valid else 'no'}")


if __name__ == "__main__":
    main()
