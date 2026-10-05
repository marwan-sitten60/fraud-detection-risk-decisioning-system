"""Verify the training transaction and identity merge on local IEEE-CIS data."""

import logging
from pathlib import Path

from fraud_detection.data import load_train_identity, load_train_transaction, merge_train_data

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    """Load, merge, and summarize the IEEE-CIS training data."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    transaction = load_train_transaction(RAW_DATA_DIR)
    identity = load_train_identity(RAW_DATA_DIR)
    transaction_shape = transaction.shape
    identity_shape = identity.shape
    matched_rows = int(transaction["TransactionID"].isin(identity["TransactionID"]).sum())
    unmatched_rows = len(transaction) - matched_rows

    merged = merge_train_data(transaction, identity)

    LOGGER.info("Original transaction shape: %s", transaction_shape)
    LOGGER.info("Original identity shape: %s", identity_shape)
    LOGGER.info("Merged shape: %s", merged.shape)
    LOGGER.info("Matched identity rows: %d", matched_rows)
    LOGGER.info("Unmatched transaction rows: %d", unmatched_rows)
    LOGGER.info("First columns: %s", merged.columns[:5].tolist())


if __name__ == "__main__":
    main()
