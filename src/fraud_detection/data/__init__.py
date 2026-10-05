"""Data ingestion utilities for the IEEE-CIS dataset."""

from fraud_detection.data.loading import (
    load_test_identity,
    load_test_transaction,
    load_train_identity,
    load_train_transaction,
)

__all__ = [
    "load_test_identity",
    "load_test_transaction",
    "load_train_identity",
    "load_train_transaction",
]
