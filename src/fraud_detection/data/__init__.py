"""Data ingestion utilities for the IEEE-CIS dataset."""

from fraud_detection.data.loading import (
    load_test_identity,
    load_test_transaction,
    load_train_identity,
    load_train_transaction,
)
from fraud_detection.data.merging import merge_train_data

__all__ = [
    "load_test_identity",
    "load_test_transaction",
    "load_train_identity",
    "load_train_transaction",
    "merge_train_data",
]
