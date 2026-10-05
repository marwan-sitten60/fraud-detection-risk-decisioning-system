"""Tests for the IEEE-CIS CSV loading functions."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from fraud_detection.data import (
    load_test_identity,
    load_test_transaction,
    load_train_identity,
    load_train_transaction,
)

Loader = Callable[[Path], pd.DataFrame]

LOADERS: tuple[tuple[Loader, str], ...] = (
    (load_train_transaction, "train_transaction.csv"),
    (load_train_identity, "train_identity.csv"),
    (load_test_transaction, "test_transaction.csv"),
    (load_test_identity, "test_identity.csv"),
)


@pytest.mark.parametrize(("loader", "filename"), LOADERS)
def test_load_dataset(loader: Loader, filename: str, tmp_path: Path) -> None:
    """Each loader should read its corresponding CSV file."""
    expected = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "feature": [10.5, 20.5],
        }
    )
    expected.to_csv(tmp_path / filename, index=False)

    result = loader(tmp_path)

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(("loader", "filename"), LOADERS)
def test_load_dataset_raises_for_missing_file(
    loader: Loader, filename: str, tmp_path: Path
) -> None:
    """Each loader should identify its missing expected file."""
    with pytest.raises(FileNotFoundError, match=filename):
        loader(tmp_path)
