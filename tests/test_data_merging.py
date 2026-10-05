"""Tests for safe IEEE-CIS training data merging."""

import pandas as pd
import pytest

from fraud_detection.data import merge_train_data


def test_merge_train_data_uses_left_join_and_preserves_columns() -> None:
    """Identity values should join to matching transactions without column loss."""
    transaction = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "isFraud": [0, 1],
            "amount": [10.0, 20.0],
        }
    )
    identity = pd.DataFrame(
        {
            "TransactionID": [2, 3],
            "device": ["mobile", "desktop"],
        }
    )

    result = merge_train_data(transaction, identity)

    assert result.shape == (2, 4)
    assert result["TransactionID"].tolist() == [1, 2]
    assert result["isFraud"].tolist() == [0, 1]
    assert pd.isna(result.loc[0, "device"])
    assert result.loc[1, "device"] == "mobile"


def test_merge_train_data_preserves_transactions_without_identity() -> None:
    """Every transaction should remain when no identity record exists."""
    transaction = pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0, 1]})
    identity = pd.DataFrame({"TransactionID": [3], "device": ["mobile"]})

    result = merge_train_data(transaction, identity)

    assert len(result) == len(transaction)
    assert result["TransactionID"].tolist() == transaction["TransactionID"].tolist()
    assert result["device"].isna().all()


@pytest.mark.parametrize("missing_from", ["transaction", "identity"])
def test_merge_train_data_requires_transaction_id(missing_from: str) -> None:
    """Both input tables should contain the join key."""
    transaction = pd.DataFrame({"TransactionID": [1], "isFraud": [0]})
    identity = pd.DataFrame({"TransactionID": [1], "device": ["mobile"]})
    if missing_from == "transaction":
        transaction = transaction.drop(columns="TransactionID")
    else:
        identity = identity.drop(columns="TransactionID")

    with pytest.raises(ValueError, match=rf"{missing_from}.*TransactionID"):
        merge_train_data(transaction, identity)


def test_merge_train_data_rejects_duplicate_transaction_ids() -> None:
    """Duplicate primary transaction keys should fail clearly."""
    transaction = pd.DataFrame({"TransactionID": [1, 1], "isFraud": [0, 1]})
    identity = pd.DataFrame({"TransactionID": [1], "device": ["mobile"]})

    with pytest.raises(ValueError, match=r"unique in the transaction DataFrame"):
        merge_train_data(transaction, identity)


def test_merge_train_data_rejects_identity_duplicates() -> None:
    """Duplicate identity keys should not be allowed to multiply transaction rows."""
    transaction = pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0, 1]})
    identity = pd.DataFrame(
        {
            "TransactionID": [1, 1],
            "device": ["mobile", "desktop"],
        }
    )

    with pytest.raises(ValueError, match=r"prevent row multiplication"):
        merge_train_data(transaction, identity)
