"""Safe merge operations for IEEE-CIS datasets."""

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)
TRANSACTION_ID = "TransactionID"


def _require_transaction_id(dataframe: pd.DataFrame, table_name: str) -> None:
    """Ensure a dataframe contains the merge key."""
    if TRANSACTION_ID not in dataframe.columns:
        raise ValueError(f"{table_name} DataFrame is missing required column: {TRANSACTION_ID}")


def _duplicate_ids(dataframe: pd.DataFrame) -> list[object]:
    """Return a short sample of duplicated transaction IDs."""
    duplicate_mask = dataframe[TRANSACTION_ID].duplicated(keep=False)
    return dataframe.loc[duplicate_mask, TRANSACTION_ID].drop_duplicates().head(5).tolist()


def merge_train_data(
    transaction: pd.DataFrame,
    identity: pd.DataFrame,
) -> pd.DataFrame:
    """Left join training transaction and identity data without multiplying rows."""
    LOGGER.info("Transaction shape before merge: %s", transaction.shape)
    LOGGER.info("Identity shape before merge: %s", identity.shape)

    _require_transaction_id(transaction, "transaction")
    _require_transaction_id(identity, "identity")

    transaction_duplicates = _duplicate_ids(transaction)
    if transaction_duplicates:
        raise ValueError(
            "TransactionID must be unique in the transaction DataFrame; "
            f"duplicate values include {transaction_duplicates}"
        )

    identity_duplicates = _duplicate_ids(identity)
    if identity_duplicates:
        raise ValueError(
            "TransactionID must be unique in the identity DataFrame to prevent row "
            f"multiplication; duplicate values include {identity_duplicates}"
        )

    matched_rows = int(transaction[TRANSACTION_ID].isin(identity[TRANSACTION_ID]).sum())
    unmatched_rows = len(transaction) - matched_rows

    merged = transaction.merge(
        identity,
        how="left",
        on=TRANSACTION_ID,
        suffixes=("", "_identity"),
        sort=False,
        validate="one_to_one",
    )

    if len(merged) != len(transaction):
        raise RuntimeError(
            "Unexpected row multiplication during training data merge: "
            f"expected {len(transaction)} rows, received {len(merged)}"
        )

    LOGGER.info("Merged shape: %s", merged.shape)
    LOGGER.info("Transaction rows with matching identity data: %d", matched_rows)
    LOGGER.info("Transaction rows without identity data: %d", unmatched_rows)
    return merged
