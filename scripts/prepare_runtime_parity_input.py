"""Save a label-free chronological HOLDOUT prefix for software parity checks."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection.data import merge_train_data  # noqa: E402

FULL_ROWS = 590540
TRAIN_END = int(FULL_ROWS * 0.70)
VALIDATION_END = TRAIN_END + int(FULL_ROWS * 0.15)
PREFIX_ROWS = 512


def main() -> None:
    destination = ROOT / "reports/runtime_golden_input_prefix.parquet"
    if destination.exists():
        raise FileExistsError(f"Refusing to replace label-free parity input: {destination}")
    transaction_path = ROOT / "data/raw/train_transaction.csv"
    sample = pd.read_csv(transaction_path, nrows=1000, low_memory=False)
    dtypes = {
        name: ("float64" if pd.api.types.is_numeric_dtype(dtype) else "object")
        for name, dtype in sample.dtypes.items()
        if name != "isFraud"
    }
    dtypes.update({"TransactionID": "int64", "TransactionDT": "int64", "card1": "int64"})
    transaction = pd.read_csv(
        transaction_path,
        skiprows=range(1, VALIDATION_END + 1),
        nrows=PREFIX_ROWS,
        usecols=lambda name: name != "isFraud",
        dtype=dtypes,
    )
    assert len(transaction) == PREFIX_ROWS and "isFraud" not in transaction
    identity = pd.read_csv(ROOT / "data/raw/train_identity.csv", low_memory=False)
    identity = identity.loc[identity.TransactionID.isin(transaction.TransactionID)]
    merged = merge_train_data(transaction, identity)
    assert len(merged) == PREFIX_ROWS and "isFraud" not in merged
    merged.to_parquet(destination, index=False)
    print(f"Saved {len(merged)} label-free chronological rows to {destination}")


if __name__ == "__main__":
    main()
