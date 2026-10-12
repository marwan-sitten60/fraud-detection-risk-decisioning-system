"""Input checks for label-free frozen inference."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

FORBIDDEN_FIELDS = frozenset({"isFraud", "y_true"})
IDENTITY_FIELDS = ("TransactionID", "TransactionDT", "TransactionAmt")


def validate_transactions(
    rows: pd.DataFrame | Sequence[Mapping[str, object]],
    required_columns: tuple[str, ...],
    numeric_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Return a validated, ordered copy without repairing malformed input."""
    frame = rows.copy(deep=True) if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    forbidden = sorted(FORBIDDEN_FIELDS.intersection(frame.columns))
    if forbidden:
        raise ValueError(f"Inference input must not contain target fields: {forbidden}")
    if frame.empty:
        raise ValueError("Inference input must contain at least one transaction")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Inference input is missing required raw columns: {missing}")
    extra = sorted(set(frame.columns).difference(required_columns))
    if extra:
        raise ValueError(f"Inference input contains unexpected columns: {extra}")
    for column in numeric_columns:
        try:
            pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must contain numeric values or missing values") from exc
    for column in IDENTITY_FIELDS:
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{column} must contain finite, non-missing numeric values")
        if column != "TransactionAmt" and not (values == np.floor(values)).all():
            raise ValueError(f"{column} must contain integer values")
        if column == "TransactionAmt" and (values < 0).any():
            raise ValueError("TransactionAmt must be non-negative")
        frame[column] = values
    if frame.TransactionID.duplicated().any():
        raise ValueError("TransactionID must be unique within an inference batch")
    ordering = frame.loc[:, ["TransactionDT", "TransactionID"]]
    if not ordering.reset_index(drop=True).equals(
        ordering.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(
            drop=True
        )
    ):
        raise ValueError("Inference rows must be ordered by TransactionDT, then TransactionID")
    return frame.loc[:, list(required_columns)]
