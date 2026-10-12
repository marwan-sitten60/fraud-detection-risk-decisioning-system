"""Label-free frozen inference artifacts and terminal evaluation helpers."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def prediction_artifact(
    features, lightgbm_score, xgboost_score, ensemble_score, calibrated_probability, decisions
):
    """Build the immutable-style prediction record without accepting labels."""
    required = ("TransactionID", "TransactionDT", "TransactionAmt")
    if any(column not in features for column in required):
        raise ValueError("Missing prediction identifier, time, or amount")
    if len(features) != len(decisions) or not np.array_equal(
        features.TransactionID.to_numpy(), decisions.TransactionID.to_numpy()
    ):
        raise ValueError("Decisions and feature rows must align")
    if features.TransactionID.isna().any() or features.TransactionID.duplicated().any():
        raise ValueError("TransactionID must be present and unique")
    arrays = [
        np.asarray(values, dtype=float)
        for values in (lightgbm_score, xgboost_score, ensemble_score, calibrated_probability)
    ]
    if any(
        values.ndim != 1
        or len(values) != len(features)
        or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
        for values in arrays
    ):
        raise ValueError("Scores must be aligned probabilities")
    if not decisions.decision.isin(("APPROVE", "REVIEW", "BLOCK")).all():
        raise ValueError("Unknown business action")
    return pd.DataFrame(
        {
            "TransactionID": features.TransactionID.to_numpy(copy=True),
            "TransactionDT": features.TransactionDT.to_numpy(copy=True),
            "TransactionAmt": features.TransactionAmt.to_numpy(copy=True),
            "lightgbm_score": arrays[0].copy(),
            "xgboost_score": arrays[1].copy(),
            "ensemble_score": arrays[2].copy(),
            "calibrated_probability": arrays[3].copy(),
            "base_decision": decisions.decision.to_numpy(copy=True),
        }
    )


def diagnostic_cutoff_metrics(labels, probability, cutoff=0.5):
    """Evaluate a diagnostic threshold only after predictions are locked."""
    y = np.asarray(labels)
    p = np.asarray(probability, dtype=float)
    if (
        y.ndim != 1
        or p.ndim != 1
        or len(y) != len(p)
        or not np.isin(y, (0, 1)).all()
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("Labels and probabilities must align")
    predicted = p >= cutoff
    return {
        "tp": int(((y == 1) & predicted).sum()),
        "fp": int(((y == 0) & predicted).sum()),
        "fn": int(((y == 1) & ~predicted).sum()),
        "tn": int(((y == 0) & ~predicted).sum()),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
    }
