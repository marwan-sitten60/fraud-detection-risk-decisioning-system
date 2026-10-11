"""Validation-only diagnostic summaries for a frozen calibrated risk system."""

import numpy as np
import pandas as pd

from fraud_detection.models.calibration import probabilities, targets

RISK_EDGES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0)
RISK_LABELS = ("0–1%", "1–2%", "2–5%", "5–10%", "10–20%", "20–40%", "40%+")


def join_decisions(frame: pd.DataFrame, decisions: pd.DataFrame):
    """Join exactly by unique row ID without changing either source."""
    if frame.TransactionID.duplicated().any() or decisions.TransactionID.duplicated().any():
        raise ValueError("TransactionID must be unique on both sides")
    result = frame.merge(decisions, on="TransactionID", how="left", validate="one_to_one")
    if len(result) != len(frame) or result.decision.isna().any():
        raise ValueError("Decision rows must align exactly")
    return result


def error_classes(label, calibrated_probability, *, cutoff=0.50):
    """Diagnostic-only cutoff; never a business policy threshold."""
    y, p = targets(label), probabilities(calibrated_probability)
    if len(y) != len(p) or not 0 <= cutoff <= 1:
        raise ValueError("Targets, probabilities, and cutoff are invalid")
    positive = p >= cutoff
    return np.select(
        [((y == 1) & positive), ((y == 0) & positive), ((y == 1) & ~positive)],
        ["TP", "FP", "FN"],
        default="TN",
    )


def amount_bands(amount, train_edges):
    values = np.asarray(amount, dtype=float)
    edges = np.asarray(train_edges, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Amounts must be a finite nonnegative vector")
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("TRAIN amount edges must be strictly increasing")
    return np.searchsorted(edges[1:-1], values, side="right")


def risk_bands(calibrated_probability):
    p = probabilities(calibrated_probability)
    return pd.Categorical.from_codes(
        np.searchsorted(RISK_EDGES[1:-1], p, side="right"), categories=RISK_LABELS, ordered=True
    )


def _checked_frame(frame):
    required = {"isFraud", "TransactionAmt", "calibrated_probability", "decision"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing diagnostic columns: {sorted(missing)}")
    targets(frame.isFraud)
    probabilities(frame.calibrated_probability)
    amount = frame.TransactionAmt.to_numpy(dtype=float)
    if not np.isfinite(amount).all() or (amount < 0).any():
        raise ValueError("TransactionAmt must be finite and nonnegative")


def amount_band_summary(frame, train_edges, *, cutoff=0.50):
    _checked_frame(frame)
    bands = amount_bands(frame.TransactionAmt, train_edges)
    classes = error_classes(frame.isFraud, frame.calibrated_probability, cutoff=cutoff)
    rows = []
    for band in range(len(train_edges) - 1):
        mask = bands == band
        y = frame.isFraud.to_numpy()[mask]
        amount = frame.TransactionAmt.to_numpy(dtype=float)[mask]
        cls = classes[mask]
        counts = {name: int((cls == name).sum()) for name in ("TP", "FP", "FN", "TN")}
        fraud = y == 1
        rows.append(
            {
                "band": band,
                "amount_min": float(train_edges[band]),
                "amount_max": float(train_edges[band + 1]),
                "count": int(mask.sum()),
                "fraud_prevalence": float(fraud.mean()) if len(y) else np.nan,
                **counts,
                "recall": counts["TP"] / (counts["TP"] + counts["FN"]) if fraud.any() else np.nan,
                "precision": counts["TP"] / (counts["TP"] + counts["FP"])
                if counts["TP"] + counts["FP"]
                else np.nan,
                "false_positive_rate": counts["FP"] / (counts["FP"] + counts["TN"])
                if counts["FP"] + counts["TN"]
                else np.nan,
                "false_negative_rate": counts["FN"] / (counts["FN"] + counts["TP"])
                if fraud.any()
                else np.nan,
                "fraud_amount": float(amount[fraud].sum()),
                "missed_fraud_amount": float(amount[cls == "FN"].sum()),
            }
        )
    return pd.DataFrame(rows)


def risk_band_summary(frame):
    _checked_frame(frame)
    band = risk_bands(frame.calibrated_probability)
    rows = []
    for label in RISK_LABELS:
        subset = frame.loc[band == label]
        if subset.empty:
            continue
        fraud = subset.isFraud == 1
        rows.append(
            {
                "risk_band": label,
                "count": len(subset),
                "fraud_count": int(fraud.sum()),
                "observed_fraud_rate": float(fraud.mean()),
                "mean_calibrated_probability": float(subset.calibrated_probability.mean()),
                "transaction_amount": float(subset.TransactionAmt.sum()),
                "fraud_amount": float(subset.loc[fraud, "TransactionAmt"].sum()),
                **{
                    f"{action.lower()}_rate": float((subset.decision == action).mean())
                    for action in ("APPROVE", "REVIEW", "BLOCK")
                },
            }
        )
    return pd.DataFrame(rows)


def categorical_slices(frame, column, *, min_support=500, cutoff=0.50):
    _checked_frame(frame)
    if column not in frame or min_support < 1:
        raise ValueError("Category column and positive minimum support are required")
    classes = error_classes(frame.isFraud, frame.calibrated_probability, cutoff=cutoff)
    working = frame.loc[
        :, [column, "isFraud", "TransactionAmt", "calibrated_probability", "decision"]
    ].copy()
    working["error_class"] = classes
    working[column] = working[column].astype("string").fillna("<MISSING>")
    rows = []
    for category, group in working.groupby(column, sort=True):
        if len(group) < min_support:
            continue
        fraud = group.isFraud == 1
        legitimate = ~fraud
        captured = (group.decision == "BLOCK") | (group.decision == "REVIEW")
        fraud_amount = group.loc[fraud, "TransactionAmt"].sum()
        rows.append(
            {
                "column": column,
                "category": str(category),
                "count": len(group),
                "fraud_prevalence": float(fraud.mean()),
                "mean_calibrated_risk": float(group.calibrated_probability.mean()),
                "recall_at_0_50": float((group.error_class == "TP").sum() / fraud.sum())
                if fraud.any()
                else np.nan,
                "legitimate_intervention_rate": float(captured[legitimate].mean())
                if legitimate.any()
                else np.nan,
                "fraud_amount_intervention_rate": float(
                    group.loc[fraud & captured, "TransactionAmt"].sum() / fraud_amount
                )
                if fraud_amount
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def disagreement_summary(
    label, amount, lightgbm_score, xgboost_score, calibrated_probability, *, high_fraction=0.10
):
    y, a = targets(label), np.asarray(amount, dtype=float)
    lightgbm, x, p = (
        probabilities(v) for v in (lightgbm_score, xgboost_score, calibrated_probability)
    )
    if not (len(y) == len(a) == len(lightgbm) == len(x) == len(p)) or not 0 < high_fraction <= 1:
        raise ValueError("Disagreement inputs must align")
    gap = np.abs(lightgbm - x)
    rank_gap = np.abs(
        pd.Series(lightgbm).rank(pct=True).to_numpy() - pd.Series(x).rank(pct=True).to_numpy()
    )
    threshold = float(np.quantile(gap, 1 - high_fraction))
    high = gap >= threshold
    fn = (y == 1) & (p < 0.50)
    return {
        "pearson": float(pd.Series(lightgbm).corr(pd.Series(x))),
        "spearman": float(pd.Series(lightgbm).corr(pd.Series(x), method="spearman")),
        "mean_absolute_gap": float(gap.mean()),
        "mean_absolute_rank_gap": float(rank_gap.mean()),
        "p90_absolute_rank_gap": float(np.quantile(rank_gap, 0.90)),
        "rank_gap_ge_0_20_rate": float((rank_gap >= 0.20).mean()),
        "median_absolute_gap": float(np.median(gap)),
        "p90_absolute_gap": float(np.quantile(gap, 0.90)),
        "high_gap_threshold": threshold,
        "high_gap_count": int(high.sum()),
        "high_gap_fraud_rate": float(y[high].mean()),
        "high_gap_false_negative_rate": float(fn[high].mean()),
        "high_gap_fraud_amount": float(a[high & (y == 1)].sum()),
        "xgboost_higher_on_lgb_low_fraud_rate": float(
            (x[(y == 1) & (lightgbm < 0.5)] > lightgbm[(y == 1) & (lightgbm < 0.5)]).mean()
        )
        if ((y == 1) & (lightgbm < 0.5)).any()
        else np.nan,
    }


def representative_cases(frame, *, high_value_amount):
    """Return deterministic row positions for anonymized diagnostic examples."""
    _checked_frame(frame)
    classes = error_classes(frame.isFraud, frame.calibrated_probability)
    p = frame.calibrated_probability.to_numpy()
    amount = frame.TransactionAmt.to_numpy(dtype=float)
    action = frame.decision.to_numpy()
    specifications = {
        "high_risk_tp": ((classes == "TP"), -p),
        "false_positive": ((classes == "FP"), -p),
        "false_negative": ((classes == "FN"), p),
        "low_risk_tn": ((classes == "TN"), p),
        "review": ((action == "REVIEW"), -p),
        "block": ((action == "BLOCK"), -p),
        "high_value_fraud_miss": ((classes == "FN") & (amount >= high_value_amount), -amount),
    }
    result = {}
    for name, (mask, priority) in specifications.items():
        positions = np.flatnonzero(mask)
        if len(positions):
            result[name] = int(sorted(positions, key=lambda i: (priority[i], i))[0])
    return result
