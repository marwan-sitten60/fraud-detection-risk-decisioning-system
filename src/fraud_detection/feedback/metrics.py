"""Metrics on mature labels joined to original stored predictions only."""

from datetime import datetime

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from fraud_detection.models.calibration import reliability_bins


def maturity_status(matured_at: datetime | None, evaluation_time: datetime) -> str:
    return "MATURE" if matured_at is not None and evaluation_time >= matured_at else "IMMATURE"


def delay_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "median_seconds": None, "p90_seconds": None, "max_seconds": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "median_seconds": float(np.median(array)),
        "p90_seconds": float(np.percentile(array, 90)),
        "max_seconds": float(array.max()),
    }


def calculate(rows: list[dict]) -> dict:
    """Use stored probability, amount, and action; never invoke inference."""
    n = len(rows)
    decisions = {
        action: sum(row["decision"] == action for row in rows)
        for action in ("APPROVE", "REVIEW", "BLOCK")
    }
    output = {
        "sample_size": n,
        "fraud_prevalence": None,
        "pr_auc": None,
        "roc_auc": None,
        "ranking_unavailable_reason": None,
        "brier": None,
        "log_loss": None,
        "ece": None,
        "mean_predicted_fraud_probability": None,
        "stored_score_means": None,
        "stored_expected_cost_means": None,
        "triggered_rule_counts": {},
        "diagnostic_cutoff": 0.5,
        "diagnostic_confusion": None,
        "diagnostic_precision": None,
        "diagnostic_recall": None,
        "diagnostic_f1": None,
        "decision_distribution": {
            action: {"count": count, "rate": count / n if n else None}
            for action, count in decisions.items()
        },
        "fraud_transaction_capture": None,
        "fraud_amount_capture": None,
        "fraudulent_transactions_approved": 0,
        "fraudulent_amount_approved": 0.0,
        "legitimate_review_rate": None,
        "legitimate_block_rate": None,
    }
    if not n:
        output["ranking_unavailable_reason"] = "No mature labels"
        return output
    y = np.asarray([int(row["is_fraud"]) for row in rows], dtype=int)
    p = np.asarray([row["probability"] for row in rows], dtype=float)
    output.update(
        fraud_prevalence=float(y.mean()),
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        mean_predicted_fraud_probability=float(p.mean()),
    )
    score_names = ("lightgbm_score", "xgboost_score", "ensemble_score")
    cost_names = (
        "expected_cost_approve",
        "expected_cost_review",
        "expected_cost_block",
    )
    if all(all(name in row for name in (*score_names, *cost_names)) for row in rows):
        output["stored_score_means"] = {
            name: float(np.mean([row[name] for row in rows])) for name in score_names
        }
        output["stored_expected_cost_means"] = {
            name: float(np.mean([row[name] for row in rows])) for name in cost_names
        }
    rules = {}
    for row in rows:
        for rule in row.get("triggered_rule_ids", []):
            rules[rule] = rules.get(rule, 0) + 1
    output["triggered_rule_counts"] = rules
    bins = reliability_bins(y, p)
    output["ece"] = float(
        np.average(abs(bins.mean_probability - bins.fraud_rate), weights=bins["count"])
    )
    if len(set(y)) == 2:
        output["pr_auc"] = float(average_precision_score(y, p))
        output["roc_auc"] = float(roc_auc_score(y, p))
    else:
        output["ranking_unavailable_reason"] = "Ranking requires both fraud and legitimate labels"
    classified = p >= 0.5
    tp = int(((y == 1) & classified).sum())
    fp = int(((y == 0) & classified).sum())
    fn = int(((y == 1) & ~classified).sum())
    tn = int(((y == 0) & ~classified).sum())
    output["diagnostic_confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    output["diagnostic_precision"] = precision
    output["diagnostic_recall"] = recall
    output["diagnostic_f1"] = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    fraud = [row for row in rows if row["is_fraud"]]
    legitimate = [row for row in rows if not row["is_fraud"]]
    captured = [row for row in fraud if row["decision"] in ("REVIEW", "BLOCK")]
    approved = [row for row in fraud if row["decision"] == "APPROVE"]
    fraud_amount = sum(row["amount"] for row in fraud)
    output["fraud_transaction_capture"] = len(captured) / len(fraud) if fraud else None
    output["fraud_amount_capture"] = (
        sum(row["amount"] for row in captured) / fraud_amount if fraud_amount > 0 else None
    )
    output["fraudulent_transactions_approved"] = len(approved)
    output["fraudulent_amount_approved"] = sum(row["amount"] for row in approved)
    output["legitimate_review_rate"] = (
        sum(row["decision"] == "REVIEW" for row in legitimate) / len(legitimate)
        if legitimate
        else None
    )
    output["legitimate_block_rate"] = (
        sum(row["decision"] == "BLOCK" for row in legitimate) / len(legitimate)
        if legitimate
        else None
    )
    return output
