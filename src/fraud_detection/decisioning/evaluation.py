"""Count, amount, and illustrative economic policy metrics."""

import numpy as np
import pandas as pd

from fraud_detection.decisioning.costs import BusinessCostConfig, expected_action_costs


def evaluate_policy(decisions: pd.DataFrame, labels, config: BusinessCostConfig):
    """Evaluate actual label groups and modeled expected costs separately."""
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != len(decisions) or not np.isin(y, (0, 1)).all():
        raise ValueError("labels must align and be binary")
    actions = decisions.decision.to_numpy()
    if not np.isin(actions, ("APPROVE", "REVIEW", "BLOCK")).all():
        raise ValueError("Unknown decision")
    amount = decisions.TransactionAmt.to_numpy(dtype=float)
    fraud = y == 1
    legitimate = ~fraud
    approve = actions == "APPROVE"
    review = actions == "REVIEW"
    block = actions == "BLOCK"
    total_fraud_amount = float(amount[fraud].sum())
    fraud_captured = fraud & block
    fraud_reviewed = fraud & review
    modeled_capture_count = float(
        fraud_captured.sum() + config.review_fraud_detection_rate * fraud_reviewed.sum()
    )
    modeled_capture_amount = float(
        amount[fraud_captured].sum()
        + config.review_fraud_detection_rate * amount[fraud_reviewed].sum()
    )
    baseline = expected_action_costs(decisions.calibrated_probability, amount, config)[0]
    modeled_cost = float(decisions.selected_expected_cost.sum())
    n = len(decisions)
    result = {
        "n": n,
        "fraud_count": int(fraud.sum()),
        "legitimate_count": int(legitimate.sum()),
        "fraud_amount_total": total_fraud_amount,
        "fraud_capture_rate": modeled_capture_count / fraud.sum() if fraud.any() else 0.0,
        "fraud_amount_capture_rate": modeled_capture_amount / total_fraud_amount
        if total_fraud_amount
        else 0.0,
        "fraud_transactions_approved": int((fraud & approve).sum()),
        "fraud_transactions_reviewed": int(fraud_reviewed.sum()),
        "fraud_transactions_blocked": int(fraud_captured.sum()),
        "fraudulent_amount_approved": float(amount[fraud & approve].sum()),
        "fraudulent_amount_reviewed": float(amount[fraud_reviewed].sum()),
        "fraudulent_amount_blocked": float(amount[fraud_captured].sum()),
        "false_decline_count": int((legitimate & block).sum()),
        "false_decline_rate": float((legitimate & block).sum() / legitimate.sum())
        if legitimate.any()
        else 0.0,
        "modeled_total_cost": modeled_cost,
        "modeled_cost_per_transaction": modeled_cost / n,
        "approve_all_modeled_cost": float(baseline.sum()),
        "approve_all_modeled_cost_per_transaction": float(baseline.mean()),
        "modeled_cost_reduction": float(baseline.sum() - modeled_cost),
        "approve_all_fraud_transactions_missed": int(fraud.sum()),
        "approve_all_fraudulent_amount_approved": total_fraud_amount,
        "review_overflow_count": int(decisions.review_overflowed.sum()),
    }
    for action, mask in (("approve", approve), ("review", review), ("block", block)):
        result[f"{action}_count"] = int(mask.sum())
        result[f"{action}_rate"] = float(mask.mean())
        result[f"legitimate_{action}_rate"] = (
            float((legitimate & mask).sum() / legitimate.sum()) if legitimate.any() else 0.0
        )
    return result
