"""Cost-minimizing three-action policy with deterministic review capacity."""

import numpy as np
import pandas as pd

from fraud_detection.decisioning.costs import BusinessCostConfig, expected_action_costs

ACTIONS = np.array(["APPROVE", "REVIEW", "BLOCK"])
REQUIRED = ("TransactionID", "calibrated_probability", "TransactionAmt")


def decide(frame, config: BusinessCostConfig, *, maximum_review_rate=None, overflow="cheapest"):
    """Choose minimum expected cost; capacity ranks review benefit, then row ID."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Decision input must be a DataFrame")
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Missing decision columns: {sorted(missing)}; calibrated probability required"
        )
    if frame.TransactionID.isna().any() or frame.TransactionID.duplicated().any():
        raise ValueError("TransactionID must be present and unique")
    if maximum_review_rate is not None and (
        not np.isfinite(maximum_review_rate) or not 0 <= maximum_review_rate <= 1
    ):
        raise ValueError("maximum_review_rate must be in [0, 1]")
    if overflow not in ("cheapest", "approve", "block"):
        raise ValueError("overflow must be cheapest, approve, or block")
    approve, review, block = expected_action_costs(
        frame.calibrated_probability, frame.TransactionAmt, config
    )
    costs = np.column_stack((approve, review, block))
    decision = ACTIONS[np.argmin(costs, axis=1)].copy()
    overflowed = np.zeros(len(frame), dtype=bool)
    priority = np.minimum(approve, block) - review
    if maximum_review_rate is not None:
        limit = int(np.floor(len(frame) * maximum_review_rate))
        candidates = np.flatnonzero(decision == "REVIEW")
        if len(candidates) > limit:
            ranked = sorted(candidates, key=lambda i: (-priority[i], frame.TransactionID.iloc[i]))
            rejected = np.asarray(ranked[limit:], dtype=int)
            overflowed[rejected] = True
            if overflow == "cheapest":
                decision[rejected] = np.where(
                    approve[rejected] <= block[rejected], "APPROVE", "BLOCK"
                )
            else:
                decision[rejected] = overflow.upper()
    chosen_cost = np.where(
        decision == "APPROVE", approve, np.where(decision == "REVIEW", review, block)
    )
    return pd.DataFrame(
        {
            "TransactionID": frame.TransactionID.to_numpy(copy=True),
            "calibrated_probability": frame.calibrated_probability.to_numpy(copy=True),
            "TransactionAmt": frame.TransactionAmt.to_numpy(copy=True),
            "expected_cost_approve": approve,
            "expected_cost_review": review,
            "expected_cost_block": block,
            "review_priority_benefit": priority,
            "review_overflowed": overflowed,
            "decision": decision,
            "selected_expected_cost": chosen_cost,
        },
        index=frame.index.copy(),
    )
