"""Illustrative expected business costs; values are replaceable scenario assumptions."""

from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class BusinessCostConfig:
    """Serializable demonstration assumptions, never actual company costs."""

    fraud_loss_multiplier: float
    manual_review_cost: float
    review_fraud_detection_rate: float
    legitimate_review_friction_cost: float
    false_decline_cost: float
    block_operational_cost: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.review_fraud_detection_rate > 1:
            raise ValueError("review_fraud_detection_rate must be in [0, 1]")


def scenario_profiles():
    """Exactly three fixed, illustrative profiles; no data-driven tuning."""
    return {
        "Balanced": BusinessCostConfig(1.0, 3.0, 0.8, 1.0, 25.0, 0.5),
        "Fraud-loss-sensitive": BusinessCostConfig(2.0, 3.0, 0.8, 1.0, 20.0, 0.5),
        "Customer-friction-sensitive": BusinessCostConfig(1.0, 5.0, 0.75, 5.0, 75.0, 0.5),
    }


def expected_action_costs(calibrated_probability, transaction_amount, config):
    """Return approve, review, block costs in the same order as the inputs."""
    if not isinstance(config, BusinessCostConfig):
        raise TypeError("config must be BusinessCostConfig")
    p = np.asarray(calibrated_probability, dtype=float)
    amount = np.asarray(transaction_amount, dtype=float)
    if p.ndim != 1 or amount.ndim != 1 or len(p) != len(amount) or not len(p):
        raise ValueError("Probability and amount must be aligned nonempty vectors")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("calibrated_probability must be finite and in [0, 1]")
    if not np.isfinite(amount).all() or (amount < 0).any():
        raise ValueError("TransactionAmt must be finite and nonnegative")
    exposure = amount * config.fraud_loss_multiplier
    approve = p * exposure
    review = (
        config.manual_review_cost
        + p * (1 - config.review_fraud_detection_rate) * exposure
        + (1 - p) * config.legitimate_review_friction_cost
    )
    block = (1 - p) * config.false_decline_cost + config.block_operational_cost
    return approve, review, block
