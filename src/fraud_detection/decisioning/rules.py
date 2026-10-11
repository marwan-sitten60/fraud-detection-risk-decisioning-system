"""Illustrative, deterministic action overrides around a frozen fraud score."""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

Action = Literal["APPROVE", "REVIEW", "BLOCK"]
RuleType = Literal["HARD", "SOFT"]


@dataclass(frozen=True, slots=True)
class RuleThresholds:
    """TRAIN-only distribution cutoffs; serialize with ``asdict``."""

    high_amount: float
    extreme_amount: float
    very_high_risk: float
    elevated_risk: float
    high_disagreement: float
    rare_composite_frequency: float
    nontrivial_amount: float


@dataclass(frozen=True, slots=True)
class Rule:
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    priority: int
    action: Action
    reason_code: str
    predicate: str


@dataclass(frozen=True, slots=True)
class RuleResult:
    rule: Rule
    triggered: np.ndarray


def derive_thresholds(train_amount, train_oof: pd.DataFrame) -> RuleThresholds:
    """Derive fixed quantiles without reading labels or validation rows."""
    amount = np.asarray(train_amount, dtype=float)
    required = {
        "calibrated_probability",
        "lightgbm_score",
        "xgboost_score",
        "composite_frequency",
    }
    if required - set(train_oof):
        raise ValueError(f"Missing TRAIN OOF columns: {sorted(required - set(train_oof))}")
    if amount.ndim != 1 or not len(amount) or not np.isfinite(amount).all():
        raise ValueError("TRAIN amounts must be a finite vector")
    p = train_oof.calibrated_probability.to_numpy(dtype=float)
    gap = np.abs(
        train_oof.lightgbm_score.to_numpy(dtype=float)
        - train_oof.xgboost_score.to_numpy(dtype=float)
    )
    frequency = train_oof.composite_frequency.to_numpy(dtype=float)
    if not len(p) or not all(np.isfinite(x).all() for x in (p, gap, frequency)):
        raise ValueError("TRAIN OOF signals must be finite and nonempty")
    if ((p < 0) | (p > 1)).any() or (frequency < 0).any():
        raise ValueError("TRAIN OOF risk/frequency outside valid range")
    return RuleThresholds(
        high_amount=float(np.quantile(amount, 0.95)),
        extreme_amount=float(np.quantile(amount, 0.99)),
        very_high_risk=float(np.quantile(p, 0.99)),
        elevated_risk=float(np.quantile(p, 0.75)),
        high_disagreement=float(np.quantile(gap, 0.95)),
        rare_composite_frequency=float(np.quantile(frequency, 0.10)),
        nontrivial_amount=float(np.quantile(amount, 0.50)),
    )


def illustrative_rules() -> tuple[Rule, ...]:
    """Three fixed examples. None uses labels or an external feed."""
    return (
        Rule(
            "R01",
            "Extreme risk and high amount",
            "TRAIN p99 calibrated risk and TRAIN p95 transaction amount",
            "HARD",
            100,
            "BLOCK",
            "EXTREME_RISK_HIGH_AMOUNT",
            "extreme_risk_amount",
        ),
        Rule(
            "R02",
            "Component disagreement",
            "TRAIN p95 component disagreement with risk above TRAIN p75",
            "SOFT",
            60,
            "REVIEW",
            "MODEL_DISAGREEMENT",
            "disagreement_review",
        ),
        Rule(
            "R03",
            "Rare composite identity",
            "Rare TRAIN reference composite frequency, elevated risk, and nontrivial amount",
            "SOFT",
            50,
            "REVIEW",
            "RARE_IDENTITY_ELEVATED_RISK",
            "rare_identity_review",
        ),
    )


class RulesEngine:
    """Apply registered predicates and deterministic HARD > SOFT > base precedence."""

    REQUIRED = (
        "TransactionID",
        "calibrated_probability",
        "TransactionAmt",
        "lightgbm_score",
        "xgboost_score",
        "composite_frequency",
        "base_decision",
    )

    def __init__(
        self,
        thresholds: RuleThresholds,
        rules: tuple[Rule, ...] | None = None,
        predicates: Mapping[str, Callable[[pd.DataFrame, RuleThresholds], np.ndarray]]
        | None = None,
    ):
        self.thresholds = thresholds
        self.rules = illustrative_rules() if rules is None else tuple(rules)
        self.predicates = {**_PREDICATES, **(predicates or {})}
        ids = [rule.rule_id for rule in self.rules]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("Rule IDs must be unique and nonempty")
        for rule in self.rules:
            if rule.rule_type not in ("HARD", "SOFT") or rule.action not in ("REVIEW", "BLOCK"):
                raise ValueError("Unsupported rule type or action")
            if rule.rule_type == "HARD" and rule.action != "BLOCK":
                raise ValueError("HARD rules must force BLOCK")
            if rule.predicate not in self.predicates:
                raise ValueError(f"Unknown predicate: {rule.predicate}")

    def configuration(self) -> dict:
        return {"thresholds": asdict(self.thresholds), "rules": [asdict(r) for r in self.rules]}

    def evaluate(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.REQUIRED) - set(frame)
        if missing:
            raise ValueError(f"Missing rule inputs: {sorted(missing)}")
        if frame.TransactionID.isna().any() or frame.TransactionID.duplicated().any():
            raise ValueError("TransactionID must be unique and present")
        if not frame.base_decision.isin(("APPROVE", "REVIEW", "BLOCK")).all():
            raise ValueError("Unknown base decision")
        for column in (
            "calibrated_probability",
            "TransactionAmt",
            "lightgbm_score",
            "xgboost_score",
            "composite_frequency",
        ):
            if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
                raise ValueError(f"Invalid rule signal: {column}")
        if not frame.calibrated_probability.between(0, 1).all():
            raise ValueError("Invalid calibrated probability")
        results = [
            RuleResult(rule, self.predicates[rule.predicate](frame, self.thresholds))
            for rule in self.rules
        ]
        ordered = sorted(
            results,
            key=lambda x: (-int(x.rule.rule_type == "HARD"), -x.rule.priority, x.rule.rule_id),
        )
        base = frame.base_decision.to_numpy(copy=True)
        final = base.copy()
        rule_ids, rule_names, reasons, highest = [], [], [], []
        for i in range(len(frame)):
            triggered = [result.rule for result in ordered if result.triggered[i]]
            rule_ids.append(tuple(rule.rule_id for rule in triggered))
            rule_names.append(tuple(rule.name for rule in triggered))
            reasons.append(tuple(rule.reason_code for rule in triggered))
            highest.append(triggered[0].rule_id if triggered else None)
            if any(rule.rule_type == "HARD" for rule in triggered):
                final[i] = "BLOCK"
            elif triggered and base[i] == "APPROVE":
                final[i] = "REVIEW"
        return pd.DataFrame(
            {
                "TransactionID": frame.TransactionID.to_numpy(copy=True),
                "calibrated_probability": frame.calibrated_probability.to_numpy(copy=True),
                "base_decision": base,
                "pre_capacity_decision": final,
                "final_decision": final.copy(),
                "triggered_rule_ids": rule_ids,
                "triggered_rule_names": rule_names,
                "rule_reasons": reasons,
                "highest_priority_rule": highest,
                "rule_override": final != base,
                "review_capacity_overflowed": np.zeros(len(frame), dtype=bool),
            },
            index=frame.index.copy(),
        )


def _extreme(frame, threshold):
    return (
        (frame.calibrated_probability >= threshold.very_high_risk)
        & (frame.TransactionAmt >= threshold.high_amount)
    ).to_numpy()


def _disagreement(frame, threshold):
    gap = np.abs(frame.lightgbm_score - frame.xgboost_score)
    return (
        (gap >= threshold.high_disagreement)
        & (frame.calibrated_probability >= threshold.elevated_risk)
        & frame.base_decision.eq("APPROVE")
    ).to_numpy()


def _rare(frame, threshold):
    return (
        (frame.composite_frequency <= threshold.rare_composite_frequency)
        & (frame.calibrated_probability >= threshold.elevated_risk)
        & (frame.TransactionAmt >= threshold.nontrivial_amount)
        & frame.base_decision.eq("APPROVE")
    ).to_numpy()


_PREDICATES = {
    "extreme_risk_amount": _extreme,
    "disagreement_review": _disagreement,
    "rare_identity_review": _rare,
}


def apply_review_capacity(
    results: pd.DataFrame, base_costs: pd.DataFrame, maximum_review_rate: float
) -> pd.DataFrame:
    """Rank all REVIEW rows by frozen economic benefit; never downgrade hard BLOCK."""
    if not np.isfinite(maximum_review_rate) or not 0 <= maximum_review_rate <= 1:
        raise ValueError("maximum_review_rate must be in [0, 1]")
    if not np.array_equal(results.TransactionID, base_costs.TransactionID):
        raise ValueError("Rule results and expected costs must align")
    result = results.copy(deep=True)
    action = result.pre_capacity_decision.to_numpy(copy=True)
    candidates = np.flatnonzero(action == "REVIEW")
    limit = int(np.floor(len(result) * maximum_review_rate))
    if len(candidates) > limit:
        benefit = base_costs.review_priority_benefit.to_numpy(dtype=float)
        ranked = sorted(candidates, key=lambda i: (-benefit[i], result.TransactionID.iloc[i]))
        rejected = np.asarray(ranked[limit:], dtype=int)
        approve = base_costs.expected_cost_approve.to_numpy(dtype=float)
        block = base_costs.expected_cost_block.to_numpy(dtype=float)
        action[rejected] = np.where(approve[rejected] <= block[rejected], "APPROVE", "BLOCK")
        result.iloc[rejected, result.columns.get_loc("review_capacity_overflowed")] = True
    result["final_decision"] = action
    return result
