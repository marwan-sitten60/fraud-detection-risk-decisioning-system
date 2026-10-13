"""Build compact monitor baselines from TRAIN inputs and TRAIN OOF scores only."""

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud_detection.decisioning.costs import scenario_profiles
from fraud_detection.decisioning.policy import decide
from fraud_detection.feedback.metrics import calculate
from fraud_detection.inference.bundle import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models/monitoring_v1"
TRAIN_ROWS = int(590540 * 0.70)
NUMERIC_INPUTS = ("TransactionAmt",)
CATEGORICAL_INPUTS = (
    "ProductCD",
    "card4",
    "card6",
    "DeviceType",
    "P_emaildomain",
    "R_emaildomain",
)
ENTITY_INPUTS = ("card1", "card2", "addr1", "DeviceType")
MISSINGNESS_INPUTS = NUMERIC_INPUTS + CATEGORICAL_INPUTS + ENTITY_INPUTS
OOF_SCORE_COLUMNS = (
    "lightgbm_score",
    "xgboost_score",
    "ensemble_score",
    "calibrated_probability",
    "model_disagreement",
)


def edges(values: pd.Series, bins: int = 10) -> list[float]:
    finite = values.dropna().astype(float)
    internal = finite.quantile(np.linspace(0, 1, bins + 1)[1:-1]).to_numpy()
    internal = np.unique(internal[np.isfinite(internal)])
    return [-float("inf"), *[float(x) for x in internal], float("inf")]


def numeric_distribution(values: pd.Series, bin_edges: list[float]) -> list[float]:
    series = pd.to_numeric(values, errors="coerce")
    bucket = pd.cut(series, bin_edges, right=True, include_lowest=True, labels=False)
    counts = np.bincount(bucket.dropna().astype(int), minlength=len(bin_edges) - 1).astype(float)
    missing = float(series.isna().sum())
    counts = np.append(counts, missing)
    return (counts / counts.sum()).tolist()


def categorical_distribution(values: pd.Series, top: int = 64) -> dict:
    normalized = values.astype("string").fillna("__MISSING__")
    counts = normalized.value_counts(dropna=False)
    known = counts.head(top).index.astype(str).tolist()
    mapped = normalized.where(normalized.isin(known), "__OTHER__")
    dist = mapped.value_counts(normalize=True).to_dict()
    return {
        "known_categories": known,
        "all_known_categories": counts.index.astype(str).tolist(),
        "frequencies": {str(k): float(v) for k, v in dist.items()},
    }


def main() -> None:
    transaction_columns = [
        "TransactionID",
        *NUMERIC_INPUTS,
        *[name for name in CATEGORICAL_INPUTS if name != "DeviceType"],
        *[name for name in ENTITY_INPUTS if name != "DeviceType"],
    ]
    transaction_columns = list(dict.fromkeys(transaction_columns))
    transactions = pd.read_csv(
        ROOT / "data/raw/train_transaction.csv",
        nrows=TRAIN_ROWS,
        usecols=transaction_columns,
        low_memory=False,
    )
    train_max_id = int(transactions.TransactionID.max())
    identity_ids = pd.read_csv(ROOT / "data/raw/train_identity.csv", usecols=["TransactionID"])
    identity_count = int(identity_ids.TransactionID.le(train_max_id).sum())
    identity = pd.read_csv(
        ROOT / "data/raw/train_identity.csv",
        nrows=identity_count,
        usecols=["TransactionID", "DeviceType"],
    )
    inputs = transactions.merge(
        identity, on="TransactionID", how="left", suffixes=("", "_identity")
    )
    if "DeviceType_identity" in inputs:
        inputs["DeviceType"] = inputs["DeviceType"].fillna(inputs.pop("DeviceType_identity"))
    del transactions, identity, identity_ids

    oof = pd.read_csv(
        ROOT / "reports/ensemble_oof_predictions.csv",
        usecols=["TransactionID", "lightgbm_score", "xgboost_score", "y_true"],
    )
    oof["ensemble_score"] = 0.8 * oof.lightgbm_score + 0.2 * oof.xgboost_score
    calibrator = joblib.load(ROOT / "models/frozen_v1/platt_calibrator.joblib")
    oof["calibrated_probability"] = calibrator.transform(oof.ensemble_score.to_numpy())
    oof["model_disagreement"] = (oof.lightgbm_score - oof.xgboost_score).abs()
    decision_input = oof.merge(
        inputs[["TransactionID", "TransactionAmt"]], on="TransactionID", how="inner"
    )
    decisions = decide(
        decision_input[["TransactionID", "calibrated_probability", "TransactionAmt"]],
        scenario_profiles()["Balanced"],
    )
    oof["decision"] = decisions.decision.to_numpy()
    perf_rows = [
        {
            "is_fraud": bool(label),
            "probability": float(probability),
            "decision": str(decision),
            "amount": float(amount),
            "lightgbm_score": float(lightgbm),
            "xgboost_score": float(xgboost),
            "ensemble_score": float(ensemble),
            "expected_cost_approve": float(cost_a),
            "expected_cost_review": float(cost_r),
            "expected_cost_block": float(cost_b),
            "triggered_rule_ids": [],
        }
        for (
            label,
            probability,
            decision,
            amount,
            lightgbm,
            xgboost,
            ensemble,
            cost_a,
            cost_r,
            cost_b,
        ) in zip(
            decision_input.y_true,
            decision_input.calibrated_probability,
            decisions.decision,
            decision_input.TransactionAmt,
            decision_input.lightgbm_score,
            decision_input.xgboost_score,
            decision_input.ensemble_score,
            decisions.expected_cost_approve,
            decisions.expected_cost_review,
            decisions.expected_cost_block,
            strict=True,
        )
    ]

    numeric = {}
    for name in (*NUMERIC_INPUTS, *OOF_SCORE_COLUMNS):
        source = inputs[name] if name in inputs else oof[name]
        bin_edges = edges(source)
        numeric[name] = {
            "edges": [
                "-inf" if np.isneginf(x) else "inf" if np.isposinf(x) else x for x in bin_edges
            ],
            "frequencies": numeric_distribution(source, bin_edges),
            "sample_count": int(len(source)),
        }
    categorical = {name: categorical_distribution(inputs[name]) for name in CATEGORICAL_INPUTS}
    categorical["decision"] = categorical_distribution(oof.decision, top=8)
    categorical["missingness"] = {
        name: categorical_distribution(
            inputs[name].isna().map({True: "missing", False: "present"}), top=2
        )
        for name in MISSINGNESS_INPUTS
    }
    known_entities = {
        name: sorted(inputs[name].dropna().astype(str).unique().tolist()) for name in ENTITY_INPUTS
    }
    risk_summary = {
        name: {"mean": float(oof[name].mean()), "median": float(oof[name].median())}
        for name in OOF_SCORE_COLUMNS
    }
    decisions_ref = {
        str(key): float(value) for key, value in oof.decision.value_counts(normalize=True).items()
    }
    reference = {
        "reference_version": "monitoring_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "bundle_version": "frozen_v1",
        "reference_sources": {
            "inputs": (
                "IEEE-CIS TRAIN first 413378 rows; train_identity restricted to "
                "TransactionID <= TRAIN max"
            ),
            "scores": (
                "reports/ensemble_oof_predictions.csv component scores, calibrated with "
                "frozen_v1 Platt artifact"
            ),
            "decision": (
                "TRAIN OOF scores and amount passed through unchanged Balanced decision policy"
            ),
            "holdout_used": False,
        },
        "sample_counts": {
            "train_inputs": int(len(inputs)),
            "train_oof_scores": int(len(oof)),
            "decision_rows": int(len(decision_input)),
        },
        "numeric": numeric,
        "categorical": categorical,
        "known_entity_values": known_entities,
        "score_summary": risk_summary,
        "decision_frequencies": decisions_ref,
        "performance_reference": {
            "source": "TRAIN-only out-of-fold labels and scores; no HOLDOUT data",
            **calculate(perf_rows),
        },
        "method_notes": {
            "numeric_bins": "TRAIN quantile edges, fixed and reused by production windows",
            "numeric_missing_bucket": "final frequency bucket after numeric bins",
            "categorical_other": (
                "non-top-64 TRAIN values aggregated as __OTHER__; "
                "unseen production values remain explicit"
            ),
            "entity_novelty": (
                "Exact membership in TRAIN unique values for card1, card2, addr1, and DeviceType"
            ),
            "decision_caveat": (
                "OOF reference decisions are a TRAIN development comparison, "
                "not a live serving guarantee"
            ),
        },
    }
    DEST.mkdir(parents=True, exist_ok=True)
    artifact = DEST / "reference.json"
    artifact.write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    checksum = sha256_file(artifact)
    manifest = {
        "artifact": artifact.name,
        "sha256": checksum,
        "bundle_version": "frozen_v1",
        "reference_version": "monitoring_v1",
        "source": reference["reference_sources"],
        "created_at": reference["created_at"],
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "reference": str(artifact),
                "sha256": checksum,
                "sample_counts": reference["sample_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
