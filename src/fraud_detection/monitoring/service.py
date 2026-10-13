"""Read-only window metrics, fixed-reference drift, and durable snapshots."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy import select

from fraud_detection.feedback.service import FeedbackService
from fraud_detection.monitoring.alerts import build_alerts
from fraud_detection.monitoring.drift import categorical_js, numeric_psi
from fraud_detection.monitoring.reference import MonitoringReference
from fraud_detection.persistence.database import PersistenceStore
from fraud_detection.persistence.models import (
    InferenceRequest,
    MonitoringAlert,
    MonitoringSnapshot,
    Prediction,
    Transaction,
)

CATEGORICAL_INPUTS = (
    "ProductCD",
    "card4",
    "card6",
    "DeviceType",
    "P_emaildomain",
    "R_emaildomain",
)
ENTITY_INPUTS = ("card1", "card2", "addr1", "DeviceType")
SCORE_INPUTS = (
    "lightgbm_score",
    "xgboost_score",
    "ensemble_score",
    "calibrated_probability",
    "model_disagreement",
)
WINDOWS = {"1h": timedelta(hours=1), "24h": timedelta(hours=24), "7d": timedelta(days=7)}


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Monitoring timestamps must be timezone-aware")
    return value.astimezone(UTC)


def choose_window(name: str, at: datetime, from_time: datetime | None, to_time: datetime | None):
    at = utc(at)
    if from_time is not None or to_time is not None:
        if from_time is None or to_time is None:
            raise ValueError("Both from_time and to_time are required")
        start, end = utc(from_time), utc(to_time)
        if start >= end:
            raise ValueError("from_time must precede to_time")
        return start, end
    if name not in WINDOWS:
        raise ValueError("window must be 1h, 24h, or 7d")
    return at - WINDOWS[name], at


class MonitoringService:
    def __init__(self, store: PersistenceStore, reference_path: Path) -> None:
        self.store = store
        self.reference = MonitoringReference(reference_path)
        self.feedback = FeedbackService(store)

    def _records(self, start: datetime, end: datetime):
        categories = [
            Transaction.raw_payload[name].as_string().label(f"input_{name}")
            for name in CATEGORICAL_INPUTS
        ]
        entities = [
            Transaction.raw_payload[name].as_string().label(f"entity_{name}")
            for name in ENTITY_INPUTS
        ]
        request_query = select(
            InferenceRequest.sequence,
            InferenceRequest.status,
            InferenceRequest.batch_size,
            InferenceRequest.latency_ms,
            InferenceRequest.bundle_version,
            InferenceRequest.rules_enabled,
        ).where(InferenceRequest.received_at >= start, InferenceRequest.received_at < end)
        prediction_query = (
            select(
                Transaction.transaction_id,
                Transaction.transaction_amt,
                *categories,
                *entities,
                Prediction.lightgbm_score,
                Prediction.xgboost_score,
                Prediction.ensemble_score,
                Prediction.calibrated_probability,
                Prediction.base_decision,
                Prediction.final_decision,
                Prediction.triggered_rule_ids,
                Prediction.predicted_at,
                InferenceRequest.bundle_version,
                InferenceRequest.rules_enabled,
            )
            .join(Transaction, Transaction.transaction_id == Prediction.transaction_id)
            .join(InferenceRequest, InferenceRequest.sequence == Prediction.request_sequence)
            .where(
                InferenceRequest.status == "success",
                Prediction.predicted_at >= start,
                Prediction.predicted_at < end,
            )
        )
        with self.store.sessions() as session:
            return session.execute(request_query).all(), session.execute(prediction_query).all()

    def evaluate(
        self,
        start: datetime,
        end: datetime,
        evaluated_at: datetime,
        *,
        database_ready: bool,
        state_recovered: bool,
        persist: bool = False,
    ) -> dict:
        start, end, evaluated_at = utc(start), utc(end), utc(evaluated_at)
        if start >= end:
            raise ValueError("Monitoring window start must precede end")
        requests, records = self._records(start, end)
        request_count = len(requests)
        successes = sum(row.status == "success" for row in requests)
        latencies = [float(row.latency_ms) for row in requests if row.latency_ms is not None]
        latency = {
            name: float(np.percentile(latencies, percentile)) if latencies else None
            for name, percentile in (("p50", 50), ("p95", 95), ("p99", 99))
        }
        total_tx = sum(row.batch_size for row in requests if row.status == "success")
        system = {
            "request_count": request_count,
            "transactions_scored": total_tx,
            "success_count": successes,
            "error_count": request_count - successes,
            "success_rate": successes / request_count if request_count else None,
            "error_rate": (request_count - successes) / request_count if request_count else None,
            "latency_ms": latency,
            "average_batch_size": (
                float(np.mean([row.batch_size for row in requests])) if requests else None
            ),
            "bundle_version": self.reference.bundle_version,
            "database_ready": database_ready,
            "state_recovered": state_recovered,
        }
        arrays = {name: [] for name in (*CATEGORICAL_INPUTS, *ENTITY_INPUTS)}
        arrays.update({name: [] for name in SCORE_INPUTS})
        arrays["TransactionAmt"] = []
        decisions = []
        rule_hits = 0
        entity_novelty = {name: [] for name in ENTITY_INPUTS}
        known_entities = self.reference.data["known_entity_values"]
        known_entity_sets = {name: set(values) for name, values in known_entities.items()}
        for row in records:
            mapping = row._mapping
            arrays["TransactionAmt"].append(mapping["transaction_amt"])
            for name in CATEGORICAL_INPUTS:
                arrays[name].append(mapping[f"input_{name}"])
            for name in ENTITY_INPUTS:
                value = mapping[f"entity_{name}"]
                if name not in CATEGORICAL_INPUTS:
                    arrays[name].append(value)
                entity_novelty[name].append(
                    value is not None and str(value) not in known_entity_sets[name]
                )
            arrays["lightgbm_score"].append(mapping["lightgbm_score"])
            arrays["xgboost_score"].append(mapping["xgboost_score"])
            arrays["ensemble_score"].append(mapping["ensemble_score"])
            arrays["calibrated_probability"].append(mapping["calibrated_probability"])
            arrays["model_disagreement"].append(
                abs(mapping["lightgbm_score"] - mapping["xgboost_score"])
            )
            decisions.append(
                mapping["final_decision"] if mapping["rules_enabled"] else mapping["base_decision"]
            )
            rule_hits += bool(mapping["triggered_rule_ids"])
        reference = self.reference.data
        input_drift = {
            "TransactionAmt": numeric_psi(
                arrays["TransactionAmt"],
                reference["numeric"]["TransactionAmt"]["edges"],
                reference["numeric"]["TransactionAmt"]["frequencies"],
            )
        }
        for name in CATEGORICAL_INPUTS:
            input_drift[name] = categorical_js(arrays[name], reference["categorical"][name])
        missingness = {}
        for name in ("TransactionAmt", *CATEGORICAL_INPUTS, *ENTITY_INPUTS):
            miss_values = ["missing" if value is None else "present" for value in arrays[name]]
            missingness[name] = categorical_js(
                miss_values, reference["categorical"]["missingness"][name]
            )
        for name, result in missingness.items():
            input_drift[f"missingness:{name}"] = result
        scores = {
            name: numeric_psi(
                arrays[name],
                reference["numeric"][name]["edges"],
                reference["numeric"][name]["frequencies"],
            )
            for name in SCORE_INPUTS
        }
        decision_drift = categorical_js(decisions, reference["categorical"]["decision"])
        drift = {
            "status": "insufficient_data" if not records else "stable",
            "inputs": input_drift,
            "scores": scores,
            "decisions": decision_drift,
            "entity_novelty_rate": {
                name: float(np.mean(values)) if values else None
                for name, values in entity_novelty.items()
            },
        }
        feedback = self.feedback.summary(
            evaluated_at,
            bundle_version=self.reference.bundle_version,
            from_time=start,
            to_time=end,
        )
        risk = np.asarray(arrays["calibrated_probability"], dtype=float)
        system["model_output"] = {
            "mean_calibrated_risk": float(risk.mean()) if len(risk) else None,
            "median_calibrated_risk": float(np.median(risk)) if len(risk) else None,
            "high_risk_share_ge_0_5": float(np.mean(risk >= 0.5)) if len(risk) else None,
            "mean_model_disagreement": float(np.mean(arrays["model_disagreement"]))
            if records
            else None,
            "decision_distribution": {
                action: {
                    "count": decisions.count(action),
                    "rate": decisions.count(action) / len(decisions) if decisions else None,
                }
                for action in ("APPROVE", "REVIEW", "BLOCK")
            },
            "rule_trigger_rate": rule_hits / len(records) if records else None,
        }
        alerts = build_alerts({"drift": drift, "system": system, "feedback": feedback}, reference)
        drift["input_status"] = self._status_mixed(input_drift)
        drift["score_status"] = self._status(scores, "psi", "psi")
        drift["decision_status"] = self._status(
            {"decision": decision_drift}, "js_divergence", "js_divergence"
        )
        result = {
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "evaluated_at": evaluated_at.isoformat(),
            "system": system,
            "drift": drift,
            "feedback": feedback,
            "alerts": alerts,
            "alert_counts": {
                "warning": sum(alert["severity"] == "WARNING" for alert in alerts),
                "critical": sum(alert["severity"] == "CRITICAL" for alert in alerts),
            },
        }
        if persist:
            result["snapshot_id"] = self.persist(
                start, end, evaluated_at, total_tx, feedback["mature_labels"], result, alerts
            )
        return result

    @staticmethod
    def _status(results: dict, metric_name: str, fallback: str) -> str:
        drifted = False
        enough = False
        for result in results.values():
            metric = result.get(metric_name)
            if metric is None:
                metric = result.get(fallback)
            if metric is not None:
                enough = True
                drifted |= metric >= (0.20 if metric_name == "psi" else 0.10)
        return "drifted" if drifted else "stable" if enough else "insufficient_data"

    @staticmethod
    def _status_mixed(results: dict) -> str:
        enough = False
        drifted = False
        for result in results.values():
            if result.get("psi") is not None:
                enough = True
                drifted |= result["psi"] >= 0.20
            elif result.get("js_divergence") is not None:
                enough = True
                drifted |= result["js_divergence"] >= 0.10
        return "drifted" if drifted else "stable" if enough else "insufficient_data"

    def persist(self, start, end, evaluated, transaction_count, mature_count, metrics, alerts):
        with self.store.session() as session:
            snapshot = MonitoringSnapshot(
                window_start=start,
                window_end=end,
                evaluated_at=evaluated,
                bundle_version=self.reference.bundle_version,
                transaction_count=transaction_count,
                mature_label_count=mature_count,
                metrics=metrics,
            )
            session.add(snapshot)
            session.flush()
            for alert in alerts:
                session.add(MonitoringAlert(snapshot_id=snapshot.id, **alert))
            session.flush()
            return snapshot.id

    def list_alerts(self, limit: int = 100, severity: str | None = None) -> list[dict]:
        query = select(MonitoringAlert).order_by(MonitoringAlert.created_at.desc()).limit(limit)
        if severity is not None:
            query = query.where(MonitoringAlert.severity == severity.upper())
        with self.store.sessions() as session:
            items = session.scalars(query).all()
            return [
                {
                    "id": item.id,
                    "snapshot_id": item.snapshot_id,
                    "alert_type": item.alert_type,
                    "severity": item.severity,
                    "metric_name": item.metric_name,
                    "observed_value": item.observed_value,
                    "threshold_reference": item.threshold_reference,
                    "message": item.message,
                    "created_at": item.created_at,
                }
                for item in items
            ]
