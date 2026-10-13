"""Immutable label ingestion and feedback over stored prediction columns."""

from datetime import UTC, datetime

from sqlalchemy import select

from fraud_detection.feedback.metrics import calculate, delay_summary, maturity_status
from fraud_detection.feedback.schemas import LabelInput
from fraud_detection.persistence.database import PersistenceStore
from fraud_detection.persistence.models import (
    AuditEvent,
    FraudLabel,
    InferenceRequest,
    Prediction,
    Transaction,
    utc_now,
)


class UnknownPrediction(Exception):
    """The transaction does not have a stored successful prediction."""


class LabelConflict(Exception):
    """An immutable label already exists with different identity or truth."""


def db_utc(value: datetime) -> datetime:
    """SQLite loses timezone information; persisted application timestamps are UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class FeedbackService:
    def __init__(self, store: PersistenceStore) -> None:
        self.store = store

    def ingest(
        self, inputs: list[LabelInput], evaluation_time: datetime | None = None
    ) -> list[dict]:
        """Commit the complete batch, or roll it all back on any missing/conflicting row."""
        at = evaluation_time or utc_now()
        results = []
        with self.store.session() as session:
            for item in inputs:
                transaction = session.scalar(
                    select(Transaction.transaction_id)
                    .where(Transaction.transaction_id == item.transaction_id)
                    .with_for_update()
                )
                if transaction is None:
                    raise UnknownPrediction("TransactionID has no stored prediction")
                prediction = session.scalar(
                    select(Prediction).where(Prediction.transaction_id == item.transaction_id)
                )
                if prediction is None:
                    raise UnknownPrediction("TransactionID has no stored prediction")
                request = session.get(InferenceRequest, prediction.request_sequence)
                if request is None or request.status != "success":
                    raise UnknownPrediction("TransactionID has no successful prediction")
                if item.observed_at < db_utc(prediction.predicted_at):
                    raise ValueError("observed_at cannot precede prediction time")
                existing = session.get(FraudLabel, item.transaction_id)
                if existing is not None:
                    same = (
                        existing.is_fraud == item.is_fraud
                        and existing.label_source == item.label_source
                        and db_utc(existing.observed_at) == item.observed_at
                        and existing.matured_at is not None
                        and db_utc(existing.matured_at) == item.matured_at
                        and existing.label_metadata == item.metadata
                    )
                    if not same:
                        raise LabelConflict("TransactionID already has a different label")
                    label = existing
                else:
                    label = FraudLabel(
                        transaction_id=item.transaction_id,
                        is_fraud=item.is_fraud,
                        label_source=item.label_source,
                        observed_at=item.observed_at,
                        matured_at=item.matured_at,
                        ingested_at=utc_now(),
                        applied_to_model_state_at=None,
                        label_metadata=item.metadata,
                    )
                    session.add(label)
                    session.flush()
                    session.add(
                        AuditEvent(
                            request_sequence=prediction.request_sequence,
                            event_type="label_received",
                            metadata_json={
                                "transaction_id": item.transaction_id,
                                "label_source": item.label_source,
                                "observed_at": item.observed_at.isoformat(),
                                "matured_at": item.matured_at.isoformat(),
                                "maturity_status": maturity_status(item.matured_at, at),
                            },
                        )
                    )
                results.append(
                    {
                        "TransactionID": label.transaction_id,
                        "is_fraud": label.is_fraud,
                        "label_source": label.label_source,
                        "observed_at": db_utc(label.observed_at),
                        "matured_at": db_utc(label.matured_at),
                        "ingested_at": db_utc(label.ingested_at),
                        "maturity_status": maturity_status(db_utc(label.matured_at), at),
                        "replayed": existing is not None,
                    }
                )
        return results

    def lookup(self, transaction_id: int, evaluation_time: datetime | None = None) -> dict | None:
        with self.store.sessions() as session:
            label = session.get(FraudLabel, transaction_id)
            if label is None:
                return None
            return {
                "TransactionID": label.transaction_id,
                "is_fraud": label.is_fraud,
                "label_source": label.label_source,
                "observed_at": db_utc(label.observed_at),
                "matured_at": db_utc(label.matured_at) if label.matured_at else None,
                "ingested_at": db_utc(label.ingested_at),
                "maturity_status": maturity_status(
                    db_utc(label.matured_at) if label.matured_at else None,
                    evaluation_time or utc_now(),
                ),
                "replayed": False,
            }

    def summary(
        self,
        evaluation_time: datetime,
        bundle_version: str | None = None,
        rules_enabled: bool | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> dict:
        """Select scalar prediction fields; never fetch transaction.raw_payload."""
        query = (
            select(
                Prediction.calibrated_probability,
                Prediction.lightgbm_score,
                Prediction.xgboost_score,
                Prediction.ensemble_score,
                Prediction.base_decision,
                Prediction.final_decision,
                Prediction.expected_cost_approve,
                Prediction.expected_cost_review,
                Prediction.expected_cost_block,
                Prediction.triggered_rule_ids,
                Prediction.predicted_at,
                Transaction.transaction_amt,
                FraudLabel.is_fraud,
                FraudLabel.label_source,
                FraudLabel.observed_at,
                FraudLabel.matured_at,
                InferenceRequest.rules_enabled,
                InferenceRequest.bundle_version,
            )
            .join(Transaction, Transaction.transaction_id == Prediction.transaction_id)
            .join(InferenceRequest, InferenceRequest.sequence == Prediction.request_sequence)
            .outerjoin(FraudLabel, FraudLabel.transaction_id == Prediction.transaction_id)
            .where(InferenceRequest.status == "success")
        )
        if bundle_version is not None:
            query = query.where(InferenceRequest.bundle_version == bundle_version)
        if rules_enabled is not None:
            query = query.where(InferenceRequest.rules_enabled == rules_enabled)
        if from_time is not None:
            query = query.where(Prediction.predicted_at >= from_time)
        if to_time is not None:
            query = query.where(Prediction.predicted_at < to_time)
        with self.store.sessions() as session:
            records = session.execute(query).all()
        mature_rows = []
        labeled = immature = 0
        prediction_observed = []
        observed_maturity = []
        sources = {}
        for record in records:
            if record.observed_at is None:
                continue
            labeled += 1
            sources[record.label_source] = sources.get(record.label_source, 0) + 1
            observed = db_utc(record.observed_at)
            prediction_observed.append((observed - db_utc(record.predicted_at)).total_seconds())
            if record.matured_at is None:
                immature += 1
                continue
            matured = db_utc(record.matured_at)
            observed_maturity.append((matured - observed).total_seconds())
            if maturity_status(matured, evaluation_time) == "IMMATURE":
                immature += 1
                continue
            mature_rows.append(
                {
                    "is_fraud": record.is_fraud,
                    "probability": record.calibrated_probability,
                    "lightgbm_score": record.lightgbm_score,
                    "xgboost_score": record.xgboost_score,
                    "ensemble_score": record.ensemble_score,
                    "decision": record.final_decision
                    if record.rules_enabled
                    else record.base_decision,
                    "amount": record.transaction_amt,
                    "expected_cost_approve": record.expected_cost_approve,
                    "expected_cost_review": record.expected_cost_review,
                    "expected_cost_block": record.expected_cost_block,
                    "triggered_rule_ids": record.triggered_rule_ids,
                    "policy": (
                        f"{record.bundle_version}:rules_"
                        f"{'enabled' if record.rules_enabled else 'disabled'}"
                    ),
                }
            )
        policies = sorted({row["policy"] for row in mature_rows})
        metrics = calculate(mature_rows)
        sample_size = metrics["sample_size"]
        fraud_count = round((metrics["fraud_prevalence"] or 0.0) * sample_size)
        legitimate_count = sample_size - fraud_count
        performance_supported = sample_size >= 200 and fraud_count >= 20 and legitimate_count >= 20
        return {
            "evaluation_time": evaluation_time.isoformat(),
            "coverage_denominator": (
                "all stored successful predictions in the selected bundle/rules cohort"
            ),
            "total_predictions": len(records),
            "labels_received": labeled,
            "mature_labels": len(mature_rows),
            "immature_labels": immature,
            "mature_label_coverage": len(mature_rows) / len(records) if records else None,
            "label_sources": sources,
            "status": "available" if performance_supported else "insufficient_labels",
            "metrics": metrics,
            "by_policy": {
                policy: calculate([row for row in mature_rows if row["policy"] == policy])
                for policy in policies
            },
            "label_delays": {
                "transaction_to_observed": None,
                "transaction_to_observed_reason": (
                    "TransactionDT is relative; no calendar epoch is stored"
                ),
                "prediction_to_observed": delay_summary(prediction_observed),
                "observed_to_maturity": delay_summary(observed_maturity),
            },
        }
