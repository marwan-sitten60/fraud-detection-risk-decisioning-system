"""Transactional idempotency gate around the frozen stateful runtime."""

import json
import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from fraud_detection.inference import FrozenFraudInferenceRuntime
from fraud_detection.inference.contracts import validate_transactions
from fraud_detection.persistence.canonical import canonical_json, payload_hash, scoring_payload
from fraud_detection.persistence.database import PersistenceStore
from fraud_detection.persistence.models import (
    AuditEvent,
    InferenceRequest,
    Prediction,
    Transaction,
    utc_now,
)


class PredictionConflict(Exception):
    """A key or TransactionID already belongs to another scored operation."""


class PersistenceUnavailable(Exception):
    """Persistence failed; mutable runtime history has been rolled back."""


@dataclass(frozen=True)
class ScoreResult:
    frame: pd.DataFrame
    replayed: bool
    validation_ms: float
    db_lookup_ms: float
    inference_ms: float
    db_write_ms: float
    total_ms: float


def prediction_frame(predictions: list[Prediction], *, rules_enabled: bool) -> pd.DataFrame:
    rows = []
    for item in predictions:
        row = {
            "TransactionID": item.transaction_id,
            "lightgbm_score": item.lightgbm_score,
            "xgboost_score": item.xgboost_score,
            "ensemble_score": item.ensemble_score,
            "calibrated_probability": item.calibrated_probability,
            "base_decision": item.base_decision,
            "expected_cost_approve": item.expected_cost_approve,
            "expected_cost_review": item.expected_cost_review,
            "expected_cost_block": item.expected_cost_block,
            "bundle_version": item.bundle_version,
        }
        if rules_enabled:
            row["final_decision"] = item.final_decision
            row["triggered_rule_ids"] = "|".join(item.triggered_rule_ids)
        rows.append(row)
    return pd.DataFrame(rows)


class PersistenceService:
    """Call only under one application-level inference lock."""

    def __init__(
        self,
        runtime: FrozenFraudInferenceRuntime,
        store: PersistenceStore,
        last_key: tuple[float, int] | None,
    ) -> None:
        self.runtime = runtime
        self.store = store
        self.last_key = last_key

    def _record_failure(
        self,
        endpoint_type: str,
        row_count: int,
        rules_enabled: bool,
        correlation_id: str,
        digest: str,
        category: str,
        started: float,
        received_at: datetime,
    ) -> None:
        """Audit a rejected valid payload without claiming a successful score."""
        with self.store.session() as session:
            request = InferenceRequest(
                id=str(uuid.uuid4()),
                correlation_id=correlation_id,
                idempotency_key=None,
                endpoint_type=endpoint_type,
                batch_size=row_count,
                rules_enabled=rules_enabled,
                bundle_version=self.runtime.manifest["bundle_version"],
                status="failed",
                received_at=received_at,
                completed_at=utc_now(),
                latency_ms=(time.perf_counter() - started) * 1000,
                payload_hash=digest,
                error_category=category,
            )
            session.add(request)
            session.flush()
            session.add(
                AuditEvent(request_sequence=request.sequence, event_type="prediction_failed")
            )

    def score(
        self,
        endpoint_type: str,
        rows: list[dict],
        *,
        rules_enabled: bool,
        idempotency_key: str | None,
        correlation_id: str,
    ) -> ScoreResult:
        started = time.perf_counter()
        received_at = utc_now()
        snapshot = None
        digest = None
        try:
            numeric_columns = tuple(
                name
                for name in self.runtime.feature_state.raw_columns
                if name not in self.runtime.feature_state.preprocessor.categorical_features_
            )
            validated = validate_transactions(
                rows, self.runtime.feature_state.raw_columns, numeric_columns
            )
            normalized = [
                json.loads(canonical_json(row)) for row in validated.to_dict(orient="records")
            ]
            digest = payload_hash(scoring_payload(endpoint_type, normalized, rules_enabled))
            ids = [int(row["TransactionID"]) for row in normalized]
            first_key = (float(normalized[0]["TransactionDT"]), ids[0])
            lookup_started = time.perf_counter()
            with self.store.session() as session:
                if idempotency_key is not None:
                    previous = session.scalar(
                        select(InferenceRequest).where(
                            InferenceRequest.idempotency_key == idempotency_key
                        )
                    )
                    if previous is not None:
                        if previous.payload_hash != digest or previous.status != "success":
                            raise PredictionConflict("Idempotency key is bound to another payload")
                        predictions = session.scalars(
                            select(Prediction)
                            .where(Prediction.request_sequence == previous.sequence)
                            .order_by(Prediction.row_position)
                        ).all()
                        if len(predictions) != previous.batch_size:
                            raise PersistenceUnavailable("Stored idempotent response is incomplete")
                        frame = prediction_frame(predictions, rules_enabled=previous.rules_enabled)
                        session.add(
                            AuditEvent(
                                request_sequence=previous.sequence,
                                event_type="idempotent_replay",
                                metadata_json={"correlation_id": correlation_id},
                            )
                        )
                        elapsed = (time.perf_counter() - started) * 1000
                        return ScoreResult(
                            frame,
                            True,
                            (lookup_started - started) * 1000,
                            (time.perf_counter() - lookup_started) * 1000,
                            0.0,
                            0.0,
                            elapsed,
                        )
                existing = session.scalars(
                    select(Transaction.transaction_id).where(Transaction.transaction_id.in_(ids))
                ).all()
                if existing:
                    raise PredictionConflict("TransactionID has already been scored")
                if self.last_key is not None and first_key <= self.last_key:
                    raise ValueError("Stateful transactions must follow prior TransactionDT and ID")
                lookup_ms = (time.perf_counter() - lookup_started) * 1000
                snapshot = self.runtime.snapshot_state()
                inference_started = time.perf_counter()
                frame = self.runtime.predict_batch(normalized, include_rules=rules_enabled)
                inference_ms = (time.perf_counter() - inference_started) * 1000
                write_started = time.perf_counter()
                request = InferenceRequest(
                    id=str(uuid.uuid4()),
                    correlation_id=correlation_id,
                    idempotency_key=idempotency_key,
                    endpoint_type=endpoint_type,
                    batch_size=len(normalized),
                    rules_enabled=rules_enabled,
                    bundle_version=self.runtime.manifest["bundle_version"],
                    status="success",
                    received_at=received_at,
                    completed_at=utc_now(),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    payload_hash=digest,
                )
                session.add(request)
                session.flush()
                session.add(
                    AuditEvent(request_sequence=request.sequence, event_type="request_received")
                )
                for position, (raw, (_, prediction)) in enumerate(
                    zip(normalized, frame.iterrows(), strict=True)
                ):
                    costs = (
                        prediction.expected_cost_approve,
                        prediction.expected_cost_review,
                        prediction.expected_cost_block,
                    )
                    scores = (
                        prediction.lightgbm_score,
                        prediction.xgboost_score,
                        prediction.ensemble_score,
                        prediction.calibrated_probability,
                    )
                    if not all(math.isfinite(float(value)) for value in (*scores, *costs)):
                        raise RuntimeError("Inference returned a non-finite score or cost")
                    session.add(
                        Transaction(
                            transaction_id=int(raw["TransactionID"]),
                            transaction_dt=int(raw["TransactionDT"]),
                            transaction_amt=float(raw["TransactionAmt"]),
                            raw_payload=raw,
                            payload_hash=payload_hash(raw),
                            first_seen_request_sequence=request.sequence,
                            row_position=position,
                        )
                    )
                    rule_ids = prediction.get("triggered_rule_ids", "")
                    session.add(
                        Prediction(
                            request_sequence=request.sequence,
                            transaction_id=int(raw["TransactionID"]),
                            row_position=position,
                            lightgbm_score=float(prediction.lightgbm_score),
                            xgboost_score=float(prediction.xgboost_score),
                            ensemble_score=float(prediction.ensemble_score),
                            calibrated_probability=float(prediction.calibrated_probability),
                            base_decision=str(prediction.base_decision),
                            final_decision=(
                                str(prediction.final_decision) if rules_enabled else None
                            ),
                            expected_cost_approve=float(costs[0]),
                            expected_cost_review=float(costs[1]),
                            expected_cost_block=float(costs[2]),
                            triggered_rule_ids=rule_ids.split("|") if rule_ids else [],
                            bundle_version=request.bundle_version,
                            feature_count=int(self.runtime.manifest["feature_count"]),
                        )
                    )
                session.add(
                    AuditEvent(
                        request_sequence=request.sequence,
                        event_type="prediction_completed",
                        metadata_json={"transaction_count": len(normalized)},
                    )
                )
            self.last_key = (float(normalized[-1]["TransactionDT"]), ids[-1])
            total_ms = (time.perf_counter() - started) * 1000
            return ScoreResult(
                frame,
                False,
                (lookup_started - started) * 1000,
                lookup_ms,
                inference_ms,
                (time.perf_counter() - write_started) * 1000,
                total_ms,
            )
        except (ValueError, PredictionConflict) as error:
            if snapshot is not None:
                self.runtime.restore_state(snapshot)
            if digest is not None:
                try:
                    self._record_failure(
                        endpoint_type,
                        len(rows),
                        rules_enabled,
                        correlation_id,
                        digest,
                        "conflict" if isinstance(error, PredictionConflict) else "invalid_input",
                        started,
                        received_at,
                    )
                except SQLAlchemyError as database_error:
                    raise PersistenceUnavailable(
                        "Failed-request audit could not be stored"
                    ) from database_error
            raise
        except SQLAlchemyError as error:
            if snapshot is not None:
                self.runtime.restore_state(snapshot)
            raise PersistenceUnavailable("Database operation failed") from error
        except Exception:
            if snapshot is not None:
                self.runtime.restore_state(snapshot)
            if digest is not None:
                try:
                    self._record_failure(
                        endpoint_type,
                        len(rows),
                        rules_enabled,
                        correlation_id,
                        digest,
                        "inference_error",
                        started,
                        received_at,
                    )
                except SQLAlchemyError as database_error:
                    raise PersistenceUnavailable(
                        "Failed-request audit could not be stored"
                    ) from database_error
            raise

    def lookup_prediction(self, transaction_id: int) -> dict | None:
        found = self.store.prediction_by_transaction(transaction_id)
        if found is None:
            return None
        prediction, request = found
        payload = (
            prediction_frame([prediction], rules_enabled=request.rules_enabled).iloc[0].to_dict()
        )
        payload["triggered_rule_ids"] = prediction.triggered_rule_ids
        payload["request_id"] = request.id
        payload["scored_at"] = prediction.predicted_at
        payload["rules_enabled"] = request.rules_enabled
        return payload
