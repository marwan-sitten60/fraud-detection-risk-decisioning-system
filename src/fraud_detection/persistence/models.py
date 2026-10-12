"""Portable SQLAlchemy schema; PostgreSQL uses JSONB for payload columns."""

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class InferenceRequest(Base):
    __tablename__ = "inference_requests"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_inference_idempotency_key"),
        Index("ix_inference_correlation_id", "correlation_id"),
        Index("ix_inference_bundle_sequence", "bundle_version", "sequence"),
    )

    sequence: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    correlation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    endpoint_type: Mapped[str] = mapped_column(String(16), nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    rules_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    bundle_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    error_category: Mapped[str | None] = mapped_column(String(64))
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="request")


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (Index("ix_transactions_chronology", "transaction_dt", "transaction_id"),)

    transaction_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    transaction_dt: Mapped[int] = mapped_column(BigInteger, nullable=False)
    transaction_amt: Mapped[float] = mapped_column(Float, nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON_TYPE, nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    first_seen_request_sequence: Mapped[int] = mapped_column(
        ForeignKey("inference_requests.sequence"), nullable=False
    )
    row_position: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("request_sequence", "row_position", name="uq_prediction_request_position"),
        Index("ix_prediction_bundle_version", "bundle_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_sequence: Mapped[int] = mapped_column(
        ForeignKey("inference_requests.sequence"), nullable=False
    )
    transaction_id: Mapped[int] = mapped_column(
        ForeignKey("transactions.transaction_id"), unique=True, nullable=False
    )
    row_position: Mapped[int] = mapped_column(Integer, nullable=False)
    lightgbm_score: Mapped[float] = mapped_column(Float, nullable=False)
    xgboost_score: Mapped[float] = mapped_column(Float, nullable=False)
    ensemble_score: Mapped[float] = mapped_column(Float, nullable=False)
    calibrated_probability: Mapped[float] = mapped_column(Float, nullable=False)
    base_decision: Mapped[str] = mapped_column(String(16), nullable=False)
    final_decision: Mapped[str | None] = mapped_column(String(16))
    expected_cost_approve: Mapped[float] = mapped_column(Float, nullable=False)
    expected_cost_review: Mapped[float] = mapped_column(Float, nullable=False)
    expected_cost_block: Mapped[float] = mapped_column(Float, nullable=False)
    triggered_rule_ids: Mapped[list] = mapped_column(JSON_TYPE, nullable=False)
    bundle_version: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_count: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    request: Mapped[InferenceRequest] = relationship(back_populates="predictions")


class FraudLabel(Base):
    __tablename__ = "fraud_labels"
    __table_args__ = (Index("ix_fraud_labels_matured_at", "matured_at"),)

    transaction_id: Mapped[int] = mapped_column(
        ForeignKey("transactions.transaction_id"), primary_key=True
    )
    is_fraud: Mapped[bool] = mapped_column(Boolean, nullable=False)
    label_source: Mapped[str] = mapped_column(String(128), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    matured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    applied_to_model_state_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    label_metadata: Mapped[dict | None] = mapped_column(JSON_TYPE)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_request_event", "request_sequence", "event_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_sequence: Mapped[int | None] = mapped_column(ForeignKey("inference_requests.sequence"))
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    metadata_json: Mapped[dict | None] = mapped_column("metadata_json", JSON_TYPE)
