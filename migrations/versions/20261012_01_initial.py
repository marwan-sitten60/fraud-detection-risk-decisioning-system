"""Initial durable inference and delayed-label schema.

Revision ID: 20261012_01
Revises:
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "20261012_01"
down_revision = None
branch_labels = None
depends_on = None
JSON_TYPE = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "inference_requests",
        sa.Column("sequence", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("correlation_id", sa.String(64), nullable=False),
        sa.Column("idempotency_key", sa.String(128)),
        sa.Column("endpoint_type", sa.String(16), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("rules_enabled", sa.Boolean(), nullable=False),
        sa.Column("bundle_version", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("latency_ms", sa.Float()),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column("error_category", sa.String(64)),
        sa.UniqueConstraint("idempotency_key", name="uq_inference_idempotency_key"),
    )
    op.create_index("ix_inference_correlation_id", "inference_requests", ["correlation_id"])
    op.create_index(
        "ix_inference_bundle_sequence", "inference_requests", ["bundle_version", "sequence"]
    )
    op.create_table(
        "transactions",
        sa.Column("transaction_id", sa.BigInteger(), primary_key=True),
        sa.Column("transaction_dt", sa.BigInteger(), nullable=False),
        sa.Column("transaction_amt", sa.Float(), nullable=False),
        sa.Column("raw_payload", JSON_TYPE, nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column(
            "first_seen_request_sequence",
            sa.Integer(),
            sa.ForeignKey("inference_requests.sequence"),
            nullable=False,
        ),
        sa.Column("row_position", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_transactions_chronology", "transactions", ["transaction_dt", "transaction_id"]
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "request_sequence",
            sa.Integer(),
            sa.ForeignKey("inference_requests.sequence"),
            nullable=False,
        ),
        sa.Column(
            "transaction_id",
            sa.BigInteger(),
            sa.ForeignKey("transactions.transaction_id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("row_position", sa.Integer(), nullable=False),
        sa.Column("lightgbm_score", sa.Float(), nullable=False),
        sa.Column("xgboost_score", sa.Float(), nullable=False),
        sa.Column("ensemble_score", sa.Float(), nullable=False),
        sa.Column("calibrated_probability", sa.Float(), nullable=False),
        sa.Column("base_decision", sa.String(16), nullable=False),
        sa.Column("final_decision", sa.String(16)),
        sa.Column("expected_cost_approve", sa.Float(), nullable=False),
        sa.Column("expected_cost_review", sa.Float(), nullable=False),
        sa.Column("expected_cost_block", sa.Float(), nullable=False),
        sa.Column("triggered_rule_ids", JSON_TYPE, nullable=False),
        sa.Column("bundle_version", sa.String(64), nullable=False),
        sa.Column("feature_count", sa.Integer(), nullable=False),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "request_sequence", "row_position", name="uq_prediction_request_position"
        ),
    )
    op.create_index("ix_prediction_bundle_version", "predictions", ["bundle_version"])
    op.create_table(
        "fraud_labels",
        sa.Column(
            "transaction_id",
            sa.BigInteger(),
            sa.ForeignKey("transactions.transaction_id"),
            primary_key=True,
        ),
        sa.Column("is_fraud", sa.Boolean(), nullable=False),
        sa.Column("label_source", sa.String(128), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("matured_at", sa.DateTime(timezone=True)),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("applied_to_model_state_at", sa.DateTime(timezone=True)),
        sa.Column("label_metadata", JSON_TYPE),
    )
    op.create_index("ix_fraud_labels_matured_at", "fraud_labels", ["matured_at"])
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("request_sequence", sa.Integer(), sa.ForeignKey("inference_requests.sequence")),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata_json", JSON_TYPE),
    )
    op.create_index("ix_audit_request_event", "audit_events", ["request_sequence", "event_type"])


def downgrade() -> None:
    raise RuntimeError("Destructive downgrade is intentionally unsupported")
