"""Durable monitoring snapshots and deterministic alerts.

Revision ID: 20261013_01
Revises: 20261012_01
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "20261013_01"
down_revision = "20261012_01"
branch_labels = None
depends_on = None
JSON_TYPE = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "monitoring_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bundle_version", sa.String(64), nullable=False),
        sa.Column("transaction_count", sa.Integer(), nullable=False),
        sa.Column("mature_label_count", sa.Integer(), nullable=False),
        sa.Column("metrics", JSON_TYPE, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_monitoring_window", "monitoring_snapshots", ["window_start", "window_end"])
    op.create_index("ix_monitoring_evaluated_at", "monitoring_snapshots", ["evaluated_at"])
    op.create_table(
        "monitoring_alerts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "snapshot_id",
            sa.Integer(),
            sa.ForeignKey("monitoring_snapshots.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("alert_type", sa.String(32), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("metric_name", sa.String(128), nullable=False),
        sa.Column("observed_value", sa.Float()),
        sa.Column("threshold_reference", sa.String(256), nullable=False),
        sa.Column("message", sa.String(512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_monitoring_alert_severity_created",
        "monitoring_alerts",
        ["severity", "created_at"],
    )


def downgrade() -> None:
    raise RuntimeError("Destructive downgrade is intentionally unsupported")
