"""Small PostgreSQL monitoring smoke test using only label-free software fixtures."""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from api.dependencies import ApiSettings
from api.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from fraud_detection.persistence import PersistenceStore
from fraud_detection.persistence.models import (
    AuditEvent,
    FraudLabel,
    InferenceRequest,
    MonitoringAlert,
    MonitoringSnapshot,
    Prediction,
    Transaction,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES = (
    InferenceRequest,
    Transaction,
    Prediction,
    FraudLabel,
    AuditEvent,
    MonitoringSnapshot,
    MonitoringAlert,
)


def counts(store: PersistenceStore) -> dict[str, int]:
    with store.sessions() as session:
        return {
            model.__tablename__: session.scalar(select(func.count()).select_from(model))
            for model in TABLES
        }


def main() -> None:
    url = os.environ.get("FRAUD_DATABASE_URL")
    if not url:
        raise RuntimeError("FRAUD_DATABASE_URL must be present")
    store = PersistenceStore.from_url(url)
    with store.sessions() as session:
        old_requests = set(session.scalars(select(InferenceRequest.sequence)).all())
        old_audits = set(session.scalars(select(AuditEvent.id)).all())
        old_snapshots = set(session.scalars(select(MonitoringSnapshot.id)).all())
        old_labels = set(session.scalars(select(FraudLabel.transaction_id)).all())
    baseline = counts(store)
    transactions = json.loads(
        pd.read_parquet(ROOT / "reports/runtime_golden_input_prefix.parquet")
        .head(4)
        .to_json(orient="records", double_precision=15)
    )
    base = transactions[3]
    batch = [
        {
            **base,
            "TransactionID": int(base["TransactionID"]) + position,
            "TransactionDT": int(base["TransactionDT"]) + position,
            "TransactionAmt": 50_000.0,
        }
        for position in range(30)
    ]
    label_ids = [int(batch[0]["TransactionID"]), int(batch[1]["TransactionID"])]
    print("database_before", baseline, flush=True)
    try:
        with TestClient(create_app(ApiSettings(ROOT / "models/frozen_v1"), store=store)) as client:
            readiness = client.get("/ready")
            assert readiness.status_code == 200, readiness.text
            scored = client.post(
                "/predict/batch",
                json={"transactions": batch},
                headers={"X-Idempotency-Key": "monitoring-live-smoke-20261013"},
            )
            assert scored.status_code == 200, scored.text
            observed = datetime.now(UTC)
            labels = [
                {
                    "TransactionID": label_ids[0],
                    "is_fraud": True,
                    "label_source": "synthetic_test",
                    "observed_at": observed.isoformat(),
                    "matured_at": observed.isoformat(),
                },
                {
                    "TransactionID": label_ids[1],
                    "is_fraud": False,
                    "label_source": "synthetic_test",
                    "observed_at": observed.isoformat(),
                    "matured_at": (observed + timedelta(days=1)).isoformat(),
                },
            ]
            label_result = client.post("/labels/batch", json={"labels": labels})
            assert label_result.status_code == 200, label_result.text
            snapshot = client.post("/monitoring/snapshot", params={"window": "24h"})
            assert snapshot.status_code == 200, snapshot.text
            data = snapshot.json()
            assert data["system"]["transactions_scored"] >= 30
            assert data["feedback"]["mature_labels"] == 1
            assert data["feedback"]["immature_labels"] == 1
            assert data["feedback"]["metrics"]["sample_size"] == 1
            drift = client.get("/monitoring/drift", params={"window": "24h"})
            alerts = client.get("/monitoring/alerts")
            assert drift.status_code == alerts.status_code == 200
            print(
                "live_result",
                json.dumps(
                    {
                        "readiness": readiness.json(),
                        "persisted_predictions": scored.json()["count"],
                        "labels_created": label_result.json()["created_count"],
                        "transaction_count": data["system"]["transactions_scored"],
                        "amount_psi": data["drift"]["inputs"]["TransactionAmt"]["psi"],
                        "input_drift_status": data["drift"]["input_status"],
                        "mature_labels": data["feedback"]["mature_labels"],
                        "immature_labels": data["feedback"]["immature_labels"],
                        "stored_snapshot_id_count": 1,
                        "stored_alerts": alerts.json()["count"],
                        "alerts": data["alerts"],
                    },
                    default=str,
                ),
                flush=True,
            )
    finally:
        with store.session() as session:
            new_requests = (
                set(session.scalars(select(InferenceRequest.sequence)).all()) - old_requests
            )
            new_audits = set(session.scalars(select(AuditEvent.id)).all()) - old_audits
            new_snapshots = (
                set(session.scalars(select(MonitoringSnapshot.id)).all()) - old_snapshots
            )
            new_labels = set(session.scalars(select(FraudLabel.transaction_id)).all()) - old_labels
            session.execute(
                delete(MonitoringAlert).where(MonitoringAlert.snapshot_id.in_(new_snapshots))
            )
            session.execute(
                delete(MonitoringSnapshot).where(MonitoringSnapshot.id.in_(new_snapshots))
            )
            session.execute(delete(FraudLabel).where(FraudLabel.transaction_id.in_(new_labels)))
            session.execute(delete(AuditEvent).where(AuditEvent.id.in_(new_audits)))
            session.execute(delete(Prediction).where(Prediction.request_sequence.in_(new_requests)))
            session.execute(
                delete(Transaction).where(Transaction.first_seen_request_sequence.in_(new_requests))
            )
            session.execute(
                delete(InferenceRequest).where(InferenceRequest.sequence.in_(new_requests))
            )
    cleaned = counts(store)
    assert cleaned == baseline, (baseline, cleaned)
    print("database_after_cleanup", cleaned, flush=True)


if __name__ == "__main__":
    main()
