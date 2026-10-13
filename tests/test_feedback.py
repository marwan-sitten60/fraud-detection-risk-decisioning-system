"""Delayed-label lifecycle on small synthetic stored predictions."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from fraud_detection.feedback.metrics import calculate, delay_summary, maturity_status
from fraud_detection.persistence.models import AuditEvent, FraudLabel, Prediction
from fraud_detection.persistence.recovery import recover_runtime
from test_api import StubRuntime, app_for, transaction


def label(identifier: int, fraud: bool, observed: datetime, mature: datetime) -> dict:
    return {
        "TransactionID": identifier,
        "is_fraud": fraud,
        "label_source": "synthetic_test",
        "observed_at": observed.isoformat(),
        "matured_at": mature.isoformat(),
        "metadata": {"case": "test"},
    }


def test_label_lifecycle_batch_atomicity_and_stored_feedback() -> None:
    runtime = StubRuntime()
    app, _ = app_for(runtime)
    with TestClient(app) as client:
        store = client.app.state.store
        empty = client.get("/feedback/summary").json()
        assert empty["total_predictions"] == 0
        assert empty["metrics"]["pr_auc"] is None
        for identifier in (1, 2, 3):
            assert (
                client.post(
                    "/predict", json={"transaction": transaction(identifier, identifier * 10)}
                ).status_code
                == 200
            )
        snapshot = runtime.snapshot_state()
        observed = datetime.now(UTC)
        mature = observed + timedelta(days=1)
        inputs = [
            label(1, True, observed, observed),
            label(2, False, observed, observed),
            label(3, True, observed, mature),
        ]
        missing = client.post("/labels", json=label(999, True, observed, observed))
        assert missing.status_code == 404
        atomic = client.post(
            "/labels/batch", json={"labels": [inputs[0], label(999, False, observed, mature)]}
        )
        assert atomic.status_code == 404
        with store.sessions() as session:
            assert session.scalar(select(func.count()).select_from(FraudLabel)) == 0
        batch = client.post("/labels/batch", json={"labels": inputs})
        assert batch.status_code == 200, batch.json()
        assert batch.json()["created_count"] == 3
        assert runtime.snapshot_state().equals(snapshot)
        replay = client.post("/labels", json=inputs[0])
        assert replay.status_code == 200 and replay.json()["replayed"]
        conflict = client.post("/labels", json={**inputs[0], "is_fraud": False})
        assert conflict.status_code == 409
        invalid = client.post("/labels", json={**inputs[0], "observed_at": "2020-01-01"})
        assert invalid.status_code == 422
        result = client.get("/feedback/summary").json()
        assert (result["total_predictions"], result["labels_received"]) == (3, 3)
        assert (result["mature_labels"], result["immature_labels"]) == (2, 1)
        assert result["status"] == "insufficient_labels"
        assert result["mature_label_coverage"] == pytest.approx(2 / 3)
        assert result["metrics"]["pr_auc"] is not None
        assert result["metrics"]["stored_score_means"]["lightgbm_score"] > 0
        assert result["metrics"]["stored_expected_cost_means"]["expected_cost_review"] == 3
        assert result["metrics"]["diagnostic_confusion"] == {"tp": 0, "fp": 0, "fn": 1, "tn": 1}
        assert result["metrics"]["fraudulent_transactions_approved"] == 1
        assert result["metrics"]["decision_distribution"]["APPROVE"]["count"] == 2
        assert result["label_delays"]["observed_to_maturity"]["count"] == 3
        assert result["label_delays"]["transaction_to_observed"] is None
        assert "raw_payload" not in str(result)
        assert "raw_payload" not in str(client.get("/labels/1").json())
        scoped = client.get(
            "/feedback/summary",
            params={
                "from": (observed - timedelta(minutes=1)).isoformat(),
                "to": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
            },
        ).json()
        assert scoped["total_predictions"] == 3
        empty = client.get(
            "/feedback/summary",
            params={"from": "2020-01-01T00:00:00Z", "to": "2020-01-02T00:00:00Z"},
        ).json()
        assert empty["total_predictions"] == 0
        assert empty["status"] == "insufficient_labels"
        with store.sessions() as session:
            saved = session.get(FraudLabel, 1)
            assert saved.label_source == "synthetic_test"
            assert saved.applied_to_model_state_at is None
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(AuditEvent)
                    .where(AuditEvent.event_type == "label_received")
                )
                == 3
            )
            first = session.scalar(select(Prediction).where(Prediction.transaction_id == 1))
            first.calibrated_probability = 0.9
            session.commit()
        assert (
            client.get("/feedback/summary").json()["metrics"]["mean_predicted_fraud_probability"]
            > result["metrics"]["mean_predicted_fraud_probability"]
        )
        restored = StubRuntime()
        recovered = recover_runtime(restored, store)
        assert recovered.transaction_count == 3
        assert restored.snapshot_state().equals(snapshot)


def test_feedback_metric_edge_cases() -> None:
    assert (
        maturity_status(datetime(2026, 1, 2, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))
        == "IMMATURE"
    )
    assert calculate([])["pr_auc"] is None
    legitimate = {"is_fraud": False, "probability": 0.1, "decision": "REVIEW", "amount": 10.0}
    fraud = {"is_fraud": True, "probability": 0.9, "decision": "BLOCK", "amount": 30.0}
    assert calculate([legitimate])["roc_auc"] is None
    assert calculate([fraud])["pr_auc"] is None
    both = calculate([legitimate, fraud])
    assert both["pr_auc"] == 1.0
    assert both["roc_auc"] == 1.0
    assert both["diagnostic_confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}
    assert both["fraud_amount_capture"] == 1.0
    assert both["legitimate_review_rate"] == 1.0
    assert delay_summary([1, 2, 3])["median_seconds"] == 2.0


def test_immature_one_class_and_conflict_rollback() -> None:
    runtime = StubRuntime()
    app, _ = app_for(runtime)
    with TestClient(app) as client:
        for identifier in (11, 12):
            assert (
                client.post(
                    "/predict", json={"transaction": transaction(identifier, identifier * 10)}
                ).status_code
                == 200
            )
        observed = datetime.now(UTC)
        future = observed + timedelta(days=2)
        first = label(11, False, observed, future)
        assert client.post("/labels", json=first).status_code == 200
        immature = client.get("/feedback/summary").json()
        assert immature["mature_labels"] == 0
        assert immature["immature_labels"] == 1
        assert immature["metrics"]["brier"] is None
        assert (
            client.post(
                "/labels/batch",
                json={"labels": [label(12, True, observed, observed), {**first, "is_fraud": True}]},
            ).status_code
            == 409
        )
        assert client.get("/labels/12").status_code == 404
        at_maturity = client.get(
            "/feedback/summary", params={"evaluation_time": future.isoformat()}
        ).json()
        assert at_maturity["mature_labels"] == 1
        assert at_maturity["metrics"]["pr_auc"] is None
        assert at_maturity["metrics"]["roc_auc"] is None
        assert at_maturity["metrics"]["ranking_unavailable_reason"]
        assert client.post("/labels", json={**first, "is_fraud": 0}).status_code == 422
        assert (
            client.post(
                "/labels",
                json={
                    **first,
                    "matured_at": observed.isoformat(),
                    "observed_at": future.isoformat(),
                },
            ).status_code
            == 422
        )
