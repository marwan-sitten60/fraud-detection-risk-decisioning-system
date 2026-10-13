"""Distribution metrics, health windows, and durable snapshot behavior."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from fraud_detection.monitoring.drift import categorical_js, numeric_psi
from fraud_detection.monitoring.reference import MonitoringReference
from fraud_detection.persistence.models import FraudLabel, MonitoringAlert, MonitoringSnapshot
from test_api import StubRuntime, app_for, transaction

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "models/monitoring_v1"


def test_fixed_bin_psi_and_unseen_category() -> None:
    edges = ["-inf", 0.5, "inf"]
    reference = [0.5, 0.5, 0.0]
    steady = numeric_psi([0.2, 0.8], edges, reference)
    shifted = numeric_psi([0.1] * 50, edges, reference)
    assert steady["psi"] == pytest.approx(0, abs=1e-8)
    assert shifted["psi"] > 0.2
    cats = categorical_js(
        ["new", "new"],
        {
            "known_categories": ["seen"],
            "all_known_categories": ["seen", "tail"],
            "frequencies": {"seen": 0.5, "__OTHER__": 0.5},
        },
    )
    assert cats["unseen_rate"] == 1
    assert cats["js_divergence"] > 0
    assert not numeric_psi([], edges, reference)["sample_count"]
    assert not categorical_js([], {"known_categories": [], "frequencies": {}})["sample_count"]


def test_reference_fixed_bins_and_hash() -> None:
    reference = MonitoringReference(REFERENCE)
    first = reference.data["numeric"]["TransactionAmt"]["edges"]
    second = MonitoringReference(REFERENCE).data["numeric"]["TransactionAmt"]["edges"]
    assert first == second
    assert reference.data["reference_sources"]["holdout_used"] is False
    assert reference.data["sample_counts"]["train_inputs"] == 413378


def test_monitoring_summary_window_and_snapshot_do_not_score() -> None:
    runtime = StubRuntime()
    app, _ = app_for(runtime, maximum=32)
    with TestClient(app) as client:
        start = datetime.now(UTC) - timedelta(minutes=30)
        end = datetime.now(UTC) + timedelta(minutes=1)
        rows = [transaction(idx, idx * 10, 500) for idx in range(1, 31)]
        assert client.post("/predict/batch", json={"transactions": rows}).status_code == 200
        before = runtime.snapshot_state()
        summary = client.get(
            "/monitoring/summary",
            params={"from": start.isoformat(), "to": end.isoformat()},
        )
        assert summary.status_code == 200, summary.text
        result = summary.json()
        assert result["system"]["request_count"] == 1
        assert result["system"]["transactions_scored"] == 30
        assert result["system"]["success_rate"] == 1
        assert result["system"]["latency_ms"]["p50"] <= result["system"]["latency_ms"]["p95"]
        assert result["system"]["model_output"]["decision_distribution"]["APPROVE"]["count"] == 30
        assert result["feedback"]["metrics"]["sample_size"] == 0
        assert result["feedback"]["metrics"]["pr_auc"] is None
        assert result["feedback"]["status"] == "insufficient_labels"
        assert "raw_payload" not in summary.text
        assert all(before[key].equals(runtime.snapshot_state()[key]) for key in before)
        empty = client.get(
            "/monitoring/summary",
            params={"from": "2020-01-01T00:00:00Z", "to": "2020-01-02T00:00:00Z"},
        )
        assert empty.status_code == 200, empty.text
        assert empty.json()["system"]["request_count"] == 0
        assert empty.json()["system"]["error_rate"] is None
        with client.app.state.store.sessions() as session:
            labels_before = session.scalar(select(func.count()).select_from(FraudLabel))
        snap = client.post(
            "/monitoring/snapshot",
            params={"from": start.isoformat(), "to": end.isoformat()},
        )
        assert snap.status_code == 200
        assert isinstance(snap.json()["snapshot_id"], int)
        assert snap.json()["system"]["transactions_scored"] == 30
        assert snap.json()["alerts"]
        assert all(before[key].equals(runtime.snapshot_state()[key]) for key in before)
        with client.app.state.store.sessions() as session:
            assert session.scalar(select(func.count()).select_from(FraudLabel)) == labels_before
        with client.app.state.store.sessions() as session:
            saved = session.scalar(
                select(MonitoringSnapshot).order_by(MonitoringSnapshot.id.desc())
            )
            assert saved.transaction_count == 30
            session.add(
                MonitoringAlert(
                    snapshot_id=saved.id,
                    alert_type="INPUT_DRIFT",
                    severity="WARNING",
                    metric_name="synthetic",
                    observed_value=0.25,
                    threshold_reference="PSI .20/.30",
                    message="Synthetic persisted alert",
                )
            )
            session.commit()
        alerts = client.get("/monitoring/alerts?severity=warning")
        assert alerts.status_code == 200
        warning_count = sum(alert["severity"] == "WARNING" for alert in snap.json()["alerts"])
        assert alerts.json()["count"] == warning_count + 1
        assert alerts.json()["alerts"][0]["severity"] == "WARNING"
        with client.app.state.store.sessions() as session:
            assert session.scalar(select(func.count()).select_from(MonitoringSnapshot)) == 1
            assert (
                session.scalar(select(func.count()).select_from(MonitoringAlert))
                == len(snap.json()["alerts"]) + 1
            )


def test_alert_severity_and_performance_support_are_deterministic() -> None:
    from fraud_detection.monitoring.alerts import (
        MIN_PERFORMANCE_PER_CLASS,
        MIN_PERFORMANCE_ROWS,
        build_alerts,
        performance_supported,
        severity,
    )

    assert severity(0.19, 0.2, 0.3) is None
    assert severity(0.2, 0.2, 0.3) == "WARNING"
    assert severity(0.3, 0.2, 0.3) == "CRITICAL"
    assert not performance_supported(199, 30, 169)
    assert not performance_supported(200, MIN_PERFORMANCE_PER_CLASS - 1, 181)
    assert performance_supported(MIN_PERFORMANCE_ROWS, 20, 180)
    base = {
        "drift": {
            "inputs": {"amount": {"psi": 0.0}},
            "scores": {"probability": {"psi": 0.0}},
            "decisions": {"js_divergence": 0.0},
        },
        "system": {
            "transactions_scored": 40,
            "error_rate": 0.0,
            "latency_ms": {"p95": 10.0},
        },
        "feedback": {
            "mature_label_coverage": 0.8,
            "metrics": {"sample_size": 0},
        },
    }
    reference = {"performance_reference": {"pr_auc": 0.6, "brier": 0.05}}
    assert build_alerts(base, reference) == []
    shifted = {
        **base,
        "drift": {
            "inputs": {"amount": {"psi": 0.35}},
            "scores": {"probability": {"psi": 0.0}},
            "decisions": {"js_divergence": 0.0},
        },
    }
    assert build_alerts(shifted, reference)[0]["severity"] == "CRITICAL"
    immature = {
        **base,
        "system": {**base["system"], "transactions_scored": 200},
        "feedback": {
            "mature_label_coverage": 0.1,
            "metrics": {"sample_size": 100, "fraud_prevalence": 0.2},
        },
    }
    assert not any(
        a["alert_type"] == "PERFORMANCE_DEGRADATION" for a in build_alerts(immature, reference)
    )
