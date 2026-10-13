"""Dashboard support tests stay independent of Streamlit's browser runtime."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

from dashboard.client import APIClient
from dashboard.components import time_window_params
from dashboard.formatting import (
    drift_status,
    format_decision,
    format_risk,
    load_simulation_summaries,
    risk_decision_note,
    safe_transaction_view,
    sort_alerts,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    status = 200

    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def test_api_client_parses_health_response(monkeypatch):
    monkeypatch.setattr(
        "dashboard.client.urlopen", lambda *_args, **_kwargs: FakeResponse(b'{"status":"ok"}')
    )
    result = APIClient("http://local.test").get("/health")
    assert result.ok
    assert result.data == {"status": "ok"}


def test_api_client_handles_connection_and_http_errors(monkeypatch):
    def offline(*_args, **_kwargs):
        raise URLError("private connection details")

    monkeypatch.setattr("dashboard.client.urlopen", offline)
    offline_result = APIClient().get("/health")
    assert not offline_result.ok
    assert "private connection details" not in offline_result.error

    def rejected(*_args, **_kwargs):
        raise HTTPError("http://local.test", 503, "private detail", {}, None)

    monkeypatch.setattr("dashboard.client.urlopen", rejected)
    error_result = APIClient().get("/ready")
    assert error_result.status_code == 503
    assert "private detail" not in error_result.error


def test_unavailable_feedback_metrics_and_risk_decision_formatting():
    from dashboard.pages.feedback import _metric

    class FakeStreamlit:
        def metric(self, _label, value):
            self.value = value

    fake = FakeStreamlit()
    _metric(fake, "PR-AUC", {"pr_auc": None}, "pr_auc")
    assert fake.value == "N/A"
    assert format_risk(0.125) == "12.50%"
    assert format_decision("review") == "REVIEW"
    assert format_decision("unexpected") == "Unknown"
    assert "expected-cost" in risk_decision_note()


def test_drift_status_is_deterministic():
    assert drift_status("drifted") == "Drift detected"
    assert drift_status("stable") == "Stable"
    assert drift_status("unknown-status") == "Unavailable"


def test_alerts_are_sorted_by_severity_stably():
    alerts = [
        {"severity": "INFO", "metric_name": "i"},
        {"severity": "WARNING", "metric_name": "w"},
        {"severity": "CRITICAL", "metric_name": "c1"},
        {"severity": "CRITICAL", "metric_name": "c2"},
    ]
    assert [item["metric_name"] for item in sort_alerts(alerts)] == ["c1", "c2", "w", "i"]


def test_transaction_view_allowlist_never_returns_raw_payload():
    result = safe_transaction_view(
        {
            "TransactionID": 42,
            "calibrated_probability": 0.2,
            "raw_payload": {"private": "secret"},
            "feature_vector": [1, 2],
            "isFraud": 1,
        }
    )
    assert result == {"TransactionID": 42, "calibrated_probability": 0.2}
    assert "raw_payload" not in result and "isFraud" not in result


def test_simulation_loader_handles_malformed_and_strips_raw_output(tmp_path):
    run_dir = tmp_path / "simulation_runs"
    run_dir.mkdir()
    (tmp_path / "simulation_latest.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "unrelated.json").write_text("not a simulation", encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "simulation_run_id": "run-1",
                "scenario": "healthy",
                "successfully_scored_count": 4,
                "prediction_results": [{"TransactionID": 9, "calibrated_probability": 0.4}],
                "label_events": [{"is_fraud": True}],
                "feature_vector": [1, 2],
            }
        ),
        encoding="utf-8",
    )
    rows, errors = load_simulation_summaries(tmp_path)
    assert errors == 1
    assert rows == [
        {
            "simulation_run_id": "run-1",
            "scenario": "healthy",
            "successfully_scored_count": 4,
        }
    ]
    assert "prediction_results" not in rows[0]
    assert "label_events" not in rows[0]


def test_time_window_uses_backend_window_semantics_and_utc_bounds():
    now = datetime(2026, 10, 13, 12, tzinfo=UTC)
    assert time_window_params("1h", now) == {
        "window": "1h",
        "from": "2026-10-13T11:00:00+00:00",
        "to": "2026-10-13T12:00:00+00:00",
    }
    assert time_window_params("24h", now)["from"] == "2026-10-12T12:00:00+00:00"


def test_time_window_query_is_passed_through_to_backend(monkeypatch):
    captured = {}

    def fake_open(request, **_kwargs):
        captured["query"] = parse_qs(urlparse(request.full_url).query)
        return FakeResponse(b'{"system": {}}')

    monkeypatch.setattr("dashboard.client.urlopen", fake_open)
    params = time_window_params("7d", datetime(2026, 10, 13, 12, tzinfo=UTC))
    result = APIClient("http://local.test").get("/monitoring/summary", params)
    assert result.ok
    assert captured["query"] == {
        "window": ["7d"],
        "from": ["2026-10-06T12:00:00+00:00"],
        "to": ["2026-10-13T12:00:00+00:00"],
    }


def test_streamlit_navigation_app_starts_without_an_api(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("FRAUD_API_BASE_URL", "http://127.0.0.1:1")
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert "Operational overview" in [item.value for item in app.title]


def test_frozen_bundle_monitoring_reference_and_holdout_hashes_unchanged():
    expected = [
        (
            "models/frozen_v1/feature_state.joblib",
            "2a7f86d6d4b417b4c948b0e2fa49449d8e06e147fe330d9f6c957cca00197af5",
        ),
        (
            "models/frozen_v1/lightgbm.txt",
            "201cb1747d25b4dc888269fb41cd389f989f2545a5ee6a6b3aab273d2da462cd",
        ),
        (
            "models/frozen_v1/xgboost.json",
            "2c13ed79860cfc593623d09932c135b6b4ed0cc34502f983a7bd089218c17a6c",
        ),
        (
            "models/frozen_v1/platt_calibrator.joblib",
            "3503bf66161b5152e9de4d638b9e7932d581fc371abd6a45b6969ba6c8134f6e",
        ),
        (
            "models/frozen_v1/rules_frozen_config.json",
            "3a4380831e177828c82a1a118259f83e354221fa52205b41d38aa21b58febb94",
        ),
        (
            "models/frozen_v1/policy.json",
            "3685f31c1af31acfb68a98d4003164fc19051b938dfc37fa9503587da669470c",
        ),
        (
            "models/frozen_v1/feature_contract.json",
            "047700d867cf360770bd0c419a0d672dd702a4a63a825569bede7897b4b54100",
        ),
        (
            "models/monitoring_v1/reference.json",
            "502e3a882c406e9dbaf971a9d965d5d79aa09557f736e57e648d3371abc0be1a",
        ),
        (
            "reports/final_holdout_evaluation_summary.md",
            "8de96f153ffd3b0e398ab0c7fb257838447e101d020aae19594bc369f12ee750",
        ),
        (
            "reports/holdout_predictions_frozen.csv",
            "0d7f5d117436cd7227d682388c5cceddbc72b689143984a7d186d9ec03c1391f",
        ),
        (
            "reports/holdout_prediction_lock.json",
            "8b849e9e087e4cb13b39a132bd3451ea5732555fd11860ce194f4f925ce4676b",
        ),
    ]
    for relative, digest in expected:
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == digest, relative
