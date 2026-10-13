"""Deterministic simulation scheduling, privacy, and HTTP lifecycle tests."""

import json
from dataclasses import asdict

import httpx2
import pytest

from fraud_detection.simulation.client import SimulationHTTPClient, SimulationHTTPError
from fraud_detection.simulation.config import SimulationConfig
from fraud_detection.simulation.events import build_event_schedule
from fraud_detection.simulation.runner import SimulationRunner
from fraud_detection.simulation.source import (
    make_synthetic_source,
    prepare_inference_payload,
)


class Response:
    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeClient:
    def __init__(self, ready=True):
        self.ready = ready
        self.calls = []
        self.retry_count = 0

    def request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        if path == "/health":
            return Response({"status": "ok"})
        if path == "/ready":
            return Response(
                {
                    "ready": self.ready,
                    "database_ready": self.ready,
                    "state_recovered": self.ready,
                    "bundle_version": "frozen_v1",
                }
            )
        if path == "/version":
            return Response({"bundle_version": "frozen_v1"})
        if path == "/predict/batch":
            payload = kwargs["json"]["transactions"]
            assert all("isFraud" not in row for row in payload)
            return Response(
                {
                    "count": len(payload),
                    "bundle_version": "frozen_v1",
                    "predictions": [
                        {
                            "TransactionID": row["TransactionID"],
                            "calibrated_probability": 0.1,
                            "base_decision": "APPROVE",
                            "final_decision": "APPROVE",
                            "bundle_version": "frozen_v1",
                        }
                        for row in payload
                    ],
                }
            )
        if path == "/labels":
            body = kwargs["json"]
            return Response(
                {
                    "TransactionID": body["TransactionID"],
                    "observed_at": body["observed_at"],
                    "matured_at": body["matured_at"],
                    "maturity_status": "MATURE",
                    "replayed": False,
                }
            )
        if path == "/monitoring/snapshot":
            return Response({"snapshot_id": len([c for c in self.calls if c[1] == path])})
        if path == "/monitoring/summary":
            return Response({"system": {"transactions_scored": 4}})
        if path == "/feedback/summary":
            return Response(
                {"status": "insufficient_labels", "labels_received": 0, "mature_labels": 0}
            )
        if path == "/monitoring/alerts":
            return Response({"count": 0, "alerts": []})
        raise AssertionError(f"Unexpected route {path}")


def config(tmp_path, **overrides):
    values = {
        "source": "synthetic",
        "max_transactions": 8,
        "batch_size": 3,
        "time_scale": 1e9,
        "label_delay_min_days": 0,
        "label_delay_max_days": 0,
        "report_dir": str(tmp_path),
        "write_latest_report": False,
    }
    values.update(overrides)
    return SimulationConfig(**values)


def test_event_schedule_is_seeded_chronological_and_keeps_batch_order():
    transactions = make_synthetic_source(12, seed=7)
    options = SimulationConfig(
        source="synthetic",
        max_transactions=12,
        batch_size=4,
        target_rps=2,
        time_scale=100,
        seed=19,
    )
    first = build_event_schedule(transactions, options)
    second = build_event_schedule(transactions, options)
    assert first == second
    predictions = [event for event in first if event.event_type == "prediction"]
    assert [event.transaction_indexes for event in predictions] == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
    ]
    assert [event.offset_seconds for event in predictions] == sorted(
        event.offset_seconds for event in predictions
    )
    label_events = [event for event in first if event.event_type == "label"]
    assert label_events and all(event.label_index is not None for event in label_events)


def test_drifted_defaults_labels_off_and_mutates_only_copy():
    source = make_synthetic_source(4, seed=5)
    options = SimulationConfig(scenario="drifted", source="synthetic", max_transactions=4)
    assert not options.deliver_labels
    assert not any(event.event_type == "label" for event in build_event_schedule(source, options))
    original = dict(source[0].payload)
    changed = prepare_inference_payload(source[0], 4, "drifted")
    assert changed["TransactionAmt"] != original["TransactionAmt"]
    assert changed["ProductCD"] == "SIMULATOR_UNSEEN"
    assert "isFraud" not in changed
    assert source[0].payload == original


def test_http_simulation_hides_labels_delivers_later_and_captures_queries(tmp_path):
    source = make_synthetic_source(8, seed=10)
    client = FakeClient()
    runner = SimulationRunner(config(tmp_path), client=client)
    summary = runner.run()
    assert summary.successfully_scored_count == 8
    assert summary.batch_count == 3
    assert summary.average_batch_size == pytest.approx(8 / 3)
    assert summary.labels_scheduled == 8
    assert summary.labels_delivered == 8
    assert summary.mature_label_count == 8
    assert summary.snapshot_ids == [1, 2]
    assert summary.monitoring_query_latency_ms["snapshot"]
    assert summary.feedback_summary["status"] == "insufficient_labels"
    assert summary.monitoring_summary["system"]["transactions_scored"] == 4
    assert len(summary.prediction_results) == 8
    prediction_calls = [call for call in client.calls if call[1] == "/predict/batch"]
    keys = [call[2]["headers"]["X-Idempotency-Key"] for call in prediction_calls]
    assert len(keys) == len(set(keys)) == 3
    assert all(
        "isFraud" not in row for call in prediction_calls for row in call[2]["json"]["transactions"]
    )
    report = json.loads((tmp_path / f"{summary.simulation_run_id}.json").read_text())
    assert "TransactionAmt" not in json.dumps(report)
    assert "source_transaction_id" not in json.dumps(report)
    assert "is_fraud" not in json.dumps(report)
    assert source[0].fraud_label is not None
    assert all(event["scheduled_label_at"] for event in summary.label_events)
    assert all(event["http_status"] == 200 for event in summary.label_events)


def test_dry_run_never_contacts_service_and_is_json_serializable(tmp_path):
    client = FakeClient()
    runner = SimulationRunner(config(tmp_path, max_transactions=5), client=client)
    summary = runner.run(dry_run=True)
    assert summary.dry_run
    assert summary.configured_transaction_count == 5
    assert summary.batch_count == 2
    assert summary.labels_scheduled == 5
    assert client.calls == []
    json.dumps(asdict(summary))
    json.loads((tmp_path / f"{summary.simulation_run_id}.json").read_text())


def test_unready_service_aborts_before_prediction(tmp_path):
    client = FakeClient(ready=False)
    runner = SimulationRunner(config(tmp_path, max_transactions=3), client=client)
    with pytest.raises(RuntimeError, match="not fully ready"):
        runner.run()
    assert not any(path == "/predict/batch" for _, path, _ in client.calls)


class RetryTransport:
    def __init__(self, error=None, status=200):
        self.error = error
        self.status = status
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None and len(self.calls) == 1:
            raise self.error
        return Response({"ok": True}, status=self.status)

    def close(self):
        pass


def test_transient_retry_reuses_key_and_conflict_is_never_retried():
    transport = RetryTransport(error=httpx2.ConnectError("transient"))
    client = SimulationHTTPClient("http://service", 1, retry_limit=2)
    client.client.close()
    client.client = transport
    response = client.request(
        "POST",
        "/predict/batch",
        retries=True,
        headers={"X-Idempotency-Key": "sim-test-predict-1"},
        json={"transactions": []},
    )
    assert response.json()["ok"]
    assert len(transport.calls) == 2
    assert transport.calls[0]["headers"] == transport.calls[1]["headers"]
    assert client.retry_count == 1

    conflict = RetryTransport(status=409)
    client.client = conflict
    with pytest.raises(SimulationHTTPError) as error:
        client.request("POST", "/predict/batch", retries=True)
    assert error.value.status_code == 409
    assert len(conflict.calls) == 1
