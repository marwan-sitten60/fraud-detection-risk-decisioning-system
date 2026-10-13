"""Sequential HTTP replay runner and compact JSON report generation."""

import json
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import numpy as np

from fraud_detection.simulation.client import SimulationHTTPClient
from fraud_detection.simulation.config import SimulationConfig
from fraud_detection.simulation.events import (
    SimulationEvent,
    build_event_schedule,
    event_wall_time,
)
from fraud_detection.simulation.source import (
    ROOT,
    SourceTransaction,
    load_validation_source,
    make_synthetic_source,
    prepare_inference_payload,
)


@dataclass(slots=True)
class SimulationRunSummary:
    simulation_run_id: str
    scenario: str
    source: str
    dry_run: bool
    started_at: str
    ended_at: str | None = None
    configured_transaction_count: int = 0
    successfully_scored_count: int = 0
    failed_count: int = 0
    batch_count: int = 0
    average_batch_size: float | None = None
    target_rps: float | None = None
    time_scale: float = 1.0
    label_delay_min_days: float = 0.0
    label_delay_max_days: float = 0.0
    achieved_rps: float | None = None
    replay_lag_seconds: float = 0.0
    request_latency_p50_ms: float | None = None
    request_latency_p95_ms: float | None = None
    label_latency_p50_ms: float | None = None
    label_latency_p95_ms: float | None = None
    monitoring_query_latency_ms: dict[str, list[float]] = field(default_factory=dict)
    total_wall_time_seconds: float = 0.0
    decision_counts: dict[str, int] = field(default_factory=dict)
    labels_scheduled: int = 0
    labels_delivered: int = 0
    mature_label_count: int = 0
    immature_label_count: int = 0
    snapshot_ids: list[int] = field(default_factory=list)
    active_alert_count: int = 0
    run_alert_count: int = 0
    retry_count: int = 0
    bundle_version: str | None = None
    prediction_results: list[dict] = field(default_factory=list)
    label_events: list[dict] = field(default_factory=list)
    feedback_summary: dict | None = None
    monitoring_summary: dict | None = None
    monitoring_alerts: list[dict] = field(default_factory=list)
    recovery_compatibility: dict | None = None
    error: str | None = None


class SimulationRunner:
    def __init__(self, config: SimulationConfig, client=None) -> None:
        self.config = config
        self.client = client
        self._owns_client = client is None
        self._run_id = config.run_id or uuid.uuid4().hex
        if not self._run_id.replace("-", "").isalnum() or len(self._run_id) > 36:
            raise ValueError("run_id must be 1-36 alphanumeric or hyphen characters")
        self._sim_ids: list[int] = []
        self._scored: set[int] = set()
        self._request_latencies = []
        self._label_latencies = []
        self._decision_counts: Counter = Counter()
        self._summary: SimulationRunSummary | None = None
        self._schedule_started_at: datetime | None = None

    def load_source(self) -> list[SourceTransaction]:
        if self.config.source == "synthetic":
            return make_synthetic_source(self.config.max_transactions, self.config.seed)
        transactions = load_validation_source(
            ROOT / self.config.train_transaction_path,
            ROOT / self.config.train_identity_path,
            self.config.max_transactions,
        )
        transactions.sort(key=lambda item: (item.transaction_dt, item.source_transaction_id))
        return transactions

    def run(self, *, dry_run: bool = False) -> SimulationRunSummary:
        started = datetime.now(UTC)
        run_started = time.perf_counter()
        summary = SimulationRunSummary(
            simulation_run_id=self._run_id,
            scenario=self.config.scenario,
            source=self.config.source,
            dry_run=dry_run,
            started_at=started.isoformat(),
            target_rps=self.config.target_rps,
            time_scale=self.config.time_scale,
            label_delay_min_days=self.config.label_delay_min_days,
            label_delay_max_days=self.config.label_delay_max_days,
        )
        self._summary = summary
        try:
            transactions = self.load_source()
            if self.config.scenario == "drifted" and self.config.labels_enabled is None:
                summary.error = None
            summary.configured_transaction_count = len(transactions)
            summary.labels_scheduled = (
                sum(item.fraud_label is not None for item in transactions)
                if self.config.deliver_labels
                else 0
            )
            self._sim_ids = self._allocate_ids(len(transactions))
            schedule = build_event_schedule(transactions, self.config)
            summary.batch_count = sum(event.event_type == "prediction" for event in schedule)
            summary.average_batch_size = (
                summary.configured_transaction_count / summary.batch_count
                if summary.batch_count
                else None
            )
            if dry_run:
                return summary

            if self.client is None:
                self.client = SimulationHTTPClient(
                    self.config.base_url,
                    self.config.timeout_seconds,
                    self.config.retry_limit,
                )
            bundle = self._startup_check()
            summary.bundle_version = bundle
            self._schedule_started_at = datetime.now(UTC)
            wall_origin = time.perf_counter()
            last_prediction_elapsed = 0.0
            for event in schedule:
                self._wait_until(wall_origin, event.offset_seconds)
                if event.event_type == "prediction":
                    self._send_prediction_batch(transactions, event, summary, wall_origin)
                    last_prediction_elapsed = time.perf_counter() - wall_origin
                elif event.event_type == "label":
                    self._send_label(transactions, event, summary)
                elif event.event_type == "monitoring":
                    self._take_snapshot(summary)
            # A final snapshot and final read-only service summaries are mandatory.
            self._take_snapshot(summary)
            self._final_queries(summary, started)
            summary.successfully_scored_count = len(self._scored)
            summary.failed_count = max(
                0, summary.configured_transaction_count - summary.successfully_scored_count
            )
            summary.decision_counts = dict(self._decision_counts)
            summary.retry_count = getattr(self.client, "retry_count", 0)
            elapsed = max(last_prediction_elapsed, 1e-9)
            summary.achieved_rps = summary.successfully_scored_count / elapsed
            summary.total_wall_time_seconds = time.perf_counter() - run_started
            due_prediction_events = [
                event.offset_seconds for event in schedule if event.event_type == "prediction"
            ]
            if due_prediction_events:
                summary.replay_lag_seconds = max(
                    0.0,
                    last_prediction_elapsed - due_prediction_events[-1],
                )
            summary.request_latency_p50_ms = self._percentile(self._request_latencies, 50)
            summary.request_latency_p95_ms = self._percentile(self._request_latencies, 95)
            summary.label_latency_p50_ms = self._percentile(self._label_latencies, 50)
            summary.label_latency_p95_ms = self._percentile(self._label_latencies, 95)
        except KeyboardInterrupt:
            summary.error = "interrupted"
            summary.successfully_scored_count = len(self._scored)
            summary.failed_count = max(
                0, summary.configured_transaction_count - summary.successfully_scored_count
            )
            raise
        except Exception as error:
            summary.error = f"{type(error).__name__}: {error}"
            raise
        finally:
            if summary.total_wall_time_seconds == 0:
                summary.total_wall_time_seconds = time.perf_counter() - run_started
            summary.decision_counts = dict(self._decision_counts)
            summary.retry_count = getattr(self.client, "retry_count", 0)
            self._finish(summary, run_started)
            if self._owns_client and self.client is not None:
                self.client.close()
        return summary

    def _startup_check(self) -> str:
        health = self._request("GET", "/health").json()
        if health.get("status") != "ok":
            raise RuntimeError("FastAPI liveness check failed")
        ready = self._request("GET", "/ready").json()
        if (
            not ready.get("ready")
            or not ready.get("database_ready")
            or not ready.get("state_recovered")
        ):
            raise RuntimeError("FastAPI is not fully ready; replay aborted")
        version = self._request("GET", "/version").json()
        if version.get("bundle_version") != ready.get("bundle_version"):
            raise RuntimeError("Service version and readiness bundle versions differ")
        return version["bundle_version"]

    @staticmethod
    def _allocate_ids(count: int) -> list[int]:
        nonce = uuid.uuid4().int
        base = 1_000_000_000_000 + nonce % 8_000_000_000_000_000_000
        if base + count >= 9_223_372_036_854_775_807:
            raise ValueError("Could not allocate safe simulator TransactionIDs")
        return [base + index for index in range(count)]

    @staticmethod
    def _wait_until(origin: float, offset: float) -> None:
        remaining = origin + max(0.0, offset) - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    def _send_prediction_batch(
        self,
        transactions: list[SourceTransaction],
        event: SimulationEvent,
        summary: SimulationRunSummary,
        wall_origin: float,
    ) -> None:
        payloads = [
            prepare_inference_payload(
                transactions[index], self._sim_ids[index], self.config.scenario
            )
            for index in event.transaction_indexes
        ]
        if any("isFraud" in payload for payload in payloads):
            raise AssertionError("Target label leaked into prediction request")
        request_sequence = event.transaction_indexes[0] // self.config.batch_size
        key = f"sim-{self._run_id}-predict-{request_sequence}"
        headers = {
            "X-Idempotency-Key": key,
            "X-Request-ID": f"sim-{self._run_id}-p-{request_sequence}",
        }
        start = time.perf_counter()
        try:
            response = self._request(
                "POST",
                "/predict/batch",
                retries=True,
                headers=headers,
                json={"transactions": payloads},
            )
        except Exception as error:
            summary.failed_count += len(payloads)
            if getattr(error, "status_code", None) == 409:
                # A semantic conflict is recorded once; it is never retried.
                return
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        self._request_latencies.append(latency_ms)
        result = response.json()
        predictions = result.get("predictions", [])
        if result.get("count") != len(payloads) or len(predictions) != len(payloads):
            raise RuntimeError("Batch prediction response count does not match submitted rows")
        if result.get("bundle_version") != summary.bundle_version:
            raise RuntimeError("Prediction response bundle differs from the startup bundle")
        for index, row in zip(event.transaction_indexes, predictions, strict=True):
            if row.get("TransactionID") != self._sim_ids[index]:
                raise RuntimeError("Prediction response transaction order/identity mismatch")
            probability = row.get("calibrated_probability")
            decision = row.get("final_decision") or row.get("base_decision")
            if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                raise RuntimeError("Prediction response has invalid calibrated probability")
            if decision not in {"APPROVE", "REVIEW", "BLOCK"}:
                raise RuntimeError("Prediction response has an invalid decision")
            self._scored.add(index)
            self._decision_counts[decision] += 1
            summary.prediction_results.append(
                {
                    "TransactionID": self._sim_ids[index],
                    "calibrated_probability": float(probability),
                    "base_decision": row["base_decision"],
                    "final_decision": decision,
                    "bundle_version": result["bundle_version"],
                    "request_latency_ms": round(latency_ms, 3),
                }
            )
        summary.successfully_scored_count = len(self._scored)
        summary.retry_count = getattr(self.client, "retry_count", 0)

    def _send_label(
        self,
        transactions: list[SourceTransaction],
        event: SimulationEvent,
        summary: SimulationRunSummary,
    ) -> None:
        index = event.label_index
        if index is None or index not in self._scored:
            return
        source = transactions[index]
        if source.fraud_label is None:
            return
        now = datetime.now(UTC)
        payload = {
            "TransactionID": self._sim_ids[index],
            "is_fraud": bool(source.fraud_label),
            "label_source": "sim_validation"
            if self.config.source == "validation"
            else "sim_synthetic",
            "observed_at": now.isoformat(),
            "matured_at": now.isoformat(),
            "metadata": {"simulation_run_id": self._run_id, "scenario": self.config.scenario},
        }
        request_start = time.perf_counter()
        response = self._request(
            "POST",
            "/labels",
            retries=True,
            headers={"X-Request-ID": f"sim-{self._run_id}-label-{index}"},
            json=payload,
        )
        latency_ms = (time.perf_counter() - request_start) * 1000
        self._label_latencies.append(latency_ms)
        item = response.json()
        if item.get("TransactionID") != self._sim_ids[index]:
            raise RuntimeError("Label response transaction identity mismatch")
        status = item.get("maturity_status")
        summary.labels_delivered += 1
        summary.mature_label_count += status == "MATURE"
        summary.immature_label_count += status == "IMMATURE"
        summary.label_events.append(
            {
                "TransactionID": self._sim_ids[index],
                "scheduled_offset_seconds": event.offset_seconds,
                "scheduled_label_at": (
                    event_wall_time(self._schedule_started_at, event.offset_seconds).isoformat()
                    if self._schedule_started_at
                    else None
                ),
                "actual_send_at": now.isoformat(),
                "observed_at": item.get("observed_at"),
                "matured_at": item.get("matured_at"),
                "maturity_status": status,
                "replayed": item.get("replayed", False),
                "http_status": response.status_code,
                "response_latency_ms": round(latency_ms, 3),
            }
        )

    def _take_snapshot(self, summary: SimulationRunSummary) -> None:
        end = datetime.now(UTC)
        start = datetime.fromisoformat(summary.started_at)
        params = {"from": start.isoformat(), "to": end.isoformat()}
        before = time.perf_counter()
        response = self._request("POST", "/monitoring/snapshot", params=params, retries=True)
        summary.monitoring_query_latency_ms.setdefault("snapshot", []).append(
            (time.perf_counter() - before) * 1000
        )
        result = response.json()
        snapshot_id = result.get("snapshot_id")
        if snapshot_id is not None:
            summary.snapshot_ids.append(int(snapshot_id))
        summary.run_alert_count = len(result.get("alerts", []))

    def _final_queries(self, summary: SimulationRunSummary, started: datetime) -> None:
        end = datetime.now(UTC)
        params = {"from": started.isoformat(), "to": end.isoformat()}
        for path, key in (
            ("/monitoring/summary", "monitoring_summary"),
            ("/feedback/summary", "feedback_summary"),
            ("/monitoring/alerts", "monitoring_alerts"),
        ):
            before = time.perf_counter()
            response = self._request("GET", path, params=params if "summary" in path else None)
            summary.monitoring_query_latency_ms.setdefault(key, []).append(
                (time.perf_counter() - before) * 1000
            )
            data = response.json()
            if key == "monitoring_alerts":
                summary.monitoring_alerts = data.get("alerts", [])
                summary.active_alert_count = data.get("count", len(summary.monitoring_alerts))
            else:
                setattr(summary, key, data)

    def _request(self, method: str, path: str, **kwargs):
        try:
            return self.client.request(method, path, **kwargs)
        except TypeError:
            # Lightweight injected test clients may implement a narrower call signature.
            kwargs.pop("retries", None)
            return self.client.request(method, path, **kwargs)

    @staticmethod
    def _percentile(values: list[float], percentile: int) -> float | None:
        return float(np.percentile(values, percentile)) if values else None

    def _finish(self, summary: SimulationRunSummary, run_started: float) -> SimulationRunSummary:
        summary.ended_at = datetime.now(UTC).isoformat()
        if not summary.total_wall_time_seconds:
            summary.total_wall_time_seconds = time.perf_counter() - run_started
        summary.request_latency_p50_ms = self._percentile(self._request_latencies, 50)
        summary.request_latency_p95_ms = self._percentile(self._request_latencies, 95)
        summary.label_latency_p50_ms = self._percentile(self._label_latencies, 50)
        summary.label_latency_p95_ms = self._percentile(self._label_latencies, 95)
        report_dir = ROOT / self.config.report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{self._run_id}.json"
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(asdict(summary), indent=2, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(report_path)
        if self.config.write_latest_report:
            latest_path = ROOT / "reports/simulation_latest.json"
            temp_latest = latest_path.with_suffix(".json.tmp")
            temp_latest.write_text(
                json.dumps(asdict(summary), indent=2, allow_nan=False), encoding="utf-8"
            )
            temp_latest.replace(latest_path)
        print(
            json.dumps(
                {
                    "run_id": summary.simulation_run_id,
                    "scenario": summary.scenario,
                    "sent": summary.successfully_scored_count,
                    "failed": summary.failed_count,
                    "achieved_rps": summary.achieved_rps,
                    "pending_labels": max(0, summary.labels_scheduled - summary.labels_delivered),
                    "alerts": summary.active_alert_count,
                }
            ),
            flush=True,
        )
        return summary
