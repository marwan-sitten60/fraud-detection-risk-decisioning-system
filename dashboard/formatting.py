"""Safe formatting and allow-list helpers shared by dashboard pages."""

from __future__ import annotations

import json
from pathlib import Path

DECISION_ORDER = {"APPROVE": 0, "REVIEW": 1, "BLOCK": 2}
SEVERITY_ORDER = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
SAFE_SIMULATION_FIELDS = (
    "simulation_run_id",
    "scenario",
    "source",
    "started_at",
    "ended_at",
    "configured_transaction_count",
    "successfully_scored_count",
    "failed_count",
    "target_rps",
    "achieved_rps",
    "replay_lag_seconds",
    "request_latency_p50_ms",
    "request_latency_p95_ms",
    "labels_delivered",
    "mature_label_count",
    "immature_label_count",
    "active_alert_count",
    "run_alert_count",
    "bundle_version",
)


def format_risk(value: object) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "—"


def format_decision(value: object) -> str:
    text = str(value or "").upper()
    return text if text in DECISION_ORDER else "Unknown"


def risk_decision_note() -> str:
    return (
        "Calibrated fraud probability estimates risk. The final action also reflects "
        "expected-cost policy."
    )


def drift_status(value: object) -> str:
    normalized = str(value or "unknown").lower()
    return {
        "stable": "Stable",
        "drifted": "Drift detected",
        "insufficient_data": "Insufficient data",
    }.get(normalized, "Unavailable")


def sort_alerts(alerts: list[dict]) -> list[dict]:
    return sorted(
        alerts,
        key=lambda item: SEVERITY_ORDER.get(str(item.get("severity", "")).upper(), 3),
    )


def safe_transaction_view(payload: dict) -> dict:
    """Keep only named audit fields, even if an API response grows new fields."""
    keys = (
        "TransactionID",
        "scored_at",
        "calibrated_probability",
        "lightgbm_score",
        "xgboost_score",
        "ensemble_score",
        "base_decision",
        "final_decision",
        "decision",
        "expected_cost_approve",
        "expected_cost_review",
        "expected_cost_block",
        "triggered_rule_ids",
        "bundle_version",
    )
    return {key: payload[key] for key in keys if key in payload}


def _safe_simulation(payload: object) -> dict | None:
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("simulation_run_id"), str):
        return None
    return {key: payload.get(key) for key in SAFE_SIMULATION_FIELDS if key in payload}


def load_simulation_summaries(report_dir: Path) -> tuple[list[dict], int]:
    """Read run summaries, drop all unlisted data, and skip malformed files safely."""
    if not report_dir.exists():
        return [], 0
    summaries: dict[str, dict] = {}
    errors = 0
    paths = []
    latest = report_dir / "simulation_latest.json"
    if latest.is_file():
        paths.append(latest)
    run_reports = sorted(
        (report_dir / "simulation_runs").glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    paths.extend(run_reports)
    paths.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    for path in paths:
        try:
            summary = _safe_simulation(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            errors += 1
            continue
        if summary is None:
            errors += 1
            continue
        summaries.setdefault(summary["simulation_run_id"], summary)
    return list(summaries.values()), errors
