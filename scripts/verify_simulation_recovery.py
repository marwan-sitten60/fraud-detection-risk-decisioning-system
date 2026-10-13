"""Recreate the API against PostgreSQL and compare next-row score parity."""

import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from api.dependencies import ApiSettings  # noqa: E402
from api.main import create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fraud_detection.simulation.client import SimulationHTTPClient  # noqa: E402
from fraud_detection.simulation.source import (  # noqa: E402
    make_synthetic_source,
    prepare_inference_payload,
)


def main() -> None:
    if not os.environ.get("FRAUD_DATABASE_URL"):
        raise RuntimeError("FRAUD_DATABASE_URL is required; its value will not be displayed")
    run_id = uuid.uuid4().hex[:16]
    continuous = SimulationHTTPClient("http://127.0.0.1:8000", 180, retry_limit=2)
    health = continuous.request("GET", "/ready").json()
    if not health.get("ready") or not health.get("database_ready"):
        raise RuntimeError("Continuous FastAPI service is not ready")

    fixture = make_synthetic_source(1, seed=992)[0]
    candidate_dt = int(time.time()) + 10_000
    continuous_id = 1_000_000_000_000 + uuid.uuid4().int % 8_000_000_000_000_000_000
    recovered_id = continuous_id + 1
    next_payload_continuous = prepare_inference_payload(fixture, continuous_id, "healthy")
    next_payload_recovered = prepare_inference_payload(fixture, recovered_id, "healthy")

    settings = ApiSettings.from_env()
    with TestClient(create_app(settings)) as recovered:
        ready = recovered.get("/ready")
        if ready.status_code != 200 or not ready.json().get("state_recovered"):
            raise RuntimeError(f"Recreated API failed readiness/recovery: {ready.text}")
        recovered_info = ready.json()
        next_payload_continuous["TransactionDT"] = candidate_dt
        next_payload_recovered["TransactionDT"] = candidate_dt
        continuous_result = continuous.request(
            "POST",
            "/predict/batch",
            retries=True,
            headers={
                "X-Idempotency-Key": f"sim-{run_id}-next-continuous",
                "X-Request-ID": f"sim-{run_id}-cont",
            },
            json={"transactions": [next_payload_continuous]},
        ).json()
        recovered_result = recovered.post(
            "/predict/batch",
            headers={
                "X-Idempotency-Key": f"sim-{run_id}-next-recovered",
                "X-Request-ID": f"sim-{run_id}-rec",
            },
            json={"transactions": [next_payload_recovered]},
        )
        if recovered_result.status_code != 200:
            raise RuntimeError(f"Recovered parity prediction failed: {recovered_result.text}")
        left = continuous_result["predictions"][0]
        right = recovered_result.json()["predictions"][0]

    numeric = (
        "lightgbm_score",
        "xgboost_score",
        "ensemble_score",
        "calibrated_probability",
        "expected_cost_approve",
        "expected_cost_review",
        "expected_cost_block",
    )
    deltas = {name: abs(left[name] - right[name]) for name in numeric}
    decisions_match = left["base_decision"] == right["base_decision"] and (
        left.get("final_decision") or left["base_decision"]
    ) == (right.get("final_decision") or right["base_decision"])
    if not decisions_match or max(deltas.values()) > 1e-10:
        raise AssertionError(
            {"decisions_match": decisions_match, "max_abs_delta": max(deltas.values())}
        )
    outcome = {
        "status": "passed",
        "recreated_readiness": recovered_info,
        "replayed_transaction_count": recovered_info["replayed_transaction_count"],
        "continuous_transaction_id": continuous_id,
        "recovered_transaction_id": recovered_id,
        "max_abs_score_or_cost_delta": max(deltas.values()),
        "per_field_abs_delta": deltas,
        "decisions_match": decisions_match,
        "bundle_version": left["bundle_version"],
    }
    latest = ROOT / "reports/simulation_latest.json"
    if latest.exists():
        data = json.loads(latest.read_text(encoding="utf-8"))
        data["recovery_compatibility"] = outcome
        temporary = latest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(latest)
        run_path = ROOT / "reports/simulation_runs" / f"{data['simulation_run_id']}.json"
        if run_path.exists():
            temporary_run = run_path.with_suffix(".json.tmp")
            temporary_run.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
            temporary_run.replace(run_path)
    print(json.dumps(outcome, default=str), flush=True)
    continuous.close()


if __name__ == "__main__":
    main()
