"""Small label-free end-to-end parity and restart recovery for the frozen bundle."""

import gc
import json
from pathlib import Path

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from api.dependencies import ApiSettings
from api.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from fraud_detection.inference import FrozenFraudInferenceRuntime
from fraud_detection.persistence import PersistenceStore
from fraud_detection.persistence.models import Prediction
from fraud_detection.persistence.recovery import recover_runtime

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = (
    "lightgbm_score",
    "xgboost_score",
    "ensemble_score",
    "calibrated_probability",
    "expected_cost_approve",
    "expected_cost_review",
    "expected_cost_block",
)


def test_frozen_api_parity_and_restart_recovery(tmp_path: Path, monkeypatch) -> None:
    """Replay predict then predict_batch exactly; compare the next direct score."""
    database_url = f"sqlite:///{(tmp_path / 'inference.sqlite').as_posix()}"
    monkeypatch.setenv("FRAUD_DATABASE_URL", database_url)
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    store = PersistenceStore.from_url(database_url)
    store.check_ready()
    rows = json.loads(
        pd.read_parquet(ROOT / "reports/runtime_golden_input_prefix.parquet")
        .head(4)
        .to_json(orient="records", double_precision=15)
    )
    direct = FrozenFraudInferenceRuntime.load(ROOT / "models/frozen_v1")
    expected_first = direct.predict_batch(rows[:1]).iloc[0].to_dict()
    expected_batch = direct.predict_batch(rows[1:3], include_rules=True).to_dict(orient="records")
    expected_next = (
        direct.predict_batch(rows[3:4], include_rules=True, update_state=False).iloc[0].to_dict()
    )
    del direct
    gc.collect()

    app = create_app(ApiSettings(ROOT / "models/frozen_v1"), store=store)
    with TestClient(app) as client:
        assert client.get("/ready").json()["replayed_transaction_count"] == 0
        first = client.post(
            "/predict", json={"transaction": rows[0]}, headers={"X-Idempotency-Key": "real-1"}
        )
        batch = client.post(
            "/predict/batch",
            json={"transactions": rows[1:3], "include_rules": True},
            headers={"X-Idempotency-Key": "real-batch"},
        )
        assert first.status_code == batch.status_code == 200
        observed = [first.json(), *batch.json()["predictions"]]
        for actual, expected in zip(observed, [expected_first, *expected_batch], strict=True):
            for name in NUMERIC:
                assert actual[name] == pytest.approx(expected[name], abs=1e-12, rel=0)
            assert actual["base_decision"] == expected["base_decision"]
            if "final_decision" in expected:
                assert actual["final_decision"] == expected["final_decision"]
                assert actual["triggered_rule_ids"] == (
                    expected["triggered_rule_ids"].split("|")
                    if expected["triggered_rule_ids"]
                    else []
                )
        replay = client.post(
            "/predict", json={"transaction": rows[0]}, headers={"X-Idempotency-Key": "real-1"}
        )
        assert replay.status_code == 200 and replay.json() == first.json()
    del app
    gc.collect()

    restored = FrozenFraudInferenceRuntime.load(ROOT / "models/frozen_v1")
    recovery = recover_runtime(restored, store)
    assert (recovery.request_count, recovery.transaction_count) == (2, 3)
    observed_next = (
        restored.predict_batch(rows[3:4], include_rules=True, update_state=False).iloc[0].to_dict()
    )
    for name in NUMERIC:
        assert observed_next[name] == pytest.approx(expected_next[name], abs=1e-12, rel=0)
    assert observed_next["base_decision"] == expected_next["base_decision"]
    assert observed_next["final_decision"] == expected_next["final_decision"]
    assert observed_next["triggered_rule_ids"] == expected_next["triggered_rule_ids"]
    with store.sessions() as session:
        assert session.scalar(select(func.count()).select_from(Prediction)) == 3
