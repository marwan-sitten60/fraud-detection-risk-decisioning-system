"""Small label-free HTTP tests; the frozen bundle is tested separately."""

import logging
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from api.dependencies import ApiSettings
from api.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

from fraud_detection.inference.contracts import validate_transactions
from fraud_detection.persistence import PersistenceStore
from fraud_detection.persistence.canonical import payload_hash
from fraud_detection.persistence.database import REVISION
from fraud_detection.persistence.models import (
    Base,
    FraudLabel,
    InferenceRequest,
    Prediction,
    Transaction,
)
from fraud_detection.persistence.recovery import recover_runtime
from fraud_detection.persistence.service import PersistenceService

RAW_COLUMNS = (
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "addr1",
    "P_emaildomain",
)


def transaction(identifier: int, timestamp: int, amount: float = 100.0) -> dict:
    return {
        "TransactionID": identifier,
        "TransactionDT": timestamp,
        "TransactionAmt": amount,
        "card1": 10,
        "addr1": 20,
        "P_emaildomain": "a.test",
    }


class StubRuntime:
    def __init__(self) -> None:
        self.manifest = {"bundle_version": "frozen_v1", "feature_count": 467}
        self.feature_state = SimpleNamespace(
            raw_columns=RAW_COLUMNS,
            preprocessor=SimpleNamespace(categorical_features_=("P_emaildomain",)),
            amount_engineer=SimpleNamespace(
                history_=pd.DataFrame({"TransactionID": [0], "TransactionDT": [0]})
            ),
        )
        self.active = 0
        self.max_active = 0
        self.entered = threading.Event()
        self.fail_once = False
        self.batch_sizes = []

    def snapshot_state(self) -> pd.DataFrame:
        return self.feature_state.amount_engineer.history_.copy(deep=True)

    def restore_state(self, snapshot: pd.DataFrame) -> None:
        self.feature_state.amount_engineer.history_ = snapshot.copy(deep=True)

    def predict_batch(self, rows: list[dict], *, include_rules: bool = False) -> pd.DataFrame:
        self.batch_sizes.append(len(rows))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            frame = validate_transactions(rows, RAW_COLUMNS, ("card1", "addr1"))
            time.sleep(0.03)
            history = self.feature_state.amount_engineer.history_
            self.feature_state.amount_engineer.history_ = pd.concat(
                [history, frame.loc[:, ["TransactionID", "TransactionDT"]]], ignore_index=True
            )
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("internal detail must stay private")
            lightgbm = frame.TransactionAmt.to_numpy(dtype=float) / 1000
            xgboost = frame.TransactionAmt.to_numpy(dtype=float) / 2000
            ensemble = 0.8 * lightgbm + 0.2 * xgboost
            probability = ensemble / 2
            output = pd.DataFrame(
                {
                    "TransactionID": frame.TransactionID.to_numpy(),
                    "lightgbm_score": lightgbm,
                    "xgboost_score": xgboost,
                    "ensemble_score": ensemble,
                    "calibrated_probability": probability,
                    "base_decision": "APPROVE",
                    "expected_cost_approve": probability * frame.TransactionAmt.to_numpy(),
                    "expected_cost_review": 3.0,
                    "expected_cost_block": 25.5,
                    "bundle_version": "frozen_v1",
                }
            )
            if include_rules:
                triggered = frame.TransactionAmt.ge(200)
                output["final_decision"] = triggered.map({True: "REVIEW", False: "APPROVE"})
                output["triggered_rule_ids"] = triggered.map({True: "R02", False: ""})
            return output
        finally:
            self.active -= 1


def app_for(
    runtime: StubRuntime,
    *,
    maximum: int = 3,
    rules: bool = False,
    store: PersistenceStore | None = None,
):
    calls = []

    def loader(path: Path) -> StubRuntime:
        calls.append(path)
        return runtime

    app = create_app(
        ApiSettings(Path("unused"), enable_rules=rules, max_batch_size=maximum),
        loader=loader,
        store=store or make_test_store(),
    )
    return app, calls


def make_test_store() -> PersistenceStore:
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        connection.execute(
            text("INSERT INTO alembic_version VALUES (:revision)"), {"revision": REVISION}
        )
    return PersistenceStore(engine)


def test_health_ready_version_and_load_once() -> None:
    app, calls = app_for(StubRuntime())
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {
            "ready": True,
            "bundle_version": "frozen_v1",
            "feature_count": 467,
            "rules_enabled": False,
            "database_ready": True,
            "state_recovered": True,
            "replayed_transaction_count": 0,
        }
        assert client.get("/version").json()["bundle_version"] == "frozen_v1"
        first = client.post("/predict", json={"transaction": transaction(1, 10)})
        assert first.status_code == 200
        assert first.headers["X-Request-ID"]
        assert first.json()["calibrated_probability"] == pytest.approx(0.045)
        assert first.json()["base_decision"] == "APPROVE"
        assert "final_decision" not in first.json()
        batch = client.post(
            "/predict/batch",
            json={"transactions": [transaction(2, 11), transaction(3, 12)]},
        )
        assert batch.status_code == 200 and batch.json()["count"] == 2
        assert len(calls) == 1


def test_schema_and_chronology_errors_are_clear_and_sanitized() -> None:
    app, _ = app_for(StubRuntime(), maximum=2)
    with TestClient(app) as client:
        assert (
            client.post(
                "/predict", json={"transaction": {**transaction(1, 10), "isFraud": 1}}
            ).status_code
            == 422
        )
        assert (
            client.post("/predict", json={"transaction": {"TransactionID": 1}}).status_code == 422
        )
        assert (
            client.post(
                "/predict", json={"transaction": transaction(1, 10, amount="bad")}
            ).status_code
            == 422
        )
        assert client.post("/predict/batch", json={"transactions": []}).status_code == 422
        assert (
            client.post(
                "/predict/batch",
                json={"transactions": [transaction(1, 10), transaction(2, 11), transaction(3, 12)]},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/predict/batch",
                json={"transactions": [transaction(2, 11), transaction(1, 10)]},
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/predict/batch",
                json={"transactions": [transaction(1, 10), transaction(1, 11)]},
            ).status_code
            == 400
        )
        assert client.post("/predict", json={"transaction": transaction(2, 11)}).status_code == 200
        wrong = client.post("/predict", json={"transaction": transaction(1, 11)})
        assert wrong.status_code == 400
        assert "Traceback" not in wrong.text and "internal detail" not in wrong.text


def test_optional_rules_and_direct_runtime_parity() -> None:
    runtime = StubRuntime()
    direct = StubRuntime().predict_batch([transaction(1, 10, 200)], include_rules=True).iloc[0]
    app, _ = app_for(runtime)
    payload = {"transaction": transaction(1, 10, 200), "include_rules": True}
    original = deepcopy(payload)
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert payload == original
        for column in (
            "lightgbm_score",
            "xgboost_score",
            "ensemble_score",
            "calibrated_probability",
            "base_decision",
            "final_decision",
        ):
            assert body[column] == direct[column]
        assert body["triggered_rule_ids"] == ["R02"]
        assert body["calibrated_probability"] == pytest.approx(0.09)
        second = client.post("/predict", json={"transaction": transaction(2, 11, 200)})
        assert second.status_code == 200
        assert second.json()["calibrated_probability"] == body["calibrated_probability"]
        assert "final_decision" not in second.json()


def test_failed_load_is_not_ready_and_unexpected_failure_rolls_back() -> None:
    def broken_loader(path: Path) -> StubRuntime:
        raise RuntimeError("private bundle path")

    broken = create_app(ApiSettings(Path("unused")), loader=broken_loader, store=make_test_store())
    with TestClient(broken) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"transaction": transaction(1, 10)}).status_code == 503
    runtime = StubRuntime()
    runtime.fail_once = True
    app, _ = app_for(runtime)
    with TestClient(app) as client:
        failure = client.post("/predict", json={"transaction": transaction(1, 10)})
        assert failure.status_code == 500
        assert "internal detail" not in failure.text and "Traceback" not in failure.text
        retry = client.post("/predict", json={"transaction": transaction(1, 10)})
        assert retry.status_code == 200


def test_concurrent_stateful_requests_do_not_interleave() -> None:
    runtime = StubRuntime()
    app, _ = app_for(runtime)
    with TestClient(app) as client:
        results = []

        def first() -> None:
            results.append(client.post("/predict", json={"transaction": transaction(1, 10)}))

        def second() -> None:
            results.append(client.post("/predict", json={"transaction": transaction(2, 11)}))

        first_thread = threading.Thread(target=first)
        second_thread = threading.Thread(target=second)
        first_thread.start()
        assert runtime.entered.wait(timeout=5)
        second_thread.start()
        first_thread.join(timeout=5)
        second_thread.join(timeout=5)
        assert sorted(response.status_code for response in results) == [200, 200]
        assert runtime.max_active == 1


def test_two_clean_apps_return_the_same_result() -> None:
    outputs = []
    for _ in range(2):
        app, _ = app_for(StubRuntime())
        with TestClient(app) as client:
            outputs.append(client.post("/predict", json={"transaction": transaction(1, 10)}).json())
    assert outputs[0] == outputs[1]


def test_canonical_hash_and_idempotent_retry_without_rescoring() -> None:
    import numpy as np

    assert payload_hash({"b": np.float64(np.nan), "a": np.int64(7)}) == payload_hash(
        {"a": 7, "b": None}
    )
    assert payload_hash([{"a": 1}, {"a": 2}]) != payload_hash([{"a": 2}, {"a": 1}])
    runtime = StubRuntime()
    store = make_test_store()
    app, _ = app_for(runtime, store=store)
    headers = {"X-Idempotency-Key": "retry-1"}
    with TestClient(app) as client:
        first = client.post("/predict", json={"transaction": transaction(1, 10)}, headers=headers)
        repeated = client.post(
            "/predict", json={"transaction": transaction(1, 10)}, headers=headers
        )
        assert first.status_code == repeated.status_code == 200
        assert first.json() == repeated.json()
        assert runtime.batch_sizes == [1]
        changed = client.post(
            "/predict", json={"transaction": transaction(1, 10, 200)}, headers=headers
        )
        assert changed.status_code == 409
        unrelated = client.post("/predict", json={"transaction": transaction(1, 10)})
        assert unrelated.status_code == 409
        lookup = client.get("/predictions/1")
        assert lookup.status_code == 200
        assert lookup.json()["calibrated_probability"] == first.json()["calibrated_probability"]
        assert "raw_payload" not in lookup.text and "P_emaildomain" not in lookup.text
        assert client.get("/predictions/999").status_code == 404
    with store.sessions() as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(InferenceRequest)
                .where(InferenceRequest.status == "success")
            )
            == 1
        )
        assert (
            session.scalar(
                select(func.count())
                .select_from(InferenceRequest)
                .where(InferenceRequest.status == "failed")
            )
            == 2
        )
        assert session.scalar(select(func.count()).select_from(Transaction)) == 1
        assert session.scalar(select(func.count()).select_from(Prediction)) == 1
        assert "isFraud" not in session.get(Transaction, 1).raw_payload


def test_batch_persists_atomically_and_rules_round_trip() -> None:
    store = make_test_store()
    app, _ = app_for(StubRuntime(), store=store)
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={
                "transactions": [transaction(1, 10, 100), transaction(2, 11, 200)],
                "include_rules": True,
            },
            headers={"X-Idempotency-Key": "batch-1"},
        )
        assert response.status_code == 200
        assert response.json()["count"] == 2
        assert response.json()["predictions"][1]["triggered_rule_ids"] == ["R02"]
        retry = client.post(
            "/predict/batch",
            json={
                "transactions": [transaction(1, 10, 100), transaction(2, 11, 200)],
                "include_rules": True,
            },
            headers={"X-Idempotency-Key": "batch-1"},
        )
        assert retry.json() == response.json()
        stored = client.get("/predictions/2")
        assert stored.status_code == 200
        assert stored.json()["triggered_rule_ids"] == ["R02"]
        assert (
            stored.json()["expected_cost_block"]
            == response.json()["predictions"][1]["expected_cost_block"]
        )
    with store.sessions() as session:
        assert session.scalar(select(func.count()).select_from(Transaction)) == 2
        assert session.scalar(select(func.count()).select_from(Prediction)) == 2


@pytest.mark.parametrize("rows", [[transaction(1, 10)], [transaction(1, 10), transaction(2, 11)]])
def test_database_write_failure_rolls_back_runtime_and_batch(rows: list[dict]) -> None:
    store = make_test_store()
    runtime = StubRuntime()
    app, _ = app_for(runtime, store=store)

    def fail_insert(connection, cursor, statement, parameters, context, executemany) -> None:
        if "INSERT INTO inference_requests" in statement:
            raise OperationalError(statement, parameters, Exception("unavailable"))

    with TestClient(app) as client:
        event.listen(store.engine, "before_cursor_execute", fail_insert)
        endpoint = "/predict" if len(rows) == 1 else "/predict/batch"
        body = {"transaction": rows[0]} if len(rows) == 1 else {"transactions": rows}
        response = client.post(endpoint, json=body)
        event.remove(store.engine, "before_cursor_execute", fail_insert)
        assert response.status_code == 503
        assert runtime.feature_state.amount_engineer.history_.TransactionID.tolist() == [0]
        assert client.get("/ready").status_code == 503
        assert client.get("/health").status_code == 200
    with store.sessions() as session:
        assert session.scalar(select(func.count()).select_from(Transaction)) == 0
        assert session.scalar(select(func.count()).select_from(Prediction)) == 0


def test_recovery_preserves_request_boundaries_and_next_score() -> None:
    store = make_test_store()
    runtime_a = StubRuntime()
    service = PersistenceService(runtime_a, store, (0.0, 0))
    for kind, rows in (
        ("predict", [transaction(1, 10)]),
        ("predict_batch", [transaction(2, 11), transaction(3, 12), transaction(4, 13)]),
        ("predict", [transaction(5, 14)]),
    ):
        service.score(kind, rows, rules_enabled=False, idempotency_key=None, correlation_id="test")
    runtime_b = StubRuntime()
    recovery = recover_runtime(runtime_b, store)
    assert (recovery.request_count, recovery.transaction_count) == (3, 5)
    assert runtime_b.batch_sizes == [1, 3, 1]
    assert runtime_a.feature_state.amount_engineer.history_.equals(
        runtime_b.feature_state.amount_engineer.history_
    )
    next_row = transaction(6, 15)
    assert runtime_a.predict_batch([next_row]).equals(runtime_b.predict_batch([next_row]))
    with store.sessions() as session:
        assert session.scalar(select(func.count()).select_from(Prediction)) == 5
        assert session.scalar(select(func.count()).select_from(FraudLabel)) == 0


def test_unconfigured_database_keeps_service_not_ready() -> None:
    app = create_app(ApiSettings(Path("unused")), loader=lambda _: StubRuntime())
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"transaction": transaction(1, 10)}).status_code == 503


def test_incompatible_schema_keeps_service_not_ready() -> None:
    store = make_test_store()
    with store.engine.begin() as connection:
        connection.execute(text("UPDATE alembic_version SET version_num='wrong_revision'"))
    app, _ = app_for(StubRuntime(), store=store)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


def test_prediction_log_excludes_raw_payload() -> None:
    from api.main import LOGGER

    messages: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = Capture()
    LOGGER.addHandler(handler)
    try:
        app, _ = app_for(StubRuntime())
        with TestClient(app) as client:
            raw = transaction(1, 10)
            raw["P_emaildomain"] = "private-marker.example"
            assert client.post("/predict", json={"transaction": raw}).status_code == 200
    finally:
        LOGGER.removeHandler(handler)
    assert any("db_lookup_ms" in message for message in messages)
    assert all("private-marker.example" not in message for message in messages)
