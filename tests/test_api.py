"""Small label-free HTTP tests; the frozen bundle is tested separately."""

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

from fraud_detection.inference.contracts import validate_transactions

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
            amount_engineer=SimpleNamespace(
                history_=pd.DataFrame({"TransactionID": [0], "TransactionDT": [0]})
            )
        )
        self.active = 0
        self.max_active = 0
        self.entered = threading.Event()
        self.fail_once = False

    def snapshot_state(self) -> pd.DataFrame:
        return self.feature_state.amount_engineer.history_.copy(deep=True)

    def restore_state(self, snapshot: pd.DataFrame) -> None:
        self.feature_state.amount_engineer.history_ = snapshot.copy(deep=True)

    def predict_batch(self, rows: list[dict], *, include_rules: bool = False) -> pd.DataFrame:
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


def app_for(runtime: StubRuntime, *, maximum: int = 3, rules: bool = False):
    calls = []

    def loader(path: Path) -> StubRuntime:
        calls.append(path)
        return runtime

    app = create_app(
        ApiSettings(Path("unused"), enable_rules=rules, max_batch_size=maximum), loader=loader
    )
    return app, calls


def test_health_ready_version_and_load_once() -> None:
    app, calls = app_for(StubRuntime())
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {
            "ready": True,
            "bundle_version": "frozen_v1",
            "feature_count": 467,
            "rules_enabled": False,
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

    broken = create_app(ApiSettings(Path("unused")), loader=broken_loader)
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
