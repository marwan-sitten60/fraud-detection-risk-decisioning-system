"""Rebuild mutable non-label runtime history from durable request boundaries."""

import time
from dataclasses import dataclass

from fraud_detection.inference import FrozenFraudInferenceRuntime
from fraud_detection.persistence.database import PersistenceStore


@dataclass(frozen=True)
class RecoveryResult:
    request_count: int
    transaction_count: int
    duration_ms: float
    last_key: tuple[float, int] | None


def recover_runtime(
    runtime: FrozenFraudInferenceRuntime, store: PersistenceStore
) -> RecoveryResult:
    """Replay only successful current-bundle requests, without DB writes or labels."""
    started = time.perf_counter()
    history = runtime.feature_state.amount_engineer.history_
    last_key = (
        (float(history.TransactionDT.iloc[-1]), int(history.TransactionID.iloc[-1]))
        if history is not None and not history.empty
        else None
    )
    request_count = transaction_count = 0
    for request, rows in store.replay_stream(runtime.manifest["bundle_version"]):
        first_key = (float(rows[0]["TransactionDT"]), int(rows[0]["TransactionID"]))
        if last_key is not None and first_key <= last_key:
            raise RuntimeError("Recovery stream is not chronological")
        # predict_one delegates to predict_batch for one row in the frozen runtime.
        result = runtime.predict_batch(rows, include_rules=request.rules_enabled)
        if result.TransactionID.tolist() != [int(row["TransactionID"]) for row in rows]:
            raise RuntimeError("Recovery replay returned unexpected transaction IDs")
        last_key = (float(rows[-1]["TransactionDT"]), int(rows[-1]["TransactionID"]))
        request_count += 1
        transaction_count += len(rows)
    return RecoveryResult(
        request_count, transaction_count, (time.perf_counter() - started) * 1000, last_key
    )
