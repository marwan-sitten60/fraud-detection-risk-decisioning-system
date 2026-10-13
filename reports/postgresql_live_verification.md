# Phase 3.5 — live PostgreSQL verification

Status: **completed** on 2026-10-13 (Africa/Cairo). `FRAUD_DATABASE_URL` was present; its value and credentials were never printed. The local server reported PostgreSQL **17.11**. Alembic reported `20261012_01 (head)`, and `alembic check` found no upgrade operations.

## Live results

| Check | Result |
| --- | --- |
| Schema | `inference_requests`, `transactions`, `predictions`, `fraud_labels`, and `audit_events` all present; initially empty. |
| Persisted `/predict` | HTTP 200 with `X-Idempotency-Key`; one successful request, transaction, and prediction committed. |
| Identical retry | HTTP 200 with the exact stored response. API timing logged `idempotent_replay: true`, `inference_ms: 0.0`, `db_write_ms: 0.0`; prediction count stayed one. One replay audit event was committed. |
| Same key, changed payload | HTTP 409; a failed-request audit was committed; no extra prediction. |
| Same TransactionID, other key | HTTP 409; a failed-request audit was committed; no extra prediction. |
| Chronological two-row batch | HTTP 200; one successful `predict_batch` request committed with two transactions and two predictions, row positions 0 and 1, rules enabled. This verified the complete committed batch and request boundary. |
| Restart/recovery | A new FastAPI app and frozen runtime replayed the two successful requests in sequence (single then two-row batch), three transactions total; no database rows were added during recovery. Failed requests and the idempotent retry were not replayed as scoring operations. |
| Next transaction parity | Continuous and recovered runtimes produced exactly equal values (maximum absolute difference **0.0**) for seven score/cost fields, with equal base/final decisions and triggered rule IDs. The comparison used the next chronological label-free input and did not persist it. |
| `/ready` | HTTP 200 before scoring and after restart; `database_ready: true`, `state_recovered: true`; replay count was 0 initially and 3 after restart. |
| `fraud_labels` | Table present and empty before and after. Inference, persistence replay queries, and recovery do not read it; the live requests and replay succeeded without labels. |

Final live row counts: `inference_requests=4` (2 successful, 2 rejected), `transactions=3`, `predictions=3`, `fraud_labels=0`, `audit_events=7` (2 received, 2 completed, 2 failed, 1 idempotent replay). Successful request sequences were 3 and 6; sequence gaps reflect rejected attempts and the rolled-back initial diagnostic request, not missing successful operations.

The initial live insert exposed a PostgreSQL foreign key ordering issue: the ORM attempted to flush a prediction before its transaction. The request returned 503 and the transaction rolled back with all five tables still empty. `src/fraud_detection/persistence/service.py` now flushes each transaction row before its prediction within the existing enclosing database transaction. The subsequent live run passed. This is a persistence ordering fix; model and persistence architecture were not changed.

## Local timing observations

These are single local measurements, not a latency benchmark or service-level objective. The first HTTP `/predict` took **30,072.03 ms** (validation 67.94 ms, PostgreSQL lookup 7.32 ms, inference 29,921.84 ms, write/commit 24.51 ms). The identical retry took **67.57 ms** (PostgreSQL lookup 6.52 ms, inference and write 0.0 ms). The two-row batch took **25,057.56 ms** (lookup 3.50 ms, inference 24,942.44 ms, write/commit 13.39 ms). A stored prediction HTTP lookup took **15.15 ms**. Runtime replay of two requests and three rows took **46,647.64 ms**; the recovered runtime's next read-only score took 23,856.95 ms. Measurements include local machine and test-client overhead where applicable.

## Gates and artifact integrity

`pytest -q`: **207 passed**, 8 existing scikit-learn deprecation warnings. `ruff check .`: **passed**. `ruff format --check .`: **passed**. `alembic current`: **20261012_01 (head)**. `alembic check`: **No new upgrade operations detected**. All nine `models/frozen_v1` SHA-256 values matched the before-work baseline. The terminal HOLDOUT report, prediction lock, and label-free frozen prediction CSV remained unchanged; the CSV hash still matches its lock (`0d7f5d117436cd7227d682388c5cceddbc72b689143984a7d186d9ec03c1391f`). No HOLDOUT labels were accessed. Delayed labels and later phases were not started.
