# PostgreSQL persistence, idempotency, audit, and runtime recovery — Phase 3

## Purpose and architecture

Client → FastAPI → persistence/idempotency gate → frozen runtime → PostgreSQL audit and prediction store. The database is the durable record of successful inference activity. It stores the complete label-free 433-column raw transaction payload for replay, but no 467-feature vector and no transaction label. The bundle, feature state, models, calibrator, costs, rules, and rules-disabled primary default are unchanged.

On restart: `frozen_v1` TRAIN-derived state + successful stored request stream, in durable request-sequence order and original request boundaries → replay → recovered mutable runtime state. The service becomes ready only after bundle verification, database/schema checks, and successful replay.

## Database and migration

The implementation uses SQLAlchemy 2.x, Alembic revision `20261012_01`, and psycopg 3 for PostgreSQL URLs. JSON payload columns use PostgreSQL JSONB there and portable JSON in SQLite tests. Configure credentials only through `FRAUD_DATABASE_URL`; none are present in the repository. Alembic does not create a database and has no destructive downgrade.

| Table | Purpose and key constraints/indexes |
| --- | --- |
| `inference_requests` | Durable integer request sequence, UUID, correlation ID, unique nullable idempotency key, endpoint, batch size, rules flag, bundle version, status, timestamps, latency, payload hash, optional error category. Indexed by correlation ID and `(bundle_version, sequence)`. |
| `transactions` | Unique `TransactionID`, chronological fields, canonical label-free raw JSON, SHA-256 hash, first request sequence, row position, timestamp. Indexed by `(transaction_dt, transaction_id)`. |
| `predictions` | One prediction per transaction, request FK, row position, component/ensemble/calibrated scores, decisions, three costs, rule IDs, bundle version, feature count, timestamp. Unique `(request_sequence, row_position)`; indexed by bundle version. |
| `fraud_labels` | Future delayed-label storage: `TransactionID`, fraud flag, source, observed/matured/ingested times, nullable applied-to-state time, metadata. Indexed by maturity. This table is **not read or used by inference or recovery**. |
| `audit_events` | Application-append-only `request_received`, `prediction_completed`, and `idempotent_replay` events with compact metadata and request FK. Indexed by request/event. |

Foreign keys link each transaction and prediction to its original request and each prediction to its transaction. Successful batch rows, prediction rows, and audit events commit in one database transaction or all roll back. Valid payloads rejected for an idempotency conflict, duplicate ID, chronology error, or inference error receive a separate `failed` request and `prediction_failed` audit event when the database is available. Schema-rejected requests and failures during a database outage are returned/logged but cannot be durably recorded in the unavailable database.

## Idempotency and transaction safety

`X-Idempotency-Key` is optional and restricted to a safe 1–128 character token. A new key scores and stores once. The same key with the same canonical ordered scoring payload returns the stored response without touching runtime history. The same key with changed payload, rules mode, or endpoint returns 409. A successfully scored `TransactionID` with no matching idempotent replay returns 409 even if a new key is supplied.

Canonical SHA-256 scoring hashes include endpoint type, effective rules setting, and the ordered complete batch. They exclude correlation IDs, generated request IDs, timestamps, and labels. Canonical JSON sorts object keys, normalizes numpy scalars and missing values to standard JSON values/null, and preserves batch order. Target fields are rejected before persistence.

Under the shared async inference lock, the service validates the frozen raw input contract, checks idempotency and duplicate IDs, snapshots mutable history, runs the existing frozen runtime, and writes request/transaction/prediction/audit records. A database write or commit failure rolls back the database transaction and restores the runtime snapshot. The service then stops accepting mutations and reports 503 until restart/recovery. `/health` remains a lightweight liveness check and does not query the database.

## Recovery and API behavior

At startup the application loads and verifies the immutable bundle, checks database connectivity and Alembic revision, then replays only successful requests for the loaded bundle version in stored sequence order. Each request is replayed using its original single or batch boundary and original row order; replay makes no database writes and never consults `fraud_labels`. Failed initialization or replay leaves `/ready` at 503. `/ready` reports safe `database_ready`, `state_recovered`, and replayed-transaction-count fields once ready, along with bundle and rules metadata. It rechecks database connectivity; if the database fails later, prediction writes stop with 503 and restart is needed before further scoring.

`GET /predictions/{transaction_id}` returns stored scores, decisions, costs, rule IDs, bundle version, request UUID, timestamp, and rules flag. It omits raw payload and feature vectors. Existing `/health`, `/ready`, `/version`, `/predict`, and `/predict/batch` remain. Prediction logs include safe correlation, action-count, and timing fields without raw transaction columns.

PostgreSQL durability does **not** coordinate in-memory histories across processes. The Phase-3 serving mode remains `--workers 1`. Multi-worker state coordination is a later architecture concern.

## Local configuration

From the repository root in PowerShell:

```powershell
$env:FRAUD_DATABASE_URL="postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME"
.venv\Scripts\python.exe -m alembic upgrade head
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Replace the URL with actual deployment credentials supplied outside version control. Migrations must run before starting the API. `FRAUD_BUNDLE_PATH`, `FRAUD_ENABLE_RULES=false`, and `FRAUD_MAX_BATCH_SIZE` retain their serving meanings. The `X-Idempotency-Key` header is recommended for all retryable prediction requests.

## Verification and performance

SQLite-backed tests exercise the same SQLAlchemy models and Alembic migration without requiring PostgreSQL or the full IEEE dataset. Focused tests cover canonical hashes, idempotency/conflicts, unique transaction IDs, single/batch persistence, numeric/rule round-trips, database-failure rollback, recovery boundaries, no duplicate recovery writes, readiness, label isolation, safe lookup, and log privacy. A separate four-row label-free test used the actual frozen bundle: direct runtime and API predictions matched within `1e-12` for all scores and costs, decisions and rule IDs matched exactly, and a fresh runtime replayed one single and one two-row batch before matching the next transaction. Replay created no prediction rows.

One local **SQLite service-layer** request using the frozen bundle measured approximately 42.11 ms input validation/canonicalization, 8.81 ms database lookup, 19,759.84 ms inference (19,754.35 ms feature generation), 17.04 ms database write/commit, and 19,846.84 ms total. Snapshot and adapter overhead accounts for the remaining time. These values are in `reports/postgresql_local_benchmark.json`; they are not PostgreSQL timings or a production SLA.

`FRAUD_DATABASE_URL` was absent in this environment. **Live PostgreSQL integration was not executed because no configured PostgreSQL instance was available.** The psycopg 3 driver imported successfully; the ORM and migration were verified against SQLite. A real PostgreSQL migration, insert, idempotent retry, lookup, and restart recovery smoke test remain necessary before deployment.

## Privacy, limits, and scope

Raw transaction JSON is necessary for deterministic replay and belongs behind database access controls. A real deployment needs encryption at rest, restricted access, a retention policy, and a data-minimization review. Database records and logs contain no `isFraud` from scoring requests. Delayed labels remain stored only in the future-feedback table and are not applied to target encoding or model state.

The service still has process-local mutable history and one-worker ordering. This phase did not add PostgreSQL hosting, Docker, dashboards, monitoring, deployment, feedback application, or retraining.

## Final checks

`pytest -q`: **207 passed** (8 existing scikit-learn deprecation warnings). `ruff check .`: **passed**. `ruff format --check .`: **passed** (128 files). The initial migration applied to a fresh SQLite database, and `alembic check` found no ORM/migration schema drift. SHA-256 comparison confirmed all nine `models/frozen_v1` files and the three protected terminal HOLDOUT artifacts exactly matched their before-work hashes. No HOLDOUT labels were accessed.
