# Phase 8 — Docker deployment summary

Status: **completed** on 2026-10-13. The live stack was built and exercised with
Docker Desktop's `desktop-linux` context and a fresh named PostgreSQL volume.
Synthetic-only inference, persistence, idempotency, conflicts, monitoring,
delayed-label ingestion, dashboard rendering, simulator traffic, recovery, and
full Compose restart checks passed. The volume was retained during restart.

## Stack

`docker-compose.yml` defines four services on a private bridge network:

| Service | Image | Role |
|---|---|---|
| `db` | `postgres:17` | Independent PostgreSQL 17 database with `fraud_postgres_data` named volume. |
| `migrate` | `fraud-detection-api:local` | One-shot `alembic upgrade head` job. |
| `api` | `fraud-detection-api:local` | FastAPI, frozen inference bundle, monitoring reference, persistence and recovery. |
| `dashboard` | `fraud-detection-dashboard:local` | Read-only Streamlit client calling `http://api:8000`. |

The API image is built from `Dockerfile.api` on Python 3.11 slim Bookworm. Its
direct runtime dependencies are pinned in `requirements/api-runtime.txt`; it
copies `api/`, `src/`, `migrations/`, `models/frozen_v1/`, and
`models/monitoring_v1/`. It excludes training data and does not rebuild model
artifacts. The dashboard image uses `Dockerfile.dashboard` and the separately
pinned `requirements/dashboard-runtime.txt`; it has no database settings and
does not load models. Both application images run as UID 10001.

The API runtime includes pinned CatBoost and MLflow compatibility dependencies:
the application package import path loads those modules eagerly even though
the production inference route uses the frozen LightGBM/XGBoost bundle. These
dependencies were added after clean-container startup exposed missing imports;
no model, persistence, or inference behavior was changed. The direct runtime
package versions are pinned. The Python base image uses a
floating patch tag and pip may resolve transitive dependencies within upstream
constraints, so the build is not a bit-for-bit supply-chain lock.

## Ports and configuration

Defaults bind only to host loopback. This verification used local ignored
`.env` port overrides (PostgreSQL `15433`, API `18000`, dashboard `18501`)
because `5433`, `8000`, and `8501` were already occupied on the host. The
Compose database stayed isolated from host PostgreSQL:

| Host | Container | Service |
|---|---:|---|
| `127.0.0.1:5433` | `db:5432` | PostgreSQL |
| `127.0.0.1:8000` | `api:8000` | FastAPI |
| `127.0.0.1:8501` | `dashboard:8501` | Streamlit |

The PostgreSQL host mapping uses 5433 so it can coexist with Windows PostgreSQL
on 5432. Containers use `db:5432`. `POSTGRES_DB`, `POSTGRES_USER`, and
`POSTGRES_PASSWORD` configure only the Compose database. The API database URL is
assembled from those values and targets `db`; it does not inherit or replace the
host `FRAUD_DATABASE_URL`. Keep the password local in ignored `.env`, use a
strong URL-safe random value, and never reuse the host database password.

`.env.example` documents the variables and leaves the password blank.
Interpolation fails until `POSTGRES_PASSWORD` is set. `POSTGRES_HOST_PORT`,
`API_PORT`, and `DASHBOARD_PORT` configure host ports. The optional
`SIMULATION_REPORTS_DIR` bind mount is read-only and defaults to the empty,
tracked `docker_simulation_reports/` directory. Host-generated summaries appear
in the container only when placed in that directory (or another configured
directory); raw replay data is not mounted.

Create `.env` from `.env.example`, set the password, then run:

```powershell
docker compose build
docker compose up -d
docker compose ps
docker compose logs
```

Check `http://localhost:8000/health`, `http://localhost:8000/ready`, and
`http://localhost:8501`. `docker compose down` preserves the named database
volume. `docker compose down -v` deletes the containerized PostgreSQL data.
No cleanup command in this project removes that volume automatically.

## Startup and runtime safety

Startup dependencies are `db healthy` → `migrate completed successfully` →
`api healthy` → `dashboard`. Migration errors stop API startup. The database
uses `pg_isready`; the API checks `/ready` after the bundle has loaded, database
connectivity succeeds, and state recovery finishes; Streamlit uses
`/_stcore/health`.

The API command uses Uvicorn with **one worker**. The chronological inference
runtime keeps evolving non-label state in process memory. Multiple workers or
replicas can diverge and are unsafe until state is externalized or inference
semantics are redesigned. The simulator CLI remains a host-side tool and can
target `http://localhost:8000` without changing its inference semantics.

## Verification record

| Check | Result |
|---|---|
| Docker engine / build | Docker Engine 29.3.1, Docker Desktop, `desktop-linux`. API and dashboard images built successfully. Final image sizes: API 925,602,574 bytes (about 883 MiB); dashboard 186,436,018 bytes (about 178 MiB). |
| Fresh-volume start and health | Fresh named volume `fraud-detection-system_fraud_postgres_data`; PostgreSQL 17.11, migration job succeeded, API and dashboard became healthy. `/health` returned `ok`; `/ready` reported database and recovery ready, `frozen_v1`, 467 features. |
| Schema and migrations | `inference_requests`, `transactions`, `predictions`, `fraud_labels`, and `audit_events` all exist. `alembic current`: `20261013_01 (head)`. `alembic check`: no new upgrade operations. PostgreSQL `version()` reported 17.11 (Debian 17.11-1.pgdg13+2). |
| Synthetic single prediction | Persisted `/predict` for TransactionID `876543210` with `X-Idempotency-Key: phase8docker-smoke-01`; bundle `frozen_v1`, REVIEW. The request, transaction, and prediction persisted. No source labels were sent. |
| Idempotency and conflicts | Identical retry returned the stored prediction and request ID, with no rescore and no additional prediction. Changed payload with the same key returned 409. Reusing the TransactionID under a different key returned 409. Successful manual idempotency-key row count remained one. |
| Chronological batch and simulator | A host simulator run sent three synthetic rows in batches of 2 and 1, with labels disabled: 3 scored, 0 failed, 0 labels. Database checks showed each successful request's batch size matched the number of predictions attached to its sequence (including 2 rows / 2 predictions for the batch request), confirming atomic persistence and original boundaries. Simulator report: `reports/simulation_runs/68b1699f8e7344a4afd8045c3e2a490d.json`. |
| Delayed label and monitoring | One synthetic label was ingested as IMMATURE; `applied_to_model_state_at` stayed NULL and no fraud label was applied to model state. Feedback summary reported one immature, zero mature labels and no performance metrics. Three monitoring snapshots and four alerts persisted; monitoring endpoints returned data. |
| Dashboard | Streamlit health endpoint and `/`, `/overview`, `/transactions`, `/monitoring`, `/feedback`, `/simulation`, and `/health` returned HTTP 200. AppTest rendered all six pages without exceptions and displayed API-backed transaction, monitoring, feedback, and readiness values. |
| API restart recovery | API restart recovered the persisted transaction history and readiness. Full-stack restart recovered 5 transactions across 4 successful requests in 90,865.9 ms; `/ready` reported `database_ready=true`, `state_recovered=true`, and 5 replayed transactions. |
| Full Compose restart persistence | `docker compose down` followed by `docker compose up -d`, without `-v`, removed/recreated containers and network while retaining the named volume. Counts were unchanged across restart: 6 request records (4 successful), 5 transactions, 5 predictions, 1 immature label, 3 monitoring snapshots, 4 alerts. API, DB, and dashboard all healthy afterward. |
| Next-transaction parity | A fresh runtime replayed the earlier successful persisted requests and scored the same synthetic next row as the recovered API. LightGBM `0.02754373346665325`, XGBoost `0.1302175521850586`, ensemble `0.048078497210334324`, calibrated probability `0.021799799648417408`, decision REVIEW; all values exactly matched the persisted API result. Request sequence and 2+1 batch boundaries were preserved. |
| PostgreSQL timings | Initial prediction: DB lookup 3.92 ms, DB write 31.16 ms; model inference 27,127.08 ms. Identical retry: lookup 4.01 ms, inference 0 ms, write 0 ms. Full restart recovery: 90,865.9 ms for 4 successful requests / 5 transactions. |
| Final host gates | `pytest -q`: 235 passed, 8 existing scikit-learn deprecation warnings, 287.47 seconds. `ruff check .`: passed. `ruff format --check .`: passed (170 files). Compose config validation passed. |
| Frozen and monitoring artifacts | All `frozen_v1` artifact hashes matched the recorded baseline and `verify_bundle` passed. Both `monitoring_v1` hashes matched. |
| Protected HOLDOUT artifacts | Terminal report, frozen prediction CSV, and prediction-lock SHA-256 values matched their protected baselines. Files were hashed only; no HOLDOUT labels were parsed, exposed to inference, or evaluated. |
| Label isolation | `fraud_labels` exists and contains only the synthetic delayed-label test row. Inference and recovery replay only `inference_requests` and `transactions`; labels do not participate in scoring or recovery. |
| Environment isolation | Compose used its own local database credentials and volume. Host PostgreSQL and host `FRAUD_DATABASE_URL` were not modified. Local `.env` remains ignored and credentials were not printed. |
| ML/policy behavior | No model, feature, inference, persistence, or recovery architecture was changed. Only deployment runtime dependencies were added to satisfy observed import requirements. |

The current environment has no GitHub Actions workflow to extend, so no Docker
CI job was introduced. `docker compose config` is the available static Compose
gate here.

## Production deployment gaps

A real deployment needs managed PostgreSQL, secret-manager integration,
TLS/reverse proxy, authentication and authorization, centralized logging,
backups, production observability, and a design for horizontally scalable
state. The current API is intentionally single-worker and is not safe to
autoscale across processes.

Phase 8 verification is complete. Do not start delayed labels or Phase 9 work as
part of this deployment verification.
