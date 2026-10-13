# Productionization Phase 7 — FraudGuard Streamlit Dashboard

Status: **implemented and live-verified** on 2026-10-13. Dashboard traffic and labels shown here are local software tests, synthetic data, and development replay.

## Architecture and configuration

FraudGuard is a presentation and read-only operations layer. Streamlit calls the existing FastAPI service using `dashboard/client.py`; it does not load `models/frozen_v1`, invoke the inference runtime, or connect to PostgreSQL. The API reads persisted, monitoring, and feedback data through existing services. The only API addition is a bounded read-only recent-predictions endpoint described below.

Set `FRAUD_API_BASE_URL` to the FastAPI base URL (default `http://127.0.0.1:8000`). The API client uses a five-second timeout and returns sanitized error messages for connection and HTTP failures. Pages fetch data when opened and on rerun; **Refresh data** reruns the active view. The global sidebar checks `/ready` and `/version` on each app rerun so readiness is not presented from a long-lived cache. No background polling is enabled.

## Pages

| Page | Main content |
|---|---|
| Overview | Windowed transaction count, mean calibrated risk, mature-label coverage, decision rates, counts within the latest 100 persisted alerts, risk distribution, request count and latency percentiles, recent alerts, and feedback status. Missing values remain unavailable. |
| Transactions | Search by `TransactionID`, actual decision displayed separately from calibrated risk, expected costs, scores, bundle, rule identifiers, and a paginated/limited recent-predictions table with optional decision and time filters. |
| Monitoring | Backend input PSI / categorical divergence and sample counts for monitored inputs, score PSI, decision divergence and mix, entity novelty rates, request health/latency, and alerts ordered by severity. The dashboard does not recompute drift. |
| Feedback | Prediction, received/mature/immature label counts, mature coverage, backend performance metrics only when `status=available`, and observed-to-maturity delay summaries. `insufficient_labels` is explicit; unavailable metrics are not shown as zeros. |
| Simulation | Safe summaries from `reports/simulation_latest.json` and `reports/simulation_runs/*.json`; raw row results and label events are dropped by a fixed allowlist. Displays scenario, counts, throughput, replay lag, latency percentiles, labels, alerts, and bundle. Provides CLI instructions only. |
| System Health | `/health`, `/ready`, and `/version`: liveness, service/database readiness, recovery state and transaction count, bundle version, feature count, ensemble, calibration, policy, and rules state. |

The banner identifies the environment as **LOCAL / SIMULATED / DEVELOPMENT REPLAY** and states that the demonstration is not a live financial institution system. The health page records the fixed model information: 80% LightGBM + 20% XGBoost, Platt calibration, 467 features, Balanced expected-cost policy, and rules disabled by default. No model or policy controls are present.

## API endpoints

The dashboard consumes `GET /health`, `GET /ready`, `GET /version`, `GET /predictions/{transaction_id}`, `GET /predictions`, `GET /monitoring/summary`, `GET /monitoring/drift`, `GET /monitoring/alerts`, and `GET /feedback/summary`.

`GET /predictions` was added because the API previously offered individual lookup but no safe recent list. It supports `limit` (1–200), `offset`, timezone-aware `from` and `to` bounds, and an optional `decision` filter. The response includes only scalar prediction/audit fields and rule IDs; it never selects or returns `transactions.raw_payload` or feature vectors. Filters use the same stored effective decision (final decision when rules were enabled, otherwise base decision). Time windows are sent as UTC `from`/`to` bounds; monitoring continues to apply its existing `choose_window` semantics.

## Privacy and limitations

The API client contains no credentials or database configuration. The dashboard allowlists fields before displaying an individual transaction and simulation reports. It does not display fraud labels, raw transaction payloads, feature vectors, API stack traces, database URLs, passwords, or filesystem paths. Simulation summaries do not expose per-transaction results or label event contents. Authentication and authorization are required before exposing this UI to sensitive production traffic; this local demonstration has no authentication layer.

The simulator page observes CLI output and never starts simulator processes. It requires local API/PostgreSQL services to be running for live operational views. The current environment may report high latency or drift from prior test traffic; dashboard values describe the selected stored data and are not production banking evidence. Screenshots are optional and are not committed.

## Local run instructions

Run the API separately with one worker:

```powershell
$env:FRAUD_API_BASE_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

In another terminal, set the same `FRAUD_API_BASE_URL` and run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

The CLI remains separate from Streamlit. For options, run `.\.venv\Scripts\python.exe scripts\run_traffic_simulation.py --help`.

## Verification

Dashboard support tests cover health parsing, safe API failures, absent feedback metrics, risk/decision formatting, deterministic drift labels, alert ordering, transaction field allowlisting, malformed/sensitive simulation report handling, UTC window bounds, and frozen/reference/HOLDOUT artifact hashes. API tests cover recent prediction ordering, filters, bounds, and the absence of raw transaction fields.

Live verification used FastAPI and local PostgreSQL with one worker. `/ready` reported `ready=true`, `database_ready=true`, and `state_recovered=true`; the runtime replayed 11 successful requests and 50 transactions. `/predictions?limit=2` returned two audit-safe records. A persisted prediction lookup returned the approved audit fields without raw transaction payload or feature vector.

Streamlit AppTest rendered Overview, Transactions, Monitoring, Feedback, Simulation, and System Health without exceptions. Feedback visibly handled `insufficient_labels`; System Health showed database/service readiness and runtime recovery. The transaction page rendered both its recent list and a real stored lookup, with calibrated probability and business decision as separate metrics. The standalone Streamlit server answered HTTP 200 at its health endpoint and app entry point on `127.0.0.1:8501`.

Final verification: `pytest -q` passed **232 tests** with 8 existing scikit-learn deprecation warnings. `ruff check .` and `ruff format --check .` passed. `alembic current` reported **20261013_01 (head)**; `alembic check` reported **No new upgrade operations detected**. Frozen bundle components matched their seven manifest SHA-256 values; `models/monitoring_v1/reference.json` matched its existing SHA-256. The protected final HOLDOUT report, frozen predictions artifact, and prediction lock matched their prior SHA-256 values. Their contents were not opened for analysis; HOLDOUT labels were not accessed or evaluated. The dashboard and recent-predictions endpoint made no changes to inference, model, policy, rule, calibration, reference, or label-maturity semantics.

## Next phase

Docker and deployment are outside Phase 7. Authentication/authorization, production secret management, hosting, operational scaling, and a deployment review are still needed before any real sensitive production use.
