# FastAPI serving layer — Phase 2

## Scope and architecture

`api.main:app` exposes the existing `FrozenFraudInferenceRuntime` from `src/fraud_detection/inference/`. The request path is HTTP validation → the loaded frozen runtime → component and 0.8/0.2 ensemble scores → frozen Platt calibration → Balanced expected-cost decision → JSON. The frozen rules engine is an optional overlay and is disabled by default. The API contains no copied feature, model, calibration, cost, or rule implementation.

The service is a local software-serving demonstration. It does not claim a production latency or availability target.

## Startup and configuration

FastAPI lifespan loads `models/frozen_v1` **once per application process** using `FrozenFraudInferenceRuntime.load`. That loader verifies the bundle manifest and artifact checksums before deserializing its fitted state. `/ready` returns 200 only after the load completes; a failed load leaves predictions and `/ready` at 503, without a fallback model. `/health` remains a lightweight process-liveness response. The loaded runtime and one shared `asyncio.Lock` live on application state.

Operational environment settings are `FRAUD_BUNDLE_PATH` (default `models/frozen_v1`), `FRAUD_ENABLE_RULES` (default `false`), and `FRAUD_MAX_BATCH_SIZE` (default `256`). They cannot change model parameters, features, ensemble weights, calibration, costs, or the frozen rules. `include_rules` in a prediction request can explicitly select the existing optional overlay for that request.

From the repository root, run:

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

The recommended Phase-2 mode has **one application worker**. Each worker would otherwise load a separate mutable history and produce divergent stateful results. The state is process-local and resets to the bundled TRAIN-derived history on restart. Durable history, distributed ordering, and multi-worker operation require a later persistence design.

## Endpoints and contracts

| Endpoint | Result |
| --- | --- |
| `GET /health` | `{"status":"ok"}` without model inference |
| `GET /ready` | Readiness, bundle version, 467-feature count, rules default |
| `GET /version` | Service/bundle versions and frozen ensemble, calibration, and policy identifiers |
| `POST /predict` | One label-free raw transaction |
| `POST /predict/batch` | Non-empty chronological list, capped by `FRAUD_MAX_BATCH_SIZE` |

OpenAPI is available at `/docs` and `/openapi.json`. The request envelope is `{"transaction": {…}, "include_rules": false}` or `{"transactions": [{…}], "include_rules": false}`. Each raw transaction must contain the exact 433 columns from the bundle's `feature_contract.json`. The API delegates field, type, label, duplicate-ID, and within-batch order checks to the existing frozen input contract; it does not maintain a second 433-field list. `isFraud` and other target fields are forbidden. Requests must advance by `(TransactionDT, TransactionID)` across calls. Rows are never reordered silently.

Each prediction returns `TransactionID`, LightGBM and XGBoost scores, ensemble score, calibrated probability, `base_decision`, three expected costs, and bundle version. With rules enabled, `final_decision` and `triggered_rule_ids` are also returned. Pydantic enforces finite probabilities in `[0,1]`, finite nonnegative costs, and `APPROVE`/`REVIEW`/`BLOCK` decisions. No feature vector is returned.

Errors use `{"error":{"code":"…","message":"…"},"request_id":"…"}`: 400 for semantic ordering/duplicate-ID errors, 422 for schema/field/type errors, 503 for an unavailable runtime, and 500 for unexpected inference failures. Details and tracebacks stay in server logs. Prediction logs contain a correlation ID, endpoint, count, elapsed time, bundle version, action counts, and status, without raw transactions, labels, or feature vectors. An acceptable incoming `X-Request-ID` is echoed; otherwise the service creates one.

Examples, with `request.json` containing a complete 433-field label-free `transaction` envelope and `batch.json` containing a complete chronological `transactions` envelope:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@request.json"
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict/batch -ContentType application/json -InFile batch.json
```

## Stateful execution

Both prediction routes acquire the same async lock before checking cross-request chronology, taking a state snapshot, and running inference. The lock prevents concurrent updates from interleaving. If inference fails, the pre-request history is restored and the request may be retried. `/health`, `/ready`, and `/version` do not acquire the inference lock. The runtime has no externally persisted transaction stream; callers must deliver chronological transactions to this single process.

## Local verification

The local Uvicorn smoke test used the first six chronological **label-free** rows of `reports/runtime_golden_input_prefix.parquet`. `/health`, `/ready`, `/version`, `/openapi.json`, the first `/predict`, a warmed `/predict`, and a four-row `/predict/batch` all returned HTTP 200. `/ready` reported `frozen_v1`, 467 features, and rules disabled. The optional-rule batch was exercised separately from the default-disabled single calls.

An independently loaded direct `FrozenFraudInferenceRuntime` scored the same six JSON-decoded rows in the same 1 + 1 + 4 sequence. HTTP and direct output had **zero maximum absolute difference** for LightGBM, XGBoost, ensemble, calibrated probability, and all three expected costs. Transaction IDs, base and optional final decisions, and triggered rule IDs matched exactly. The result is recorded in `reports/fastapi_local_benchmark.json`.

On this Windows local run, the first request after startup took 19.65 s, a warmed single request took 18.91 s, and a four-row batch took 19.10 s, approximately 0.21 rows/s for that small batch. The server process after scoring showed about 741 MB working set and 1.51 GB private bytes. These are local observations, not a service-level target. The runtime's feature generation and state snapshots dominate small-request latency; no frozen feature semantics were changed to improve the measurement.

Focused small-fixture tests cover liveness/readiness/version, startup load once, request and response validation, ordering, rules, error sanitation and rollback, independent-process determinism, direct adapter parity, and non-interleaving concurrent calls. The full repository test and Ruff results are recorded below after final verification.

## Integrity and limits

The frozen bundle is read-only in this phase. Its nine files were SHA-256 checked before and after work, and the bundle manifest verifier passed. The terminal HOLDOUT report, locked prediction CSV, and prediction lock were not overwritten; no HOLDOUT labels were accessed. No model, feature, calibrator, ensemble, business-cost, decision, or rule semantics changed.

Current limits are process-local history, single-worker serving, chronological caller responsibility, and substantial per-request memory and latency for small payloads. A later database/state-persistence phase can address durable history and controlled scaling; it was not started here.

## Final checks

`pytest -q`: **198 passed**, 8 existing library deprecation warnings. `ruff check .`: **passed**. `ruff format --check .`: **passed** (118 files formatted). All nine bundle-file SHA-256 hashes and all three protected terminal-artifact hashes matched the before-work baseline; `verify_bundle` passed after implementation.
