# Phase 6 — Traffic Simulator, Chronological Replay, and Delayed Labels

Status: **implemented and live-verified** on 2026-10-13 (Africa/Cairo). This is software simulation. Validation replay measurements, synthetic labels, scores, and alerts are not new model-selection or performance evidence.

## Purpose and architecture

The reusable simulator is under `src/fraud_detection/simulation/`, with CLI `scripts/run_traffic_simulation.py`. It sends one sequential stream of HTTP requests to the existing FastAPI service: readiness/version checks, `/predict/batch`, delayed `/labels`, monitoring snapshots, and final monitoring/feedback queries. It does not call the inference runtime directly, write predictions to PostgreSQL, apply labels to model state, or change the frozen model or policy. Requests and rows remain chronological; source timestamps are kept in model `TransactionDT`, while a separate accelerated simulator clock schedules sends and delayed-label events.

Every run gets a UUID simulation ID. Transaction IDs are remapped into a high, run-specific integer range; the source ID and label mapping stay private in memory. Prediction requests carry distinct `X-Idempotency-Key` and `X-Request-ID` values. Bounded retries reuse the same prediction key and retry only transport errors, HTTP 429, or 5xx; HTTP 409 is returned once and not retried. Successful predictions and audit records remain persisted for later monitoring/dashboard use. Only the exact recovered-side synthetic parity copy was cleaned after its test; existing audit history was retained.

Reports contain safe response fields and aggregates only—no raw transaction rows, source IDs, feature vectors, or label truths. Environment-specific JSON reports and local API logs are Git-ignored.

## Source modes and HOLDOUT safety

`validation` mode reads the deterministic chronological 70%–85% development slice. It reads at most 501,959 transaction rows (TRAIN plus VALIDATION), checks that the prefix remains chronological, and separately holds `isFraud` in a private source record. Identity IDs are scanned to the VALIDATION boundary; identity features are then read only from the preceding allowed prefix. The inference payload is built from the frozen raw-column contract and never contains `isFraud`. `synthetic` mode creates schema-complete artificial rows for software tests and bounded live smoke runs.

The validation source was dry-run successfully for 32 rows and four batches. During the first exploratory dry-run, the identity loader parsed one CSV chunk that crossed the VALIDATION boundary, then filtered the chunk to validation IDs before joining. It did not read labels from those later identity rows, use them in payloads, or send them to the API. I corrected the loader to scan IDs only to find the boundary, then load identity feature columns only through the maximum VALIDATION ID; a second dry-run passed with that bounded loader.

The configured PostgreSQL history already had three successful transactions from an earlier smoke test just beyond the VALIDATION endpoint (`TransactionDT` 13,151,880–13,152,098 versus VALIDATION maximum 13,151,840). The live service correctly rejected an attempt to send earlier VALIDATION rows. I kept that existing history intact and used synthetic rows with later synthetic timestamps for live HTTP verification. No HOLDOUT labels were read, and no new HOLDOUT source rows were submitted. On API restart, recovery necessarily replayed the three pre-existing post-VALIDATION smoke rows because they were already successful records in PostgreSQL; this is distinct from simulator source selection and is recorded here explicitly.

## Scheduling, batching, load, and label delivery

Transactions are ordered by `(TransactionDT, source TransactionID)` before batching. Batch boundaries are retained. Prediction, label, and monitoring events use a deterministic tie order; seeded delay sampling makes the schedule reproducible for a fixed source/configuration. A configured time scale of `86,400` means one simulated day elapses per real second. Target RPS sets the offered schedule; sequential submission preserves the stateful single-worker constraint and reports actual throughput and accumulated prediction replay lag instead of adding concurrency.

Healthy mode sends unmodified source copies and delivers source labels later by default. Label observation/maturity timestamps are generated at delivery, after inference, and labels reach the real `/labels` endpoint. Drifted mode copies each payload, multiplies `TransactionAmt` by 8 and adds 10,000, changes about 25% of `ProductCD` values to `SIMULATOR_UNSEEN`, and makes `card4` missing in about half the rows. These changes never modify source data. Drifted mode defaults to labels disabled; original VALIDATION labels are not valid ground truth for perturbed rows.

The runner snapshots once near the midpoint and once at the end, then queries `/monitoring/summary`, `/monitoring/alerts`, and a time-filtered `/feedback/summary`. It records snapshot IDs, decision counts, request and label latency percentiles, monitoring-query latency, retry count, feedback status, and alerts. The recovery verifier recreates FastAPI from PostgreSQL and compares a subsequent HTTP prediction.

## Bounded live results

Both live runs used FastAPI with one worker and PostgreSQL 17.11. Startup `/health`, `/ready`, and `/version` checks passed. The default bundle remained `frozen_v1` with 467 features.

| Run | Rows / batch | Offered / achieved rows/sec | Replay lag | Result |
| --- | --- | --- | --- | --- |
| Healthy, synthetic software source, run `3d1681143c174965ac2a0311ffcfcacb` | 8 / 8 (average 8) | 2 / 0.350 | 18.855 sec | 8 predictions succeeded, 0 failed; 3 APPROVE and 5 REVIEW; 8 synthetic labels delivered and mature; 2 snapshots. Feedback status was `insufficient_labels` at 8 labels. No input-drift alert was emitted; the local 18,853.94 ms request produced one critical high-latency alert. |
| Drifted, synthetic software source, run `c7cbfc9746a443569c8fc0343efde112` | 32 / 32 (average 32) | 5 / 1.222 | 19.785 sec | 32 predictions succeeded, 0 failed; 31 BLOCK and 1 REVIEW; labels disabled (0 scheduled/delivered); 2 snapshots. Input, score, and decision status were `drifted`; the run snapshot contained 16 alerts, including critical input, score, decision, and latency alerts. |

Healthy mode introduced no deliberate drift, but the synthetic source is not a TRAIN-like distribution. Its input status should not be read as a healthy-distribution claim. The observed high latency reflects this local machine and batch; it is not an SLA. In the drift run, `GET /monitoring/alerts` returned 34 persisted alerts across current database history, while the run-window snapshot contained 16. Neither synthetic run is an accuracy evaluation.

Healthy-run label API p50/p95 latency was 6.33/10.75 ms. Monitoring-query latencies in request order were 55.04 ms (midpoint snapshot), 31.70 ms (final snapshot), 30.60 ms (monitoring summary), 16.09 ms (feedback summary), and 7.43 ms (alerts). Drift-run values were 35.84, 30.96, 22.79, 5.54, and 4.28 ms respectively. These are single local observations, not service-level objectives.

The healthy run used a 0.01 simulated-day label delay (compressed by the 86,400 time scale); labels arrived after inference and were immediately mature under the existing Phase 4 semantics. It reported 8 labels received/mature and `insufficient_labels` with null ranking metrics. The drifted run left delivery disabled as required. Its feedback window reported no labels.

## Restart/recovery check

The local API was restarted from its single-worker Uvicorn process. `/ready` then reported database and recovery readiness and replayed 49 persisted transactions into the runtime. A separately recreated FastAPI app recovered the same PostgreSQL stream. Continuous and recreated runtimes scored the same next synthetic feature row through `/predict/batch`, using distinct remapped IDs; all seven stored score/cost fields and effective decisions matched exactly (maximum absolute difference **0.0**). Both calls succeeded. The recovered-side copy was removed as a synthetic verification fixture so the still-running service and durable history remain aligned. Recovery results are included in the latest structured run report.

The startup replay count includes pre-existing PostgreSQL history from earlier phases as well as Phase 6 traffic. Recovery replays only successful requests and does not read labels or update target encoding.

## CLI and reports

PowerShell examples:

```powershell
$env:PYTHONPATH = '.;src'
\.venv\Scripts\python.exe scripts/run_traffic_simulation.py --scenario healthy --source validation --max-transactions 32 --batch-size 8 --dry-run
\.venv\Scripts\python.exe scripts/run_traffic_simulation.py --scenario healthy --source synthetic --max-transactions 8 --batch-size 8 --target-rps 2 --time-scale 86400
\.venv\Scripts\python.exe scripts/run_traffic_simulation.py --scenario drifted --source synthetic --max-transactions 32 --batch-size 32 --target-rps 5 --time-scale 86400
```

Healthy mode delivers labels by default. Drifted mode keeps labels off unless `--labels-enabled` is explicitly provided. `--labels-disabled`, `--label-delay-min`, `--label-delay-max`, `--seed`, and `--dry-run` configure delivery and scheduling. Dry-run loads the selected source and schedule, writes a compact report, and performs no HTTP or database mutations.

Structured output is in ignored `reports/simulation_runs/<simulation_run_id>.json` plus `reports/simulation_latest.json`. The latter includes the recovery parity result. The CLI and scripts are reusable against a running API; keep `--workers 1` for this stateful runtime.

## Tests, artifacts, and limitations

Focused tests cover deterministic schedules, chronological batch boundaries, label isolation and mapping, drift copy-only behavior/default label suppression, unique idempotency keys, bounded retry/key reuse, no 409 retry, dry-run, readiness abort, snapshot/feedback capture, JSON-safe reports, and absence of raw rows. Standard full-suite, Ruff, Alembic, and protected-hash results are recorded after final verification.

Development replay requires local IEEE-CIS TRAIN files and intentionally refuses a non-chronological prefix. A pre-populated database may have later state than VALIDATION, preventing that replay without changing the stored history; use the synthetic mode for live smoke checks in that case. Synthetic labels verify the lifecycle only. The simulator is sequential and local, not a load generator or SLA benchmark. No scheduler, dashboard, Docker, or cloud deployment is part of this phase.

Final gates: `pytest -q` **220 passed** (8 existing scikit-learn deprecation warnings); `ruff check .` and `ruff format --check .` passed; `alembic current` reported `20261013_01 (head)` and `alembic check` reported no new upgrade operations. Frozen model component hashes were verified against `artifact_manifest.json` (7/7); the `monitoring_v1` reference hash was verified; the terminal HOLDOUT report, prediction CSV, and lock matched their protected SHA-256 values. PostgreSQL verification ended with 15 inference requests, 50 transactions, 50 predictions, 8 synthetic labels, 4 monitoring snapshots, and 34 persisted alerts. No frozen model or monitoring reference artifacts changed.
