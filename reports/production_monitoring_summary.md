# Production Monitoring, Drift Detection, and Alerting

Status: **completed** on 2026-10-13 (Africa/Cairo). Monitoring was added around the existing `frozen_v1` inference and PostgreSQL feedback services. It does not train or score transactions, change inference state, or start automatic retraining.

## Architecture and reference

The reusable implementation is under `src/fraud_detection/monitoring/`. `MonitoringService` reads scalar request, prediction, transaction, and mature-label fields from PostgreSQL. It does not select or return full transaction payloads. `FeedbackService` remains the sole implementation of matured-label metrics; the monitoring service passes its selected UTC window to that service. The monitoring API exposes `GET /monitoring/summary`, `GET /monitoring/drift`, `GET /monitoring/alerts`, and manual `POST /monitoring/snapshot`. No scheduler is included.

The compact, versioned reference is `models/monitoring_v1/reference.json`, verified against its SHA-256 in `models/monitoring_v1/manifest.json` (`502e3a882c406e9dbaf971a9d965d5d79aa09557f736e57e648d3371abc0be1a`). Numeric edges and categorical frequencies are fixed in that artifact. Inputs come from the first 413,378 TRAIN transaction rows, with identity rows restricted to TRAIN transaction IDs. Score references use TRAIN OOF component scores transformed by the existing frozen Platt calibrator; policy decisions use the existing Balanced policy. TRAIN OOF labels are used only for a reference performance comparison. The manifest records `holdout_used: false`; no HOLDOUT labels were accessed.

## Metrics and interpretation

Label-free monitoring includes request volume, success and error rates, scored transactions, average batch size, p50/p95/p99 latency, bundle/database/recovery readiness, amount and selected input distributions, missingness, entity novelty for card1/card2/addr1/DeviceType, LightGBM/XGBoost/ensemble/calibrated score distributions, model disagreement, decision mix, and rule trigger rate. Inputs monitored are `TransactionAmt`, `ProductCD`, `card4`, `card6`, `DeviceType`, `P_emaildomain`, `R_emaildomain`, selected missingness indicators, and the listed entity novelty fields.

Numeric drift uses Population Stability Index (PSI) over reference-defined bins plus a missing bucket. Categorical drift uses Jensen-Shannon divergence; categories outside the reference's complete known set are explicitly counted as unseen. No production-window bins are learned. Score drift uses PSI; decision drift uses Jensen-Shannon divergence. Drift describes a distribution change and does not establish model failure.

Matured-label monitoring reports labels received, mature and immature counts, coverage, fraud prevalence, PR-AUC, ROC-AUC, Brier score, Log Loss, diagnostic precision/recall/F1, transaction and amount capture, decision outcomes by label, and label-delay summaries through the Phase 4 feedback service. With fewer than 200 mature examples or fewer than 20 examples of either class, the feedback status is `insufficient_labels`; performance alerts use the same support rule. Empty-window performance metrics are null, not zero.

Windows are UTC `1h`, `24h`, and `7d`, or an explicit timezone-aware `from` and `to` interval. Labels are eligible only when mature by the evaluation time. The service reports observed request latencies; the separate database query/snapshot execution duration is not currently instrumented.

## Alert policy

Drift alerts require at least 30 scored transactions in the window. Operational thresholds are documented defaults, not learned values: PSI warning/critical at 0.20/0.30; Jensen-Shannon warning/critical at 0.10/0.20; request error-rate warning/critical at 5%/15%; p95 latency warning/critical at 2,000/5,000 ms; and label-coverage warning below 10% after at least 50 scored transactions. With adequate label support, performance warnings compare PR-AUC against 75% of the TRAIN OOF reference and Brier score against reference +0.03. Thresholds do not use HOLDOUT data. Alerts are operational signals only and never trigger retraining.

## Persistence and verification scenarios

Alembic migration `20261013_01_monitoring` adds `monitoring_snapshots` and `monitoring_alerts`, including UTC window/evaluation times, bundle, counts, JSON metrics, alert details, and lookup indexes. Snapshots and their alerts are committed together.

The healthy synthetic alert-policy case used reference-matched zero drift, zero error rate, 10 ms p95 latency, adequate coverage, and no mature performance sample; it emitted **no alerts**. The intentionally shifted case used amount PSI 0.35 and emitted a **CRITICAL input-drift alert**. Focused tests also verified the 30-row shifted distribution, deterministic severity, filtered alert persistence, empty windows, and no inference/runtime/label mutation.

The live PostgreSQL smoke test confirmed `FRAUD_DATABASE_URL` was present without printing it and connected to PostgreSQL **17.11**. `/ready` returned ready with `database_ready: true` and `state_recovered: true`. A synthetic label-free batch persisted 30 predictions; two test labels were ingested, one mature and one immature at evaluation time. The 24-hour snapshot counted 33 transactions including existing history, reported 1 mature and 1 immature label, and persisted 12 alerts. The amount PSI was **8.9073** and input drift was `drifted`; decision and score drift alerts were also emitted. The window's measured error rate was 40% and p95 latency about 29,028 ms, which includes existing requests in the selected window and is not attributed solely to this batch. The request log measured 22,409.55 ms total latency, including 22,174.43 ms inference, 5.19 ms database lookup, and 124.49 ms database write. Cleanup restored exact pre-test counts: 4 inference requests, 3 transactions, 3 predictions, 0 fraud labels, 7 audit events, 0 monitoring snapshots, and 0 monitoring alerts.

`fraud_labels` is queried only through the existing feedback service for label coverage and mature performance. It is not read by inference or runtime recovery. Synthetic labels were removed after verification.

## Limitations and next phase

Entity novelty uses exact membership in compact TRAIN value sets and does not estimate composite identity novelty. Monitoring queries and snapshot generation are synchronous and have no dedicated execution-duration instrumentation. The phase provides API and PostgreSQL history but no scheduler, simulator, or dashboard; those belong to later work. Input or score drift alone must not be interpreted as degraded model performance; that conclusion requires matured labels and adequate class support.

## Gates and protected artifacts

`pytest -q`: **214 passed**, 8 existing scikit-learn deprecation warnings. `ruff check .` and `ruff format --check .` passed. `alembic current`: **20261013_01 (head)**. `alembic check`: **No new upgrade operations detected**. The live PostgreSQL smoke test restored all seven table counts exactly to their pre-test values.

All seven frozen model component files match `artifact_manifest.json`: feature state `2a7f86d6d4b417b4c948b0e2fa49449d8e06e147fe330d9f6c957cca00197af5`, LightGBM `201cb1747d25b4dc888269fb41cd389f989f2545a5ee6a6b3aab273d2da462cd`, XGBoost `2c13ed79860cfc593623d09932c135b6b4ed0cc34502f983a7bd089218c17a6c`, calibrator `3503bf66161b5152e9de4d638b9e7932d581fc371abd6a45b6969ba6c8134f6e`, rules `3a4380831e177828c82a1a118259f83e354221fa52205b41d38aa21b58febb94`, policy `3685f31c1af31acfb68a98d4003164fc19051b938dfc37fa9503587da669470c`, feature contract `047700d867cf360770bd0c419a0d672dd702a4a63a825569bede7897b4b54100`. Protected terminal report, prediction CSV, and lock hashes remain `8de96f153ffd3b0e398ab0c7fb257838447e101d020aae19594bc369f12ee750`, `0d7f5d117436cd7227d682388c5cceddbc72b689143984a7d186d9ec03c1391f`, and `8b849e9e087e4cb13b39a132bd3451ea5732555fd11860ce194f4f925ce4676b` respectively.

No HOLDOUT labels were accessed. Monitoring did not alter model, calibration, policy, target-encoding, or inference semantics.
