# Phase 4 — delayed labels and feedback lifecycle

## Scope and architecture

Transaction → frozen inference → original prediction persisted → ground truth unavailable.

Later: external ground truth → `POST /labels` or `/labels/batch` → `fraud_labels` → maturity check → join **stored** prediction → feedback metrics. `fraud_labels` never feeds frozen inference state or recovery in this phase. No historical transaction is rescored for evaluation.

This phase adds immutable ground-truth ingestion and evaluation only. The existing `fraud_labels` schema was sufficient; no migration or frozen ML/policy change was made. The earlier 7/14/30-day target-encoding study was an experiment, not an operational maturity rule.

## Timestamp and maturity contract

- `TransactionDT` is relative transaction time, with no calendar epoch in the persisted contract. It is distinct from the wall-clock fields and cannot support a trustworthy transaction-to-label-observed duration.
- `predictions.predicted_at` is when the frozen service originally scored the transaction.
- `fraud_labels.observed_at` is when the external source first observed ground truth. It must be timezone-aware UTC-normalizable, no earlier than stored prediction time, and no later than ingestion time.
- `fraud_labels.matured_at` is **required from the trusted label source**. It must be timezone-aware and at or after `observed_at`. No 7, 14, or 30-day production delay is assumed.
- `fraud_labels.ingested_at` is set by this service when the immutable label is first committed; a duplicate retry retains the original value.
- At a supplied UTC evaluation time, a label is mature exactly when `evaluation_time >= matured_at`. Maturity is derived, not written as a status change. Legacy rows without `matured_at`, if any, remain immature.
- `applied_to_model_state_at` remains NULL. Label ingestion and wall-clock maturity do not apply labels to target encoding or any inference state.

## API and persistence behavior

`POST /labels` accepts `TransactionID`, strict boolean `is_fraud`, bounded audit-safe `label_source`, timezone-aware `observed_at` and `matured_at`, and optional small scalar metadata. It does not accept the 433-field inference payload. `POST /labels/batch` accepts 1–256 labels in one database transaction. An unknown transaction, missing stored prediction, or unsuccessful inference request returns 404; no orphan label is inserted. An identical duplicate returns the original label with `replayed: true` and no new audit event. Changed truth or identity returns 409; labels are never overwritten. Batch conflicts or missing rows roll back the complete batch. The transaction row is locked during ingestion to serialize concurrent claims to the same ID on PostgreSQL.

Each new label commits a `label_received` audit event with only ID, source, observed/maturity timestamps, and derived status. `GET /labels/{transaction_id}` returns safe label fields; `GET /feedback/summary` returns aggregates only. No raw transaction JSON or feature vectors are returned or queried for feedback. Label source and metadata are bounded and validated. Production label sources will require authentication, authorization, source trust, and stronger operational controls; this phase does not add a full authentication system.

## Feedback definition

The summary reads scalar transaction amount, original prediction scores, probability, expected costs, base/final decisions, rule IDs, timestamps, and labels using SQLAlchemy joins. It never calls the frozen runtime. Optional `bundle_version` and `rules_enabled` filters select a prediction cohort. The overall metrics are accompanied by `by_policy` groups identified by bundle version and rules status; business decision comparisons should use those groups rather than assume mixed policies are equivalent.

`total_predictions` counts stored successful predictions in the selected cohort. `labels_received` counts labels on those predictions, split into mature and immature. `mature_label_coverage = mature_labels / total_predictions`; its denominator is **all** stored predictions in that cohort because maturity of an unobserved label is unknowable. This is observed mature-label coverage, not an assertion that every prediction is eligible for a label. Zero predictions yield null coverage. Only mature labels enter performance metrics.

PR-AUC (average precision) is the primary ranking metric; ROC-AUC is secondary. Both are null with an explicit reason for zero labels or one class. Brier, log loss, equal-frequency-bin ECE, mean stored calibrated probability, and fraud prevalence describe calibration. The 0.50 TP/FP/FN/TN, precision, recall, and F1 are **diagnostic cutoff only**, separate from the frozen Balanced business policy. Undefined precision/recall/F1 are null. Stored action counts/rates, review-or-block fraud transaction and amount capture, fraudulent approved count/amount, and legitimate review/block rates describe actual decisions. Review-or-block capture measures intervention, not proven fraud prevention. Mean stored component scores and expected costs and triggered-rule counts preserve further audit context.

Delay summaries report count, median, p90, and maximum seconds for prediction-to-observed and observed-to-maturity. Transaction-to-observed is null with the relative-time reason. Synthetic/local timings do not estimate real fraud-business delays.

## Verification and limitations

Focused tests exercise valid/unknown/duplicate/conflicting labels, atomic batch rollback, maturity filtering, zero/one/two-class metrics, diagnostic and business metrics, delay summaries, audit rows, stored-prediction evaluation, inference history immutability, recovery separation, null applied-to-state, and API privacy. The real frozen-bundle parity test additionally ingests a synthetic label before restart recovery and verifies the next transaction still matches the continuous runtime across stored score/cost fields and decisions. Live PostgreSQL verification uses only the label-free golden software fixture and synthetic labels; records are removed afterward. No HOLDOUT labels or full IEEE dataset are read for normal tests.

Operational source authentication, label correction/versioning, incomplete external truth, and calendar mapping of relative `TransactionDT` remain future design work. The completed ingestion and feedback contracts provide stored-prediction and maturity inputs for a later monitoring phase; no monitoring implementation is included here.

## Live PostgreSQL 17 verification (2026-10-13)

The configured PostgreSQL database started from the prior Phase-3 state: 4 requests, 3 transactions, 3 predictions, 0 labels, 7 audit events. A first Phase-4 smoke run successfully scored and persisted two additional chronological label-free software-fixture transactions (HTTP 200 each) and accepted a three-label batch (HTTP 200, 3 created). Its verification script then failed on a test-only snapshot comparison: the frozen runtime snapshot is a dictionary of DataFrames, not a single DataFrame. The `finally` cleanup restored the exact baseline counts; this was not a product/API failure.

The corrected live run used the three existing software-fixture predictions and three synthetic labels: two mature (one fraud, one legitimate), one immature. The batch created all three. An identical retry returned 200 with `replayed: true`; changed fraud truth returned 409. `/feedback/summary` returned 3 total predictions, 3 labels received, 2 mature, 1 immature, coverage 2/3, and metric sample size 2. The immature synthetic fraud label did not enter metrics. Example **synthetic/local** metrics: fraud prevalence 0.5, PR-AUC 0.5, ROC-AUC 0.0, Brier 0.4996444597, diagnostic confusion TP=0/FP=0/FN=1/TN=1, and one fraudulent approved transaction. These tiny fixtures are not performance estimates.

Both frozen non-label history DataFrames were identical immediately before and after label ingestion. A new app recovered the same three persisted transactions with identical histories after labels were stored; `/ready` reported database and recovery ready. All three `applied_to_model_state_at` values were NULL. The live test removed only its new label/audit rows; final PostgreSQL counts again matched the baseline: 4 requests, 3 transactions, 3 predictions, 0 labels, 7 audit events. The scored rows from the first smoke run were also removed by exact request IDs after that run. No HOLDOUT labels were used.

Verification gates: `pytest -q` **210 passed, 8 existing scikit-learn warnings**; `ruff check .` passed; `ruff format --check .` passed (135 files); `alembic current` returned `20261012_01 (head)`; `alembic check` found no new upgrade operations. All nine `models/frozen_v1` SHA-256 values matched the pre-phase baseline and `verify_bundle` passed. The three protected files were unchanged: terminal HOLDOUT report SHA-256 `8de96f153ffd3b0e398ab0c7fb257838447e101d020aae19594bc369f12ee750`, prediction lock `8b849e9e087e4cb13b39a132bd3451ea5732555fd11860ce194f4f925ce4676b`, and frozen prediction CSV `0d7f5d117436cd7227d682388c5cceddbc72b689143984a7d186d9ec03c1391f`. The CSV still matches the lock. No HOLDOUT labels were accessed and no frozen ML or policy semantics changed.


