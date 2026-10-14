# Final project release summary

**Status: packaging and repository audit complete**, 2026-10-14. This is a portfolio and production-style demonstration, not a financial institution deployment. Phase 9 finalized documentation, repository ignore rules, CI, and the Dashboard import-path configuration (`PYTHONPATH=/app`). Model, inference, persistence, recovery, and business decision behavior remain frozen.

## Project goal and data

The system demonstrates fraud risk estimation followed by calibrated, cost-aware business decisioning and the software needed to preserve and observe those decisions.

The IEEE-CIS labeled transaction file contains 590,540 rows with approximately 3.50% fraud. The chronological split is 413,378 TRAIN rows, 88,581 development VALIDATION rows, and 88,581 later terminal HOLDOUT rows. TRAIN contains 14,538 fraud rows (3.52%). Raw competition files are not included; normal frozen inference does not need them.

## Architecture and modeling

The final 467-feature pipeline uses frequency, missingness/time, past-only behavioral and composite-identity features, and temporally safe target encoding. LightGBM and XGBoost probabilities are blended 80/20, followed by the frozen TRAIN-OOF Platt calibrator. A separate Balanced illustrative expected-cost policy selects APPROVE, REVIEW, or BLOCK. Rules are available as an optional overlay but disabled in the primary policy because they increased modeled expected cost despite improving some capture measures.

Development explored Logistic Regression, boosting, temporal and behavioral features, composite and graph features, CatBoost, sample weighting, ensembling, calibration, rules, and TreeSHAP diagnostics. Some features improved chronological validation; alternatives that did not justify their complexity were not included. The later HOLDOUT was evaluated after the model/policy freeze and was not used for tuning.

See [system architecture](../docs/architecture.md) for offline, online, feedback, monitoring, and deployment diagrams.

## Evaluation

| Stage | PR-AUC | ROC-AUC |
|---|---:|---:|
| Development VALIDATION | 0.615265 | 0.916224 |
| Terminal HOLDOUT | 0.536900 | 0.878637 |

The future-period HOLDOUT showed lower ranking performance: absolute PR-AUC change -0.078365 and ROC-AUC change -0.037587. This indicates temporal distribution shift and supports future-time evaluation. Validation is not presented as the final performance.

The rules-disabled Balanced policy modeled 62.25% fraud transaction capture and 76.49% fraud amount capture on HOLDOUT. These calculations assume 80% REVIEW detection; they are not observed review outcomes, fraud prevention, or actual financial savings.

## Productionization delivered

- Frozen, checksum-verified TRAIN-only inference bundle.
- FastAPI serving with a single worker and chronological input contract.
- PostgreSQL request, transaction, prediction, audit, label, and monitoring persistence.
- Idempotent retry handling, atomic batches, and sequence/boundary-preserving restart recovery.
- Delayed-label ingestion and mature-label feedback metrics; labels do not update the frozen model.
- Fixed TRAIN-based drift reference, monitoring snapshots, and alerts.
- Chronological HTTP traffic simulator, including delayed labels and drifted synthetic input.
- FraudGuard Streamlit operations dashboard that calls the API.
- Docker Compose stack with PostgreSQL 17, Alembic migration job, API, and dashboard. Live Phase 8 verification confirmed health, inference, persistence, restart recovery, and next-transaction parity; the named volume survived down/up without -v.

## Automated quality and integrity

- Pytest: **235 passed**, with 8 existing scikit-learn deprecation warnings.
- Ruff lint and format checks: passed.
- Python package wheel `fraud_detection_system-0.1.0-py3-none-any.whl` built with the declared isolated build requirements, installed into a temporary target, and imported successfully. The 62-entry wheel contains the reusable `fraud_detection` modules and distribution metadata; it contains no datasets, trained model files, reports, or dashboard files. The Docker images package the service-specific runtime separately.
- GitHub Actions workflow covers pytest, both Ruff checks, and Compose configuration. The same local gates passed. No Git remote is configured, so this workflow has not run on GitHub and no CI badge is shown.
- Compose config: passed. Docker API, dashboard, and database: healthy. GET /ready reports database and recovery readiness.
- Dashboard import-path follow-up: Compose sets `PYTHONPATH=/app`; after recreating only the Dashboard container, the package and all six page modules imported, all six Streamlit page tests completed without exceptions, and the health/root/page routes returned HTTP 200. API and database containers were unchanged. No connected browser was available, so no visual browser verification is claimed.
- Alembic: **20261013_01 (head)**; check reports no schema drift.
- frozen_v1 payloads and manifest: SHA-256 verification passed; bundle size is 24,482,230 bytes.
- monitoring_v1 reference: hash matches its manifest (502e3a882c406e9dbaf971a9d965d5d79aa09557f736e57e648d3371abc0be1a); reference semantics were unchanged.
- Protected terminal HOLDOUT report, label-free frozen predictions, and prediction lock matched their expected hashes: 8de96f153ffd3b0e398ab0c7fb257838447e101d020aae19594bc369f12ee750, 0d7f5d117436cd7227d682388c5cceddbc72b689143984a7d186d9ec03c1391f, and 8b849e9e087e4cb13b39a132bd3451ea5732555fd11860ce194f4f925ce4676b. These files were hashed only. No new HOLDOUT labels were accessed.

## Repository and security audit

The local raw IEEE-CIS CSVs remain on disk and are ignored; no raw transaction or identity CSV is tracked. The 24.48 MB frozen inference bundle and 228,333-byte monitoring bundle are now explicitly allowed by Git because the demo requires them. Other generated model artifacts stay ignored.

The largest tracked files are the protected label-free HOLDOUT prediction CSV (11,634,555 bytes), OOF prediction CSV (5,659,236 bytes), and EDA notebook (582,277 bytes). No unexpected large file was found. Local databases, runtime logs, temporary golden input, simulator output, caches, and .env remain ignored.

The current publishable files and reachable history scan found no high-confidence credential candidates outside HOLDOUT-protected artifacts. No Git history rewrite is indicated by that scan. No user-profile path was found. Hardcoded repository-root tracking URIs were replaced with relative paths in non-protected reports; notebooks resolve their tracking database from the project root. A generic repository-root tracking URI remains in the byte-protected terminal HOLDOUT summary, which was left unchanged.

.env.example contains only safe database names, user/port settings, and an empty password field. The real local .env is ignored and its value was not printed. No LICENSE file exists; none was invented. IEEE-CIS data is identified as separately obtained and not redistributed. No Git remote is configured; technology badges were omitted because no repository URL supports a truthful CI badge.

### Local credential checklist

- Rotate temporary local PostgreSQL passwords before reuse.
- Update the local FRAUD_DATABASE_URL after rotation.
- Verify .env remains ignored.
- Never reuse local development credentials elsewhere.

The user's Windows environment was not modified.

## Known limitations and next production work

The current API is single-worker because chronological feature history is process-local. PostgreSQL supports restart recovery but does not coordinate state across live workers. There is no authentication or authorization, production SLA, cloud deployment, or automatic retraining. Business-cost and review assumptions are illustrative, and local feature generation is computationally expensive.

Future production work would require shared deterministic ordered state, managed PostgreSQL, secret management, access control, centralized observability, data-retention policy, and a separately governed retraining/calibration process with champion/challenger and shadow/canary evaluation.

## Release recommendation

The documentation and intended deployment artifacts are ready for commit and portfolio review. The application remains explicitly a local/synthetic production-style demonstration. Before public reuse, the repository owner should choose a software license and rotate local PostgreSQL credentials. No new product functionality, model training, tuning, or HOLDOUT evaluation was performed in this phase.

