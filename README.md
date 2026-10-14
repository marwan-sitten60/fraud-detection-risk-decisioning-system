# Fraud Detection & Risk Decisioning System

**Temporal ML, calibrated fraud risk, and cost-aware decisions in a recoverable FastAPI + PostgreSQL system with a Streamlit operations dashboard.**

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![PostgreSQL 17](https://img.shields.io/badge/PostgreSQL-17-4169E1?logo=postgresql&logoColor=white)
![Docker Compose](https://img.shields.io/badge/Deployment-Docker%20Compose-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-235%20passed-2ea44f)

**Run locally:** copy .env.example to .env, set POSTGRES_PASSWORD, then run docker compose build and docker compose up -d. Defaults: API at localhost:8000 and Dashboard at localhost:8501.

> A portfolio and production-style demonstration using IEEE-CIS Fraud Detection data and local or synthetic development traffic. It does not represent a bank deployment or live financial-institution traffic.

## At a Glance

| Metric | Verified result |
|---|---|
| Dataset / fraud prevalence | 590,540 transactions / ~3.50% |
| Model features | 467 |
| Development validation (PR-AUC / ROC-AUC) | 0.615265 / 0.916224 |
| Final terminal HOLDOUT (PR-AUC / ROC-AUC) | 0.536900 / 0.878637 |
| Fraud transaction / amount capture | 62.25% / 76.49% |
| Latest verified full test suite | 235 passed |

Capture is from the frozen Balanced expected-cost policy with rules disabled. REVIEW capture assumes an illustrative 80% review detection rate; these are modeled metrics, not observed prevention or savings.

## What I Built

Temporal ML research → leakage-controlled features → 80/20 LightGBM + XGBoost ensemble → Platt calibration → expected-cost APPROVE / REVIEW / BLOCK decisions<br>
→ frozen inference runtime → FastAPI → PostgreSQL audit, idempotency, and recovery → delayed labels → drift monitoring → traffic simulator → FraudGuard dashboard → Docker Compose.

Fraud detection needs more than classification accuracy: the system ranks rare events, estimates risk, chooses actions under explicit cost assumptions, preserves an audit trail, and tracks outcomes as they mature. Risk estimation and business decisioning remain separate.

## Architecture

The deeper online, offline development, label, monitoring, and deployment diagrams are in [docs/architecture.md](docs/architecture.md).

```mermaid
flowchart LR
    Client[External client or simulator] --> API[FastAPI, one worker]
    API --> Runtime[Frozen feature runtime]
    Runtime --> Models[LightGBM 80% + XGBoost 20%]
    Models --> Cal[Platt calibration]
    Cal --> Decision[Illustrative expected-cost decision]
    Decision --> Response[APPROVE / REVIEW / BLOCK]
    API <--> DB[(PostgreSQL 17)]
    DB --> Services[Monitoring and feedback API]
    Services --> UI[FraudGuard Streamlit]
    Labeler[Trusted external label source] --> LabelAPI[/labels API]
    LabelAPI --> Labels[(fraud_labels)]
    Labels --> Mature[Mature-label performance metrics]
    Mature --> Services
```

## Key Results

| Stage | PR-AUC | ROC-AUC |
|---|---:|---:|
| Development validation | 0.615265 | 0.916224 |
| Final terminal HOLDOUT | 0.536900 | 0.878637 |

The later HOLDOUT had lower ranking performance: PR-AUC decreased by 0.078365 and ROC-AUC by 0.037587 in absolute terms. This indicates temporal distribution shift and shows why future-time evaluation matters. Validation is not the final performance estimate, and the HOLDOUT result was not used for tuning.

Under the frozen Balanced expected-cost policy, with rules disabled, final HOLDOUT fraud transaction capture was **62.25%** and fraud amount capture was **76.49%**. These are modeled metrics; REVIEW capture assumes an illustrative 80% review detection rate. They are not observed review outcomes, prevented fraud, or financial savings.

## FraudGuard Demo

FraudGuard is a read-only, API-backed Streamlit dashboard. It calls FastAPI, not PostgreSQL or the model directly, and identifies its local or simulated environment.

- **Overview:** traffic, risk, decision, alert, and feedback summaries.
- **Transactions:** search and inspect stored scores, decisions, and expected costs.
- **Monitoring:** drift measures, sample counts, service signals, and alerts.
- **Feedback:** label coverage, maturity, and metrics when support is sufficient.
- **Simulation:** allowlisted run summaries and replay measurements.
- **System Health:** liveness, readiness, recovered state, and frozen bundle details.

**Optional screenshots:** none are currently available in the repository. After manual capture, add sanitized images under docs/images/ for Overview, Monitoring, and Transaction Inspection. Do not include credentials, local paths, or raw transaction payloads.

## Production-style system

The immutable TRAIN-fitted bundle is in **models/frozen_v1**. Normal inference loads this bundle and does not need raw IEEE-CIS data.

- **FastAPI** validates label-free chronological inputs and serves single or batch predictions.
- **PostgreSQL** stores request sequences, label-free payloads, predictions, audit events, delayed labels, and monitoring snapshots.
- **Idempotency** returns a stored result for an identical retry; a changed request with the same key returns 409.
- **Recovery** rebuilds non-label history from successful requests in original sequence and batch boundaries.
- **Delayed feedback** accepts outcomes after prediction; labels do not silently update the frozen model.
- **Monitoring** separates label-free drift and service signals from mature-label performance.
- **Simulator** sends chronological traffic through the HTTP API and supports time compression, delayed labels, backpressure, and a drifted scenario.
- **FraudGuard** is a read-only Streamlit dashboard. It calls FastAPI, not PostgreSQL or the model directly.
- **Docker Compose** runs PostgreSQL 17, a one-shot migration job, FastAPI, and Streamlit. The named volume retained data during a verified full Compose restart.

## Key Engineering Decisions

- **Chronological evaluation:** use a temporal split instead of a random split to reflect future traffic.
- **Rare-event metric:** prioritize PR-AUC for ranking fraud in an imbalanced dataset; report ROC-AUC as a complementary measure.
- **Leakage control:** fit preprocessing within folds and build behavioral histories from past events only. Rows sharing a timestamp do not see each other.
- **Ensemble:** blend LightGBM and XGBoost 80/20; the blend improved PR-AUC over LightGBM alone by about 0.0031 on the out-of-fold development comparison.
- **Calibrate before decisioning:** fit Platt calibration using TRAIN out-of-fold predictions before mapping scores to actions.
- **Use expected cost:** choose APPROVE, REVIEW, or BLOCK from explicit cost assumptions instead of a fixed 0.5 probability threshold.
- **Keep rules optional:** rules are disabled by default because the overlay increased modeled expected cost and review burden.
- **Separate delayed labels:** outcomes are persisted for later evaluation and do not update inference or recovery state.
- **Separate monitoring signals:** input, score, and service drift can be tracked without labels; predictive performance requires sufficiently mature labels.
- **Keep one API worker:** chronological feature state is process-local. PostgreSQL supports restart recovery but does not coordinate live state across multiple workers.

## Quick Start

No raw dataset is needed for normal frozen inference or the Docker demo.

In PowerShell:

    Copy-Item .env.example .env
    # Edit .env and set POSTGRES_PASSWORD to a strong, unique, local-only value.
    docker compose build
    docker compose up -d
    docker compose ps
    Invoke-RestMethod http://localhost:8000/ready

Open API documentation at http://localhost:8000/docs and FraudGuard at http://localhost:8501. On macOS or Linux, use cp .env.example .env. Keep .env private; it is ignored by Git. Docker binds to loopback by default.

To stop while preserving PostgreSQL data:

    docker compose down

docker compose down -v removes the named volume and deletes its database contents.

## Monitoring and Feedback

Input and score drift, decision mix, entity novelty, request health and latency, and label coverage can be measured before outcomes mature. Numeric drift uses PSI; categorical drift uses Jensen-Shannon divergence. Drift signals a distribution change; it does not by itself prove model failure.

Trusted outcomes may arrive later through chargebacks, fraud reports, analyst resolution, or another validated source. Performance metrics such as PR-AUC, ROC-AUC, Brier score, and fraud capture require mature labels joined to stored predictions. Feedback reports insufficient support instead of presenting metrics from too few labels. Labels do not update model features, calibration, or policy.

## API

Interactive OpenAPI documentation is available at /docs.

| Category | Endpoints |
|---|---|
| System | GET /health, GET /ready, GET /version |
| Inference and audit | POST /predict, POST /predict/batch, GET /predictions, GET /predictions/{transaction_id} |
| Feedback | POST /labels, POST /labels/batch, GET /labels/{transaction_id}, GET /feedback/summary |
| Monitoring | GET /monitoring/summary, GET /monitoring/drift, GET /monitoring/alerts, POST /monitoring/snapshot |

Prediction requests follow the frozen raw-input contract and remain chronological by TransactionDT then TransactionID. Target labels are rejected. Use X-Idempotency-Key for retryable requests.

## Traffic Simulator

The simulator preserves chronological row order and batch boundaries while sending through the HTTP API. It supports synthetic and development replay sources, idempotent retries, delayed label scheduling, time compression, target versus achieved throughput, replay lag, and a drifted scenario. Throughput is a local software measurement, not model performance or an SLA. Feature generation is the main local inference bottleneck.

## Manual development setup

For model research, use Python 3.11 or newer:

    python -m venv .venv
    python -m pip install -e ".[dev]"

Offline model work requires separately obtained IEEE-CIS files. Place train_transaction.csv and train_identity.csv under data/raw/. The data is not included and must not be redistributed from this repository. Normal frozen inference does not read those files.

For local serving, configure FRAUD_DATABASE_URL outside version control, apply migrations with python -m alembic upgrade head, then run Uvicorn with one worker. Set FRAUD_API_BASE_URL before starting Streamlit.

    python -m streamlit run dashboard/app.py

## Testing

Final automated verification on 2026-10-14 passed **235 pytest tests** with 8 existing scikit-learn deprecation warnings. Ruff lint and formatting passed. Docker health and Compose configuration passed; Alembic reported 20261013_01 (head) with no schema drift. See the [Docker verification report](reports/docker_deployment_summary.md) and [final release summary](reports/final_project_release_summary.md).

## Limitations

- The API uses one worker because chronological feature history is process-local. PostgreSQL supports restart recovery but does not coordinate live state across workers; horizontal scaling is not safe in this version.
- The demonstration uses local, synthetic, and development replay traffic. It is not a bank deployment and has no production SLA.
- Authentication and authorization are not implemented.
- There is no automatic retraining.
- Business costs and review effectiveness are illustrative assumptions, not measured financial outcomes.
- Stateful feature generation is computationally expensive for local single-row inference.

## Future production work

A production system would need an externally coordinated ordered feature-state service, managed PostgreSQL, secret management, authentication and authorization, centralized observability, and an evaluated retention policy. A separately governed model lifecycle could use matured labels for temporal retraining and calibration, followed by champion/challenger evaluation and shadow or canary promotion. None of these capabilities are implemented here.

## Repository Map

    api/                         FastAPI application
    dashboard/                   FraudGuard Streamlit UI
    migrations/                  Alembic schema revisions
    models/frozen_v1/            Immutable inference artifacts
    models/monitoring_v1/         Fixed TRAIN-based monitoring reference
    reports/                     Evaluation and engineering records
    scripts/                     Build, simulation, and verification commands
    src/fraud_detection/         Reusable feature, model, persistence, and service code
    tests/                       Automated tests
    docker-compose.yml           Local four-service stack

## Deep Dive

- [Report index](reports/README.md)
- [Final terminal HOLDOUT evaluation](reports/final_holdout_evaluation_summary.md)
- [Production monitoring](reports/production_monitoring_summary.md)
- [Traffic simulator](reports/traffic_simulator_summary.md)
- [Docker deployment](reports/docker_deployment_summary.md)
- [Final release summary](reports/final_project_release_summary.md)

### Modeling Details

The IEEE-CIS labeled transaction file contains 590,540 rows with roughly 3.50% fraud. The chronological development split used 413,378 TRAIN rows, 88,581 VALIDATION rows, and 88,581 later HOLDOUT rows. TRAIN prevalence was 3.52%.

Fold-local preprocessing and past-only histories prevent future rows from contributing to earlier features; rows sharing a timestamp do not see each other. The final frozen pipeline has 467 model features.

The development path included Logistic Regression, LightGBM, XGBoost, frequency and missingness features, time transformations, past transaction behavior, composite identities, graph features, temporal target encoding, CatBoost, sample weighting, ensembling, calibration, expected-cost decisions, optional rules, and TreeSHAP diagnostics. Historical/composite features and temporal target encoding were retained. CatBoost trailed the preferred LightGBM challenger, time-decay weighting did not beat unweighted fitting, and graph ablations were mixed and are not in the final feature bundle.

The selected risk model is an 80/20 LightGBM/XGBoost probability blend. It improved PR-AUC over LightGBM alone by about 0.0031 on the out-of-fold development comparison. Platt calibration was selected using TRAIN out-of-fold predictions. TreeSHAP supported component-model interpretation and error analysis; it does not establish causation or explain the calibrated business decision exactly. Detailed records are indexed in [reports/README.md](reports/README.md).

The frozen rules overlay improved some fraud-capture measures but increased modeled expected cost and review burden. It remains available for analysis and is disabled by default in the primary policy.

### Data and License

The IEEE-CIS data is not included. Obtain it through the [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/competitions/ieee-fraud-detection) and follow its terms. The repository has no LICENSE file; no software license is asserted by this project.
