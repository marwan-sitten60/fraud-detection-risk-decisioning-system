# System architecture

This repository is a local, production-style portfolio demonstration. It does not represent a deployed financial institution system. The final model and policy artifacts are frozen; these diagrams describe their existing behavior.

## Offline development and evaluation

```mermaid
flowchart LR
    TRAIN[TRAIN partition only] --> CV[Expanding chronological folds]
    CV --> FE[Leakage-controlled feature experiments]
    FE --> CAND[Candidate models and features]
    CAND --> SEL[OOF blend, calibration, and policy selection]
    SEL --> VAL[Development VALIDATION assessment]
    VAL --> FREEZE[Freeze model, features, calibrator, and policy]
    FREEZE --> HOLD[Terminal HOLDOUT evaluation once]
    FREEZE --> BUNDLE[frozen_v1 artifacts]
    HOLD --> LOCK[Locked terminal result]
    HOLD -. no feedback into tuning .-> SEL
```

The dataset is split in time order: 413,378 TRAIN, 88,581 VALIDATION, and 88,581 later HOLDOUT rows. Preprocessing and target-derived historical features are fit or replayed using only information available earlier in time. The terminal HOLDOUT result was not used to revise the frozen system.

## Online scoring and audit

```mermaid
flowchart LR
    CLIENT[External client or simulator] --> API[FastAPI, one worker]
    API --> CONTRACT[433-column label-free input contract]
    CONTRACT --> FEAT[Frozen chronological feature runtime]
    FEAT --> LGB[LightGBM]
    FEAT --> XGB[XGBoost]
    LGB --> BLEND[80/20 probability blend]
    XGB --> BLEND
    BLEND --> CAL[TRAIN-OOF Platt calibrator]
    CAL --> COST[Balanced illustrative expected costs]
    COST --> ACTION[APPROVE / REVIEW / BLOCK]
    API <--> DB[(PostgreSQL)]
    API --> RESULT[Prediction response]
    DB --> RECOVER[Startup replay of successful requests]
    RECOVER --> FEAT
```

For each accepted prediction operation, the API stores a durable request sequence and boundary, canonical label-free raw payload, prediction, and audit records. A successful batch writes its rows atomically. An identical idempotency-key retry returns the stored response and does not update runtime history. A changed payload with that key returns 409. Reusing an already-scored TransactionID with a new key returns 409.

At process startup, the API verifies the frozen bundle and schema, then replays successful requests in sequence order using their original batch boundaries. Replay reads requests and transaction payloads, does not write new predictions, and does not read fraud labels.

## Delayed labels and feedback

```mermaid
flowchart LR
    Source[Trusted external outcome source] --> API[POST /labels]
    API --> LABELS[(fraud_labels)]
    PRED[(Stored predictions)] --> JOIN[Feedback service joins stored outcomes]
    LABELS --> JOIN
    JOIN --> READY{Mature and sufficient support?}
    READY -->|yes| METRICS[Mature-label metrics]
    READY -->|no| NULL[Insufficient support; metrics unavailable]
```

Labels can represent a later chargeback, trusted fraud report, or analyst resolution. The source supplies observation and maturity timestamps. Immature labels can be stored but do not enter performance calculations until mature. Label ingestion never updates frozen features, calibration, or policy.

## Monitoring

```mermaid
flowchart LR
    DB[(PostgreSQL request, prediction, and label records)] --> MF[Monitoring and feedback services]
    REF[Fixed TRAIN-based reference] --> MF
    MF --> LF[Label-free signals: PSI, JS divergence, score and decision drift, novelty, health, latency]
    MF --> ML[Mature-label performance and coverage]
    LF --> SNAP[Persisted snapshots and alerts]
    ML --> SNAP
    SNAP --> API[Monitoring API]
    API --> DASH[FraudGuard dashboard]
```

A drift signal is evidence of a changed distribution, not proof of model failure. Performance degradation requires mature labels and sufficient support. No alert triggers automatic retraining.

## Local deployment

```mermaid
flowchart LR
    DB[(PostgreSQL 17 named volume)] --> MIG[One-shot Alembic migration]
    MIG --> API[FastAPI, one worker]
    API --> DASH[Streamlit FraudGuard]
    Client[Local caller or simulator] --> API
```

Compose health checks gate startup: the database becomes healthy before migration, migration must finish successfully before API startup, and dashboard waits for API health. The API is intentionally single-worker because mutable chronological feature history is process-local. PostgreSQL enables recovery after restart; it does not coordinate live histories across replicas.

