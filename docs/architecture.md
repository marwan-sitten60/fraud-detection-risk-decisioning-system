# System architecture

This repository is a local, production-style portfolio demonstration. It does not represent a deployed financial institution system. The final model and policy artifacts are frozen; these diagrams describe their existing behavior.

## Offline development and evaluation

```mermaid
flowchart LR
    TRAIN["TRAIN Partition Only"] --> CV["Expanding Chronological Folds"]
    CV --> FE["Leakage-Controlled Feature Experiments"]
    FE --> CAND["Candidate Models and Features"]
    CAND --> SEL["OOF Blend, Calibration, and Policy Selection"]
    SEL --> VAL["Development VALIDATION Assessment"]
    VAL --> FREEZE["Freeze Model, Features, Calibrator, and Policy"]
    FREEZE --> HOLD["Terminal HOLDOUT Evaluation Once"]
    FREEZE --> BUNDLE["frozen_v1 Artifacts"]
    HOLD --> LOCK["Locked Terminal Result"]
    HOLD -. no feedback into tuning .-> SEL
```

The dataset is split in time order: 413,378 TRAIN, 88,581 VALIDATION, and 88,581 later HOLDOUT rows. Preprocessing and target-derived historical features are fit or replayed using only information available earlier in time. The terminal HOLDOUT result was not used to revise the frozen system.

## Online scoring and audit

```mermaid
flowchart LR
    CLIENT["External Client / Simulator"] --> API["FastAPI — Single Worker"]
    API --> CONTRACT["433-Column Label-Free Input Contract"]
    CONTRACT --> FEAT["Frozen Chronological Feature Runtime"]
    FEAT --> LGB["LightGBM"]
    FEAT --> XGB["XGBoost"]
    LGB --> BLEND["80/20 Probability Blend"]
    XGB --> BLEND
    BLEND --> CAL["TRAIN-OOF Platt Calibrator"]
    CAL --> COST["Balanced Illustrative Expected Costs"]
    COST --> ACTION["APPROVE / REVIEW / BLOCK"]
    API <--> DB[("PostgreSQL 17")]
    API --> RESULT["Prediction Response"]
    DB --> RECOVER["Startup Replay of Successful Requests"]
    RECOVER --> FEAT
```

For each accepted prediction operation, the API stores a durable request sequence and boundary, canonical label-free raw payload, prediction, and audit records. A successful batch writes its rows atomically. An identical idempotency-key retry returns the stored response and does not update runtime history. A changed payload with that key returns 409. Reusing an already-scored TransactionID with a new key returns 409.

At process startup, the API verifies the frozen bundle and schema, then replays successful requests in sequence order using their original batch boundaries. Replay reads requests and transaction payloads, does not write new predictions, and does not read fraud labels.

## Delayed labels and feedback

```mermaid
flowchart LR
    Source["Trusted External Outcome Source"] --> API["POST /labels"]
    API --> LABELS[("fraud_labels")]
    PRED[("Stored Predictions")] --> JOIN["Feedback Service Joins Stored Outcomes"]
    LABELS --> JOIN
    JOIN --> READY{"Mature and Sufficient Support?"}
    READY -->|yes| METRICS["Mature-Label Metrics"]
    READY -->|no| NULL["Insufficient Support; Metrics Unavailable"]
```

Labels can represent a later chargeback, trusted fraud report, or analyst resolution. The source supplies observation and maturity timestamps. Immature labels can be stored but do not enter performance calculations until mature. Label ingestion never updates frozen features, calibration, or policy.

## Monitoring

```mermaid
flowchart LR
    DB[("PostgreSQL Request, Prediction, and Label Records")] --> MF["Monitoring and Feedback Services"]
    REF["Fixed TRAIN-Based Reference"] --> MF
    MF --> LF["Label-Free Signals: PSI, JS Divergence, Score and Decision Drift, Novelty, Health, Latency"]
    MF --> ML["Mature-Label Performance and Coverage"]
    LF --> SNAP["Persisted Snapshots and Alerts"]
    ML --> SNAP
    SNAP --> API["Monitoring API"]
    API --> DASH["FraudGuard Dashboard"]
```

A drift signal is evidence of a changed distribution, not proof of model failure. Performance degradation requires mature labels and sufficient support. No alert triggers automatic retraining.

## Local deployment

```mermaid
flowchart LR
    DB[("PostgreSQL 17 Named Volume")] --> MIG["One-Shot Alembic Migration"]
    MIG --> API["FastAPI — Single Worker"]
    API --> DASH["Streamlit FraudGuard"]
    Client["Local Caller or Simulator"] --> API
```

Compose health checks gate startup: the database becomes healthy before migration, migration must finish successfully before API startup, and dashboard waits for API health. The API is intentionally single-worker because mutable chronological feature history is process-local. PostgreSQL enables recovery after restart; it does not coordinate live histories across replicas.

