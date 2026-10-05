# Fraud Detection and Risk Decisioning System

A production-minded portfolio project for detecting fraudulent card transactions and turning
model probabilities into explainable business decisions. The project will use the IEEE-CIS
Fraud Detection dataset and will be developed incrementally from data validation through model
serving, monitoring, and retraining.

> **Current status:** repository foundation only. No dataset, analysis, trained model, API,
> dashboard, database integration, or deployment has been implemented yet.

## Business problem

Payment fraud creates direct financial loss, operational cost, and customer friction. A useful
fraud system must do more than rank transactions: it must convert risk into an approve, review,
or block decision while balancing fraud loss against false declines and manual-review capacity.

## Why fraud detection is challenging

- Fraud is rare, so accuracy can look high even for an ineffective model.
- Fraud patterns change as attackers adapt and customer behavior evolves.
- Labels often arrive late after investigations or chargebacks.
- False positives have real costs, including lost revenue and poor customer experience.
- Transaction data can leak future information unless validation respects time.
- Decisions need to be explainable and observable in production.

## Planned architecture

```text
Transaction
    ↓
Feature Engineering
    ↓
Tabular ML Model
    ↓
Graph-Based Features
    ↓
Rules Engine
    ↓
Risk Score
    ↓
Approve / Review / Block
    ↓
Prediction Storage
    ↓
Feedback
    ↓
Monitoring
    ↓
Retraining
```

The exact component boundaries may evolve as the data and operational requirements become
clear. PostgreSQL is planned for prediction storage, MLflow for experiment tracking and model
registry, FastAPI for serving, and Streamlit for a demonstration dashboard.

## Repository layout

```text
.
├── api/                     # Future FastAPI application
├── configs/                 # Versioned, non-secret experiment configuration
├── dashboard/               # Future Streamlit application
├── data/                    # Local data stages (contents ignored by Git)
├── docker/                  # Future container definitions
├── models/                  # Local trained artifacts (ignored by Git)
├── notebooks/               # Exploratory work, added only when substantive
├── reports/                 # Generated reports and figures
├── scripts/                 # Command-line workflow entry points
├── src/fraud_detection/     # Installable Python package
├── tests/                   # Automated tests
└── .github/workflows/       # Future CI workflows
```

## Local setup

Python 3.11 or newer is required.

### macOS or Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit `.env` for the local environment. It is excluded from version control.

## Development commands

With the virtual environment active, use the Make targets where `make` is available:

```bash
make test
make lint
make format
```

The equivalent cross-platform Python commands are:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m ruff check --fix .
python -m ruff format .
```

## Planned development phases

1. **Foundation** — package structure, configuration, code-quality tooling, and smoke tests.
2. **Data understanding** — acquire IEEE-CIS data manually, validate schemas, perform EDA, and
   establish leakage-safe time-based splits.
3. **Modeling** — engineer tabular features, build LightGBM/XGBoost baselines, address class
   imbalance, calibrate probabilities, and evaluate business costs.
4. **Decisioning and explainability** — add cost-sensitive thresholds, rules, graph-derived
   features, and SHAP explanations.
5. **Serving and persistence** — expose predictions through FastAPI, store them in PostgreSQL,
   and build a Streamlit dashboard.
6. **MLOps** — add MLflow tracking and registry, drift monitoring, delayed-label simulation,
   retraining, and champion/challenger comparisons.
7. **Delivery** — containerize components, expand pytest coverage, and add GitHub Actions CI/CD.

## Data policy

The IEEE-CIS dataset is not downloaded automatically and must not be committed. Place local
source files under `data/raw/`; Git ignores files inside all data-stage directories.

