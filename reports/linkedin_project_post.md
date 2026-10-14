I built a fraud risk decisioning system around a practical idea: a fraud score is not yet a business decision.

The portfolio project uses IEEE-CIS Fraud Detection data and chronological evaluation. The selected model blends LightGBM and XGBoost (80/20), then applies TRAIN-OOF Platt calibration before an illustrative expected-cost policy chooses APPROVE, REVIEW, or BLOCK.

The development validation PR-AUC was 0.6153. On the later terminal HOLDOUT, PR-AUC was 0.5369 and ROC-AUC was 0.8786. I kept that drop visible: future-time evaluation exposed temporal distribution shift, and the HOLDOUT was not used to retune the system.

I also built the surrounding software: a one-worker FastAPI API, PostgreSQL persistence and idempotency, restart recovery, delayed-label feedback, drift and service monitoring, a chronological HTTP simulator, the FraudGuard Streamlit dashboard, and a Docker Compose demo.

The HOLDOUT fraud-capture figures (62.25% of transactions and 76.49% of amount) are modeled under an assumed 80% review detection rate. They are not observed reviewer outcomes or financial savings.

This is a local/synthetic production-style demonstration, not a bank deployment. The next engineering challenge would be coordinating ordered feature state across workers while preserving deterministic recovery.

#FraudDetection #MachineLearning #MLOps

