# CV bullets

- Built a leakage-controlled temporal fraud modeling pipeline over 590k IEEE-CIS transactions, engineering a 467-feature system and selecting an 80/20 LightGBM/XGBoost ensemble; achieved 0.6153 validation PR-AUC and 0.5369 on the later terminal HOLDOUT, documenting the temporal generalization drop.
- Connected out-of-fold Platt calibration to an illustrative expected-cost policy for APPROVE / REVIEW / BLOCK; the frozen rules-disabled policy modeled 62.25% fraud transaction capture and 76.49% fraud amount capture on HOLDOUT under an assumed 80% review detection rate.
- Delivered a one-worker FastAPI service with PostgreSQL audit, idempotent retries, atomic batches, delayed-label feedback, monitoring, chronological recovery, FraudGuard, and Docker Compose; verified 235 tests and live PostgreSQL restart and parity behavior.

The business capture figures are modeled outcomes under illustrative review assumptions, not observed savings or fraud prevention.

