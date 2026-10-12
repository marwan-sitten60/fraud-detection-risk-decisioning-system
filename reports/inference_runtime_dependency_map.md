# Frozen inference dependency map

The terminal evaluation rebuilt fitted components inside notebook 18. Only the TRAIN-OOF Platt calibrator (`models/probability_calibrator.joblib`) and optional rule definitions (`reports/rules_frozen_config.json`) were previously persisted. LightGBM, XGBoost, frequency maps, non-label histories, target-encoding history, tabular preprocessing, and the scaler existed only in the notebook process.

| Input or state | Frozen source | Inference behavior |
| --- | --- | --- |
| Raw transaction and identity fields | Merged IEEE-CIS TRAIN schema, excluding `isFraud` | Caller supplies the same ordered fields; the runtime rejects `isFraud`. `TransactionID`, `TransactionDT`, and `TransactionAmt` are mandatory. |
| Base frequency maps | `Phase1FeatureEngineer`, TRAIN fit | Immutable after TRAIN. |
| Historical amount state | `BehavioralFeatureEngineer`, TRAIN fit | Label-free transactions update history in chronological order. |
| Composite frequency and history | `CompositeFeatureEngineer`, TRAIN fit | Frequency maps remain fixed; label-free transaction history updates. |
| Temporal target encoding | `TemporalTargetEncoder(alpha=50, initial_prior=0.5, label_delay_days=0)` | TRAIN-label history remains fixed. Same-timestamp TRAIN labels are excluded; incoming transactions never supply labels or update this state. |
| Tabular maps and scaler | `TrainingOnlyPreprocessor` and `StandardScaler`, TRAIN fit | Immutable after TRAIN. The ordered 467-feature selection is fixed. |
| Component models | Frozen TRAIN-only LightGBM and XGBoost configurations | Rebuilt once for artifact materialization; native model files are loaded for inference. |
| Ensemble and calibration | 0.8/0.2 blend; existing TRAIN-OOF Platt artifact | Immutable. |
| Decisions and optional rules | Balanced cost profile; `rules_frozen_config.json` | Balanced is primary with rules disabled. Rules are an optional frozen overlay and never change calibrated probability. |

The canonical bundle is immutable. A runtime instance owns its evolving non-label history; loading the bundle again resets that history. The terminal HOLDOUT predictions provide label-free golden outputs. HOLDOUT labels are not needed for artifactization or inference.
