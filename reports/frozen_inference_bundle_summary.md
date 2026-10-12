# Frozen inference bundle and runtime

This is the exact frozen **TRAIN-only** system evaluated in the terminal HOLDOUT study, materialized as loadable software artifacts. It is not a retrained production model. The terminal HOLDOUT result remains locked. This phase used the saved HOLDOUT predictions only as a **label-free software regression reference** and did not read HOLDOUT labels or change model, feature, calibration, rule, threshold, or cost semantics.

## Architecture and dependency audit

The [dependency map](inference_runtime_dependency_map.md) identifies what one prediction needs. Notebook 18 had reconstructed the fitted boosters, frequency maps, non-label histories, target-encoding history, preprocessor, and scaler in memory; only the Platt calibrator and frozen rules were previously saved. `scripts/build_frozen_bundle.py` fitted the missing components on the original 413,378 TRAIN rows once with the frozen parameters. The runtime orchestrates the existing feature, model, calibration, cost, and rule implementations. Normal inference executes no notebook, training step, or raw TRAIN CSV read.

## Bundle inventory and integrity

Bundle: `models/frozen_v1/`, **24,482,230 bytes** (about 23.35 MiB). Ordered 467-feature SHA-256: `65c0c29dffb0132229d666128374ef419edf5f19b0f3613b9a262385b84c79d1`.

| File | Purpose | Format |
| --- | --- | --- |
| `lightgbm.txt` | TRAIN-only LightGBM booster | Native LightGBM text |
| `xgboost.json` | TRAIN-only XGBoost booster | Native XGBoost JSON |
| `feature_state.joblib` | TRAIN-fitted frequency maps, behavioral/composite histories, temporal target encoding, tabular preprocessing, scaler | Joblib for pandas/sklearn/project state |
| `platt_calibrator.joblib` | Unchanged TRAIN-OOF Platt artifact | Existing joblib artifact |
| `rules_frozen_config.json` | Unchanged optional rules and thresholds | JSON |
| `policy.json` | 0.8/0.2 blend, Balanced costs, rules-disabled default, zero-day delay, TRAIN-only scope | JSON |
| `feature_contract.json` | Ordered raw and model feature names | JSON |
| `artifact_manifest.json` | Version, parameters, software versions, paths, serializers, ordered-feature hash, payload SHA-256 values | JSON |
| `artifact_manifest.sha256` | Detached SHA-256 of the manifest | Text |

The manifest covers every payload file; the detached checksum covers the manifest itself. A checksum file cannot meaningfully contain its own hash. Loading verifies all hashes before deserializing state. Native boosters reproduced their corresponding fitted classifier scores exactly on 100 TRAIN rows after saving and reloading. The bundle contains no raw TRAIN CSV and no HOLDOUT labels.

Verified SHA-256 checksums:

| Bundle file | SHA-256 |
| --- | --- |
| `feature_state.joblib` | `2a7f86d6d4b417b4c948b0e2fa49449d8e06e147fe330d9f6c957cca00197af5` |
| `lightgbm.txt` | `201cb1747d25b4dc888269fb41cd389f989f2545a5ee6a6b3aab273d2da462cd` |
| `xgboost.json` | `2c13ed79860cfc593623d09932c135b6b4ed0cc34502f983a7bd089218c17a6c` |
| `platt_calibrator.joblib` | `3503bf66161b5152e9de4d638b9e7932d581fc371abd6a45b6969ba6c8134f6e` |
| `rules_frozen_config.json` | `3a4380831e177828c82a1a118259f83e354221fa52205b41d38aa21b58febb94` |
| `policy.json` | `3685f31c1af31acfb68a98d4003164fc19051b938dfc37fa9503587da669470c` |
| `feature_contract.json` | `047700d867cf360770bd0c419a0d672dd702a4a63a825569bede7897b4b54100` |
| `artifact_manifest.json` | `f2b3ac5a226b0f419f0b35f588d4e93572966d5459aea75df30e0252c71a7068` (stored in `artifact_manifest.sha256`) |

## Input and feature contract

The caller supplies **433 ordered raw transaction and identity columns** listed in `feature_contract.json`, including `TransactionID`, `TransactionDT`, and `TransactionAmt`. Missing identity observations remain missing values in their required columns. The runtime rejects `isFraud` and `y_true`, missing or extra raw columns, invalid numeric values, duplicate IDs, nonfinite IDs/times/amounts, negative amounts, and rows out of `TransactionDT` then `TransactionID` order. It does not silently reorder or repair input. Frequency, historical amount, composite identity, and target-encoding fields are derived internally; callers must not provide them as substitutes for raw fields. The caller's frame is not mutated.

## Historical and label-state semantics

Base frequency maps, composite frequency maps, numeric medians, scaler, models, calibration, policy, and rule thresholds are immutable after TRAIN. The behavioral amount and composite-identity histories start with TRAIN and incorporate incoming **label-free** transactions in chronological order. The default `predict_batch` and `predict_one` update only an instance's non-label history. `update_state=False` scores without retaining incoming rows; `snapshot_state`, `restore_state`, and `reset_state` make replay explicit. Bundle files are never mutated by scoring.

The temporal target encoder remains fixed to legally available TRAIN labels with alpha 50, prior 0.5, zero-day delay, past-only ordering, and same-timestamp isolation. Inference rows cannot provide labels and never update target-encoding label state. There is no new delayed-label ingestion policy in Phase 1; such a capability would require a separately specified future phase. The optional rules overlay uses only frozen TRAIN frequency maps and current label-free model signals. Rules are **disabled by default** for the primary Balanced policy and cannot change calibrated probability.

## Load and inference procedure

```python
from fraud_detection.inference import FrozenFraudInferenceRuntime

runtime = FrozenFraudInferenceRuntime.load("models/frozen_v1")
result = runtime.predict_batch(label_free_chronological_rows)
optional = runtime.predict_batch(other_later_rows, include_rules=True)
```

Each result includes the transaction ID, component scores, 0.8/0.2 ensemble score, calibrated probability, primary decision, per-action expected costs, and bundle version. Optional scoring adds final decision and triggered rule IDs. Use a new runtime instance or `reset_state()` to replay from the original TRAIN state.

## Golden parity and determinism

A fresh process loaded only the saved bundle and a 512-row **chronological prefix** beginning at the original HOLDOUT boundary; it did not access raw TRAIN CSVs. Its 467-feature matrix was **bit-identical** to the established full-TRAIN research feature builder on those rows. Against the locked label-free HOLDOUT predictions, maximum absolute score differences were LightGBM `7.3277e-12`, XGBoost `1.11e-16`, ensemble `5.8621e-12`, and calibrated probability `4.6075e-12`. The bound is `1e-11` absolute with zero relative tolerance, following diagnosis of exact feature parity and picounit-level LightGBM variation from independent frozen artifact materialization. **All 512 primary decisions, optional-rule decisions, and rule triggers matched exactly.** No tolerance was applied to decisions.

Two independently loaded runtimes produced exactly equal scores and actions. A three-row chronological `predict_batch` matched repeated `predict_one` calls exactly, and `reset_state()` reproduced the starting-state predictions within the same numerical gate. The separate [parity and benchmark record](frozen_runtime_parity_benchmark.json) contains measured differences and timings. No earlier row-level VALIDATION score artifact exists for an additional direct comparison; the established VALIDATION reference remains PR-AUC 0.615265 and ROC-AUC 0.916224. This phase made no new model-performance claim.

## Local memory and timing

For the 512-row prefix on this machine: bundle load **4.18 s**; feature generation **20.33 s**; model inference **0.010 s**; calibration and decisioning **0.053 s**; total batch **20.39 s**, about **25.1 rows/s**. Process resident memory was approximately **272 MB** before loading, **622 MB** after loading, and **659 MB** after inference. These are snapshots, not measured peaks or a production SLA. The approved target-encoder column projection remains in place; no wide-frame copy was reintroduced.

## Scope and limitations

The runtime currently accepts the full 433-column merged raw schema and expects callers to supply transaction and identity fields already joined by `TransactionID`. It retains full non-label TRAIN history in memory and recomputes historical features for each batch, so repeated single-row scoring is slower than a batch. This phase provides no service endpoint, database, monitoring, retraining, deployment, or newly defined label-update process. FastAPI work would need a request adapter for this input contract, runtime lifecycle and state ownership, and error-to-response mapping; none was started here.
