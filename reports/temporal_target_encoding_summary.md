# Leakage-Safe Temporal Target Encoding

## Encoded entities and features

- `card1_target_encoded_fraud_rate`
- `card2_target_encoded_fraud_rate`
- `addr1_target_encoded_fraud_rate`
- `P_emaildomain_target_encoded_fraud_rate`
- `R_emaildomain_target_encoded_fraud_rate`
- `DeviceInfo_target_encoded_fraud_rate`
- `card1_card2_addr1_target_encoded_fraud_rate`
- `card1_addr1_p_emaildomain_target_encoded_fraud_rate`
- `card1_card2_addr1_productcd_target_encoded_fraud_rate`
- `card1_addr1_deviceinfo_target_encoded_fraud_rate`

Exactly 10 target-encoded predictors are available: six base entities and four existing
composite identities. No support count or fraud count is exposed to the model.

## Formula, prior behavior, and missing values

`encoded_rate = (fraud_count + 50 * historical_global_prior) / (labeled_count + 50)`.
The earliest timestamp uses the explicit fixed prior **0.5**, because no historical labels exist.
Later TRAIN timestamps use the prior from strictly earlier labeled timestamps. Frozen evaluation
maps use the fit-derived global fraud rate for unseen entities. Base missing values use the
deterministic `<MISSING>` bucket; composites reuse length-prefixed keys and the `M` component.

## Temporal methodology and leakage audit

TRAIN features are timestamp-batched temporal OOF values. Rows at time `T` use only labels where
`TransactionDT < T`; an entire timestamp updates state only after every row at that timestamp is
encoded. Evaluation uses frozen fit-only maps and ignores evaluation labels.

- own_target_independent: **PASS**
- future_target_invariant: **PASS**
- same_timestamp_isolated: **PASS**
- evaluation_label_independent: **PASS**

## TRAIN-only temporal CV

| Feature set | TE entities | Features | Fold PR-AUCs | Mean | Std | Mean ROC-AUC |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| baseline_plus_composite_target_encoding | 4 | 467 | 0.596947 / 0.576060 / 0.630908 | 0.601305 | 0.022603 | 0.890932 |
| baseline_plus_all_target_encoding | 10 | 473 | 0.599215 / 0.575731 / 0.617945 | 0.597630 | 0.017270 | 0.891550 |
| baseline | 0 | 463 | 0.554229 / 0.574851 / 0.625544 | 0.584875 | 0.029965 | 0.906202 |
| baseline_plus_base_target_encoding | 6 | 469 | 0.576419 / 0.541327 / 0.612933 | 0.576893 | 0.029235 | 0.895151 |

- Selected configuration: **baseline_plus_composite_target_encoding**
- Selected feature count: **467**

## External validation

- PR-AUC: **0.612204**
- ROC-AUC: **0.914085**
- Precision at 0.50: **0.600277**
- Recall at 0.50: **0.569691**
- F1 at 0.50: **0.584584**
- Confusion matrix: **((84385, 1154), (1309, 1733))**
- Previous preferred PR-AUC: **0.595387**
- Absolute improvement: **+0.016817**
- Relative improvement: **+2.82%**

## Runtime and memory

- Fold preparation seconds: **52.7, 68.3, 69.8**
- Mean model-fit seconds: **{"baseline": 29.7, "baseline_plus_base_target_encoding": 38.7, "baseline_plus_composite_target_encoding": 33.7, "baseline_plus_all_target_encoding": 36.1}**
- Final TRAIN/VALIDATION workflow: **135.2 seconds**
- Peak prepared matrix estimate: **905.7 MiB**
- Total notebook time before report write: **777.0 seconds**

## MLflow

- Experiment: `fraud-detection-temporal-target-encoding`
- **baseline:** `ae23ee6afa0e44e8b42d7c69c0bdbf9b`
- **baseline_plus_base_target_encoding:** `1d1e6123d8fc460eac7f0ed9c311b6b4`
- **baseline_plus_composite_target_encoding:** `730f6ac2d90a4e0caec77a9688f89641`
- **baseline_plus_all_target_encoding:** `671aa5d734ef4073b8d17e966e8b43fe`
- **Final winner:** `8ddc6a38eed74be2a071b43b98d3c188`

## Production implications and limitations

Target-encoding maps may only be refreshed using historical labels available during retraining
or a controlled feature-state update. Label delay, state versioning, and timestamp-batch semantics
must be preserved in production. Alpha and the fixed earliest prior were not tuned.

External VALIDATION was not used for feature selection and was evaluated once. HOLDOUT was never
assigned, transformed, predicted, summarized, or evaluated. No weighting, graph features, tuning,
calibration, threshold optimization, ensembling, SHAP, or rules are included.
