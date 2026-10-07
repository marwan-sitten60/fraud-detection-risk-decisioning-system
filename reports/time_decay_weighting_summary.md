# Time-Aware Sample Weighting

## Formula and normalization

For each fit portion, `age_days = (max_fit_TransactionDT - TransactionDT) / 86400` and
`raw_weight = exp(-ln(2) * age_days / half_life_days)`. Weights are divided by the raw
fit-set mean, producing mean one while preserving relative ratios. No lower floor is applied.
Only fit timestamps determine the reference time.

## TRAIN-only temporal CV

Three expanding chronological folds rebuild the 463-feature pipeline independently. The pipeline
contains existing frequency encodings, five historical amount features, and all 20 composite
identity features. The fixed LightGBM configuration remains unchanged across candidates.

| Strategy | Half-life | Fold PR-AUCs | Mean | Std | Mean ROC-AUC | Mean fit s |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| unweighted | none | 0.554229 / 0.574851 / 0.625544 | 0.584875 | 0.029965 | 0.906202 | 29.8 |
| half_life_90_days | 90 | 0.575379 / 0.556912 / 0.615404 | 0.582565 | 0.024414 | 0.903198 | 31.1 |
| half_life_60_days | 60 | 0.554720 / 0.568685 / 0.622962 | 0.582122 | 0.029435 | 0.903318 | 29.9 |
| half_life_30_days | 30 | 0.556336 / 0.556732 / 0.612671 | 0.575246 | 0.026464 | 0.900510 | 28.3 |

- Selected strategy: **unweighted**
- Selected half-life: **none**

## Winner fold weight statistics

| Fold | Minimum | Median | Maximum | Mean |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

Full-TRAIN weights: minimum **1.000000**, median
**1.000000**, maximum **1.000000**,
mean **1.000000**.

## External validation

- PR-AUC: **0.595387**
- ROC-AUC: **0.926655**
- Precision at 0.50: **0.538343**
- Recall at 0.50: **0.572321**
- F1 at 0.50: **0.554812**
- Confusion matrix: **((84046, 1493), (1301, 1741))**
- Previous preferred PR-AUC: **0.595387**
- Absolute improvement: **+0.000000**
- Relative improvement: **+0.00%**

## Runtime and memory

- Fold feature-preparation seconds: **41.7, 44.1, 49.8**
- Mean fit seconds by strategy: **{"unweighted": 29.8, "half_life_30_days": 28.3, "half_life_60_days": 29.9, "half_life_90_days": 31.1}**
- Final TRAIN/VALIDATION workflow: **101.9 seconds**
- Peak prepared matrix estimate: **889.7 MiB**
- Total notebook time before report write: **628.6 seconds**

## MLflow

- Experiment: `fraud-detection-time-weighting`
- **unweighted:** `789682d8e85c4fd295887d713c3a1556`
- **half_life_30_days:** `ca01d155d6a5495ca0d93c3b58f09aba`
- **half_life_60_days:** `1618fcb388f64b3bb1e4e4cc1c19c6b3`
- **half_life_90_days:** `087bcba5344a45fcb4295931a68a51ea`
- **Final winner:** `fd07db6771a14f6088c9d0b9177dfd9b`

## Scope confirmation

External VALIDATION was not used to select the weighting strategy and was evaluated once after
selection. HOLDOUT was never assigned, transformed, predicted, summarized, or evaluated. No new
features, tuning, calibration, threshold optimization, target encoding, ensembling, SHAP, or rules
are included.
