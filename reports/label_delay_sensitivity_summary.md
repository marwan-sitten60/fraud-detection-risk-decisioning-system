# Label-Delay Sensitivity Analysis

## Why delay matters and exact maturity rule

Fraud outcomes often mature after transactions are scored. For a query at `T` and delay `D`,
a historical FIT label contributes only when `historical_TransactionDT <= T - D * 86400`.
Own labels, future labels, same-timestamp labels, immature labels, and evaluation labels are
excluded. Labels are activated in timestamp batches. Alpha remains 50 and the no-history
fallback remains the fixed neutral prior 0.5.

**Label delay is an operational condition, not a tuned model parameter.**

## Fixed pipeline and leakage safeguards

The fixed 467-feature pipeline retains frequency encoding, five past-only amount
features, 20 composite identity features, and only four composite temporal target encodings.
Each fold rebuilds preprocessing from FIT history. Evaluation features use progressively
maturing FIT labels only; evaluation labels never update state. Models and 80/20 weight are fixed.

## Zero-day reproduction

The observed folds **0.602235 / 0.579301 / 0.631579** reproduce the reference
0.602235 / 0.579301 / 0.631579 within absolute tolerance 0.000005: **PASS**.

## Ensemble sensitivity

| Delay | Fold PR-AUCs | Mean | Std | Mean ROC-AUC | Absolute change | Relative change |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0d | 0.602235 / 0.579301 / 0.631579 | 0.604372 | 0.021396 | 0.893230 | +0.000000 | +0.00% |
| 7d | 0.610818 / 0.599545 / 0.647550 | 0.619304 | 0.020496 | 0.918588 | +0.014932 | +2.47% |
| 14d | 0.595537 / 0.579641 / 0.637583 | 0.604253 | 0.024444 | 0.917317 | -0.000118 | -0.02% |
| 30d | 0.570488 / 0.584447 / 0.634278 | 0.596404 | 0.027381 | 0.917387 | -0.007967 | -1.32% |

## Component-model sensitivity

| Delay | Model | Fold PR-AUCs | Mean | Std | Mean ROC-AUC |
| ---: | --- | --- | ---: | ---: | ---: |
| 0d | LightGBM | 0.596947 / 0.576060 / 0.630908 | 0.601305 | 0.022603 | 0.890932 |
| 0d | XGBoost | 0.579794 / 0.550805 / 0.605440 | 0.578680 | 0.022318 | 0.884211 |
| 0d | Ensemble 80/20 | 0.602235 / 0.579301 / 0.631579 | 0.604372 | 0.021396 | 0.893230 |
| 7d | LightGBM | 0.605944 / 0.597203 / 0.646505 | 0.616551 | 0.021479 | 0.914336 |
| 7d | XGBoost | 0.574999 / 0.561450 / 0.624963 | 0.587137 | 0.027313 | 0.913006 |
| 7d | Ensemble 80/20 | 0.610818 / 0.599545 / 0.647550 | 0.619304 | 0.020496 | 0.918588 |
| 14d | LightGBM | 0.590569 / 0.579019 / 0.631864 | 0.600484 | 0.022684 | 0.912657 |
| 14d | XGBoost | 0.561279 / 0.532775 / 0.623882 | 0.572645 | 0.038053 | 0.908621 |
| 14d | Ensemble 80/20 | 0.595537 / 0.579641 / 0.637583 | 0.604253 | 0.024444 | 0.917317 |
| 30d | LightGBM | 0.565252 / 0.580811 / 0.630333 | 0.592132 | 0.027749 | 0.913500 |
| 30d | XGBoost | 0.539669 / 0.552246 / 0.609934 | 0.567283 | 0.030593 | 0.906584 |
| 30d | Ensemble 80/20 | 0.570488 / 0.584447 / 0.634278 | 0.596404 | 0.027381 | 0.917387 |

## Label availability

Statistics are fold means. Representative matured fractions use the first, middle, and final
evaluation transaction times and divide matured FIT labels by all FIT labels.

| Delay | TRAIN fallback | EVAL fallback | Matured FIT labels (start / middle / end) | Mean composite support |
| ---: | ---: | ---: | --- | ---: |
| 0d | 0.0003% | 0.0000% | 100.00% / 100.00% / 100.00% | 300.9 |
| 7d | 8.4330% | 0.0000% | 93.04% / 99.60% / 100.00% | 294.4 |
| 14d | 17.1309% | 0.0000% | 86.42% / 92.76% / 98.76% | 277.1 |
| 30d | 41.0522% | 0.0000% | 71.16% / 77.17% / 82.83% | 227.5 |

## Operational interpretation

The composite-TE signal appears robust across these fixed scenarios: 7 days improved mean PR-AUC,
14 days was essentially unchanged, and 30 days produced a modest decline without a collapse. The
non-monotonic result may reflect regularization from excluding the freshest labels and is not
evidence that delaying labels is beneficial. Delay reduces the freshest labeled history and entity
support; the tables quantify the effect without selecting a preferred delay. The
historical external-validation PR-AUC 0.615265 is context only and was not recomputed.

Production target encoding must track event time, label-maturity time, state version, and the
same-timestamp batch rule. This offline simulation assumes labels become available at exactly
the configured fixed delay; real maturity is heterogeneous and should later be monitored.

## Runtime and memory

- Total runtime before report write: **1625.1 seconds**
- Peak prepared-matrix estimate: **745.9 MiB**
- Observed peak kernel working set: **about 6.1 GiB**
- Observed peak private-memory reservation: **about 12.0 GiB**
- Preparation seconds by delay: `{"0": [68.32893669995246, 77.7002613999648, 83.51699269999517], "7": [62.18507360003423, 73.23923330003163, 83.59554670000216], "14": [63.229912899958435, 72.4043027999578, 80.21005330001935], "30": [61.806555600021966, 71.61457600002177, 90.39770540001336]}`
- Combined model seconds by delay: `{"0": [56.13190149998991, 57.78039410000201, 58.817367700045], "7": [50.19555800000671, 59.47613460000139, 56.795446200005244], "14": [48.54892299999483, 56.84118970000418, 57.8005145000061], "30": [50.12429680000059, 57.94640920002712, 62.754074199998286]}`

## MLflow

- Experiment: `fraud-detection-label-delay-sensitivity`
- Tracking URI: `sqlite:///C:/fraud-detection-system/model_comparison.db`
- **0d:** `aa4e973d135e45899021e1aa95dbfbbd`
- **7d:** `c408fd42c0704fdfa7fe2f952711f695`
- **14d:** `2b9f665ea780452ea853b8af74ea5b25`
- **30d:** `77aa9cccf32040359be1f32813dc64b1`

## Scope confirmation

No model, blend weight, feature, delay, or threshold was selected. External VALIDATION and
HOLDOUT were not materialized, transformed, predicted, summarized, or evaluated.
