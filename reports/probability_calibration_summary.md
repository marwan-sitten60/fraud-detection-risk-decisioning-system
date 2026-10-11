# Probability calibration summary

Calibration improves probability interpretation; it is not intended to improve ranking PR-AUC.
The raw ensemble overpredicts fraud on the temporal OOF evaluation rows.

## OOF source and integrity

The aligned source is reports/ensemble_oof_predictions.csv: 124,014 unique TRAIN rows, three chronological folds of 41,338 rows, complete binary targets and finite component probabilities. Frozen score = 0.8 LightGBM + 0.2 XGBoost. OOF identifiers belong to TRAIN and do not overlap external VALIDATION. Raw all-fold mean score = 0.074550; prevalence = 0.038875; range = [0.000005, 0.999976].

## Temporal method selection

Fit on Fold 1 then assess Fold 2; fit on Folds 1+2 then assess Fold 3. The raw baseline uses identical evaluation rows. Pooled Brier is primary; Log Loss and ten-bin equal-frequency ECE are secondary. ECE weights absolute bin mean-probability versus fraud-rate gaps by row count. The bins partition all rows, including tied scores. Neither external VALIDATION nor HOLDOUT participates in selection.

| method | evaluation_fold | fit_rows | evaluation_rows | brier | log_loss | ece | mean_probability | prevalence | calibration_gap | pr_auc | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 2 | 41338.000000 | 41338 | 0.026363 | 0.107436 | 0.032965 | 0.068081 | 0.035222 | 0.032859 | 0.579301 | 0.897137 |
| raw | 3 | 82676.000000 | 41338 | 0.034268 | 0.135178 | 0.046762 | 0.091463 | 0.046011 | 0.045452 | 0.631579 | 0.904517 |
| raw | pooled | nan | 82676 | 0.030316 | 0.121307 | 0.039711 | 0.079772 | 0.040616 | 0.039155 | 0.608317 | 0.901369 |
| platt | 2 | 41338.000000 | 41338 | 0.020791 | 0.088950 | 0.003287 | 0.037077 | 0.035222 | 0.001855 | 0.579301 | 0.897137 |
| platt | 3 | 82676.000000 | 41338 | 0.024938 | 0.104078 | 0.007847 | 0.050771 | 0.046011 | 0.004760 | 0.631579 | 0.904517 |
| platt | pooled | nan | 82676 | 0.022864 | 0.096514 | 0.005599 | 0.043924 | 0.040616 | 0.003308 | 0.608349 | 0.901322 |
| isotonic | 2 | 41338.000000 | 41338 | 0.020809 | 0.089653 | 0.003648 | 0.037106 | 0.035222 | 0.001884 | 0.561445 | 0.896308 |
| isotonic | 3 | 82676.000000 | 41338 | 0.025153 | 0.105260 | 0.006805 | 0.050916 | 0.046011 | 0.004905 | 0.617172 | 0.903967 |
| isotonic | pooled | nan | 82676 | 0.022981 | 0.097456 | 0.004616 | 0.044011 | 0.040616 | 0.003394 | 0.598370 | 0.899432 |

Selected: **platt** by lowest pooled Brier. Platt is simpler and continuous; isotonic introduced score ties and lower ranking sanity metrics. Selected mapping was then fitted on all TRAIN OOF labels and serialized at models/probability_calibrator.joblib. No class weights were used in calibration.

## External VALIDATION

| score | brier | log_loss | ece | mean_probability | prevalence | calibration_gap | pr_auc | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.024976 | 0.103692 | 0.039006 | 0.073348 | 0.034341 | 0.039006 | 0.615265 | 0.916224 |
| calibrated | 0.019493 | 0.082039 | 0.005434 | 0.036660 | 0.034341 | 0.002319 | 0.615265 | 0.916224 |

The rebuilt raw PR-AUC and ROC-AUC match the frozen champion reference to six decimals.

Raw-versus-calibrated reliability bins are stored in reports/probability_calibration_raw_bins.csv and reports/probability_calibration_calibrated_bins.csv. Each bin lists count, score range, mean predicted probability, and observed fraud rate.

| bin | count | score_min | score_max | mean_probability | fraud_rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 8859 | 0.000005 | 0.001240 | 0.000729 | 0.000452 |
| 2 | 8858 | 0.001240 | 0.002424 | 0.001807 | 0.001693 |
| 3 | 8858 | 0.002424 | 0.004088 | 0.003203 | 0.003613 |
| 4 | 8858 | 0.004088 | 0.006443 | 0.005202 | 0.004741 |
| 5 | 8858 | 0.006443 | 0.009829 | 0.008049 | 0.006999 |
| 6 | 8858 | 0.009830 | 0.014727 | 0.012138 | 0.008241 |
| 7 | 8858 | 0.014727 | 0.021915 | 0.018067 | 0.012305 |
| 8 | 8858 | 0.021916 | 0.033916 | 0.027325 | 0.015353 |
| 9 | 8858 | 0.033917 | 0.065151 | 0.046278 | 0.031045 |
| 10 | 8858 | 0.065154 | 0.997197 | 0.243810 | 0.258975 |

The calibration curve is at reports/figures/probability_calibration_curve.png. Calibrated bin points sit closer to the diagonal overall; the highest risk bin still has residual error. PR-AUC and ROC-AUC are ranking sanity checks, not selection criteria. The Platt transform is monotone, so within-period ranking is preserved apart from numerical precision.

## Limitations and production implications

Ten-bin ECE is a summary sensitive to binning and can conceal within-bin error. Temporal distribution shift remains possible. This is one external VALIDATION assessment and does not establish future reliability. A 0.50 score is not a production threshold; decision threshold design is a later phase. The calibrator must only be refreshed using matured historical labels during controlled retraining/calibration updates.

## Runtime, memory, and tracking

Notebook runtime: 229.7 seconds. Prepared matrix estimate: 905.7 MiB. MLflow experiment: fraud-detection-probability-calibration; tracking URI: sqlite:///C:/fraud-detection-system/model_comparison.db. Run IDs: {"raw": "6984e88918124c27a5c9e82ea98e9b89", "platt": "f7fe9f23b08c48978cb729bdecd73eba", "isotonic": "9763c7c5cd414fa3bb4abe43b5b5f707", "final": "4d7425b5daf14dc0b3ba6f5741c55397"}.

HOLDOUT remained untouched: no HOLDOUT access, transformation, prediction, summary, calibration, or evaluation.
