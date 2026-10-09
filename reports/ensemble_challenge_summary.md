# LightGBM-XGBoost Ensemble Challenge

## Component models and feature pipeline

Both models use the same **467** predictors: existing frequency encodings, five
past-only historical amount features, 20 composite identity features, and four composite
temporal target-encoded rates. Graph features and time-decay weighting are excluded.

- LightGBM: `{"class_weight": "balanced", "learning_rate": 0.1, "n_estimators": 400, "n_jobs": -1, "num_leaves": 63, "random_state": 42, "verbosity": -1}`
- XGBoost: `{"eval_metric": "aucpr", "learning_rate": 0.1, "max_depth": 7, "n_estimators": 300, "n_jobs": -1, "random_state": 42, "tree_method": "hist"}`, plus fit-period
  `scale_pos_weight = non_fraud / fraud` (fold values: 28.779150 / 28.579785 / 28.442862).

## Temporal OOF methodology

- Fold 1: fit 289,364, evaluate 41,338, DT after 7135483 through 8121450
- Fold 2: fit 330,702, evaluate 41,338, DT after 8121450 through 9247568
- Fold 3: fit 372,040, evaluate 41,338, DT after 9247568 through 10437996

All preprocessing was rebuilt from each fold's historical fit rows. Evaluation labels were used
only for scoring. The aligned OOF artifact contains 124,014 rows.

## Individual model CV metrics

| Model | Fold PR-AUCs | Mean PR-AUC | Std PR-AUC | Mean ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| LightGBM | 0.596947 / 0.576060 / 0.630908 | 0.601305 | 0.022603 | 0.890932 |
| XGBoost | 0.579794 / 0.550805 / 0.605440 | 0.578680 | 0.022318 | 0.884211 |

## Prediction correlation

| Fold | Rows | Pearson | Spearman |
| --- | ---: | ---: | ---: |
| 1 | 41338 | 0.892714 | 0.881451 |
| 2 | 41338 | 0.913881 | 0.898374 |
| 3 | 41338 | 0.932636 | 0.927224 |
| overall | 124014 | 0.916229 | 0.900851 |

- Material rank-difference threshold: **20%**
- Rows materially different: **11.92%**
- Fraud rows below LightGBM 0.50: **2,073**
- Mean XGBoost minus LightGBM score on those rows: **+0.049968**
- XGBoost score higher on those rows: **72.84%**
- XGBoost score at least 0.50 on those rows: **8.25%**

## Probability-blend search

| LightGBM / XGBoost | Fold PR-AUCs | Mean PR-AUC | Std PR-AUC | Mean ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| 1.0 / 0.0 | 0.596947 / 0.576060 / 0.630908 | 0.601305 | 0.022603 | 0.890932 |
| 0.9 / 0.1 | 0.600221 / 0.578867 / 0.631584 | 0.603558 | 0.021651 | 0.892699 |
| 0.8 / 0.2 | 0.602235 / 0.579301 / 0.631579 | 0.604372 | 0.021396 | 0.893230 |
| 0.7 / 0.3 | 0.603144 / 0.579061 / 0.630794 | 0.604333 | 0.021137 | 0.893203 |
| 0.6 / 0.4 | 0.603008 / 0.577594 / 0.629261 | 0.603288 | 0.021094 | 0.892827 |
| 0.5 / 0.5 | 0.601932 / 0.575403 / 0.627177 | 0.601504 | 0.021139 | 0.892175 |

- Selected OOF candidate: **0.8 / 0.2**
- Absolute gain over LightGBM: **+0.003066**
- Relative gain over LightGBM: **+0.51%**
- Ensemble strictly beat LightGBM: **True**

## External validation

- PR-AUC: **0.615265**
- ROC-AUC: **0.916224**
- Precision at 0.50: **0.588788**
- Recall at 0.50: **0.576594**
- F1 at 0.50: **0.582627**
- Confusion matrix: **((84314, 1225), (1288, 1754))**
- Difference from LightGBM external reference 0.612204: **+0.003061**

The existing LightGBM external reference is PR-AUC **0.612204**, ROC-AUC **0.914085**,
precision **0.600277**, recall **0.569691**, and F1 **0.584584**. These values did not
participate in OOF blend selection.

## Runtime, memory, and serving trade-off

- Fold preparation seconds: **51.354577 / 57.495770 / 72.437872**
- LightGBM fit/predict seconds: **23.616208 / 25.571467 / 28.502925**
- XGBoost fit/predict seconds: **27.189204 / 37.454374 / 39.562199**
- Conditional final TRAIN/VALIDATION seconds: **149.7**
- Peak prepared-matrix estimate: **905.7 MiB**
- Total notebook runtime before report write: **547.8 seconds**

An ensemble requires storing, loading, scoring, monitoring, and versioning two models. This
roughly doubles booster inference work and adds blend coordination.

**Recommendation:** Use the selected ensemble: its OOF gain is at least 0.001 PR-AUC.

## MLflow

- Experiment: `fraud-detection-ensemble-challenge`
- Tracking URI: `sqlite:///C:/fraud-detection-system/model_comparison.db`
- **model_LightGBM:** `34832a1924e540c69c7c0b243c666c06`
- **model_XGBoost:** `2f398c03db6e4ebf89be72339048a7e9`
- **blend-lgbm-1.0-xgb-0.0:** `b5bf98142fe64a518326111c4832f554`
- **blend-lgbm-0.9-xgb-0.1:** `039b728f98844d2e994d57b8960c5b54`
- **blend-lgbm-0.8-xgb-0.2:** `594489309b8448f3a49c39c8d823c762`
- **blend-lgbm-0.7-xgb-0.3:** `ee545684ab00447794df7dc4c44a5b25`
- **blend-lgbm-0.6-xgb-0.4:** `65db65cbe5384da698ad53c4aa18af5c`
- **blend-lgbm-0.5-xgb-0.5:** `714571575ee54a5fb36843bd790ef6b4`

## Leakage boundary

Ensemble weights were selected only from TRAIN temporal OOF predictions. External VALIDATION
did not influence selection and was evaluated only if an ensemble won. HOLDOUT was never
assigned, inspected, transformed, predicted, summarized, or evaluated.
