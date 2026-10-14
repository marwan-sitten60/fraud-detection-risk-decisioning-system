# Baseline Model Comparison

## Experimental design

Three baselines were fitted on the chronological training split and selected on the immediately
following validation split: Logistic Regression, LightGBM, and XGBoost. The holdout set was not
used for fitting, preprocessing, parameter selection, feature selection, or evaluation.

Numeric medians and categorical frequencies were learned from training data only. Categories not
seen during training map to zero frequency. `isFraud` and `TransactionID` were excluded;
`TransactionDT` remains as relative time. No target encoding, SMOTE, or validation resampling was
used.

Validation fraud prevalence and no-skill PR-AUC: **0.034341**

## Parameter grids

- **Logistic Regression:** `{"C": [1.0]}`
- **LightGBM:** `{"learning_rate": [0.03, 0.1], "n_estimators": [400], "num_leaves": [31, 63]}`
- **XGBoost:** `{"learning_rate": [0.03, 0.1], "max_depth": [4, 7], "n_estimators": [300]}`

The boosting grids are deliberately limited to four configurations each because the dataset has
413k training rows and 432 baseline predictors. Logistic Regression uses one full-data baseline
configuration because repeated SAGA fits were disproportionately expensive; no rows were sampled.

## Best validation result per model

| Model | Best parameters | PR-AUC | ROC-AUC | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | `{"learning_rate": 0.1, "n_estimators": 400, "num_leaves": 63}` | 0.587317 | 0.921482 | 0.572638 | 0.533859 | 0.552569 |
| XGBoost | `{"learning_rate": 0.1, "max_depth": 7, "n_estimators": 300}` | 0.561597 | 0.913651 | 0.420699 | 0.597304 | 0.493683 |
| Logistic Regression | `{"C": 1.0}` | 0.367155 | 0.836731 | 0.090270 | 0.776463 | 0.161737 |

## Winner

**BEST VALIDATION MODEL: LightGBM**

It won because it achieved the highest validation PR-AUC (0.587317), the primary
metric for this imbalanced fraud-ranking problem. Threshold metrics use 0.50 only and have not been
optimized for business cost.

## MLflow

- Experiment: `fraud-detection-model-comparison`
- Tracking URI: `sqlite:///./model_comparison.db`
- **LightGBM:** best run `b9c11c3ae13a49afbfc8e4342eff7a81`
- **XGBoost:** best run `1e0ad5e44daf4081a17620902b55efc6`
- **Logistic Regression:** best run `723edaee63bc40de9fc325135877a1d4`

Every grid configuration logs parameters, row counts, feature count, ranking metrics, and threshold
metrics. The best fitted estimator for each model is logged to its winning run.

## Known limitations

- This is baseline preprocessing, not final feature engineering.
- Frequency encoding can lose category-specific information, especially for the linear model.
- The logged estimator artifacts expect already-preprocessed matrices; preprocessing is not yet
  packaged into a production inference pipeline.
- Probabilities are not calibrated and the 0.50 threshold is not cost-optimized.
- No holdout metrics are reported. The holdout remains reserved for final evaluation and later
  production simulation.
