# CatBoost Native-Categorical Challenger

## Why CatBoost and feature strategy

CatBoost preserves category identity and learns ordered categorical statistics. The candidate
uses 437 raw-plus-historical-amount features, including
37 native categorical features. Missing categories use
__MISSING__; numeric missing values remain native. isFraud and TransactionID are excluded.
Class imbalance uses auto_class_weights=Balanced consistently.

## Temporal CV and configurations

Three expanding TRAIN-only folds preserve timestamps. Each fold rebuilds historical amount
state. Evaluation labels are used only for fold-local early stopping and metrics. External
VALIDATION is untouched until winner selection.

1. {"allow_writing_files": false, "auto_class_weights": "Balanced", "bagging_temperature": 1.0, "border_count": 64, "depth": 6, "eval_metric": "PRAUC:type=Classic", "iterations": 400, "l2_leaf_reg": 3.0, "learning_rate": 0.08, "loss_function": "Logloss", "max_ctr_complexity": 1, "one_hot_max_size": 10, "random_seed": 42, "random_strength": 1.0, "task_type": "CPU", "thread_count": -1}
2. {"allow_writing_files": false, "auto_class_weights": "Balanced", "bagging_temperature": 0.5, "border_count": 64, "depth": 7, "eval_metric": "PRAUC:type=Classic", "iterations": 600, "l2_leaf_reg": 5.0, "learning_rate": 0.05, "loss_function": "Logloss", "max_ctr_complexity": 1, "one_hot_max_size": 10, "random_seed": 42, "random_strength": 0.5, "task_type": "CPU", "thread_count": -1}
3. {"allow_writing_files": false, "auto_class_weights": "Balanced", "bagging_temperature": 1.0, "border_count": 64, "depth": 8, "eval_metric": "PRAUC:type=Classic", "iterations": 500, "l2_leaf_reg": 8.0, "learning_rate": 0.06, "loss_function": "Logloss", "max_ctr_complexity": 1, "one_hot_max_size": 10, "random_seed": 42, "random_strength": 1.0, "task_type": "CPU", "thread_count": -1}
4. {"allow_writing_files": false, "auto_class_weights": "Balanced", "bagging_temperature": 0.0, "border_count": 64, "depth": 6, "eval_metric": "PRAUC:type=Classic", "iterations": 700, "l2_leaf_reg": 10.0, "learning_rate": 0.04, "loss_function": "Logloss", "max_ctr_complexity": 1, "one_hot_max_size": 10, "random_seed": 42, "random_strength": 0.5, "task_type": "CPU", "thread_count": -1}

## TRAIN-only CV results

| Config | Fold PR-AUC | Mean | Std | Mean ROC-AUC | Iterations |
| ---: | --- | ---: | ---: | ---: | --- |
| 3 | 0.569977 / 0.549772 / 0.625724 | 0.581825 | 0.032119 | 0.914958 | 455 / 452 / 494 |
| 2 | 0.565204 / 0.546050 / 0.618580 | 0.576611 | 0.030689 | 0.915253 | 511 / 574 / 600 |
| 4 | 0.553634 / 0.534687 / 0.601654 | 0.563325 | 0.028185 | 0.910760 | 698 / 699 / 556 |
| 1 | 0.547806 / 0.529372 / 0.610469 | 0.562549 | 0.034710 | 0.910348 | 376 / 390 / 399 |

- Best configuration: 3
- Best parameters: {"allow_writing_files": false, "auto_class_weights": "Balanced", "bagging_temperature": 1.0, "border_count": 64, "depth": 8, "eval_metric": "PRAUC:type=Classic", "iterations": 500, "l2_leaf_reg": 8.0, "learning_rate": 0.06, "loss_function": "Logloss", "max_ctr_complexity": 1, "one_hot_max_size": 10, "random_seed": 42, "random_strength": 1.0, "task_type": "CPU", "thread_count": -1}
- Final full-TRAIN iterations: 455

## External validation

- PR-AUC: 0.566402
- ROC-AUC: 0.918453
- Precision at 0.50: 0.264790
- Recall at 0.50: 0.726824
- F1 at 0.50: 0.388167
- Confusion matrix: ((79400, 6139), (831, 2211))

## LightGBM comparison and conclusion

- LightGBM reference PR-AUC: 0.593452
- CatBoost absolute change: -0.027050
- CatBoost relative change: -4.56%
- CatBoost does not beat LightGBM; LightGBM remains preferred.

CatBoost avoids frequency encoding but requires categorical schema handling and heavier CPU
inference. LightGBM uses a simpler all-numeric matrix.

## MLflow

- Experiment: fraud-detection-catboost-challenge
- Winning CV run: 3c56b65db5e246009479a099497039c9
- Final-refit run: ef99fdf63da74675a299334d92284ce1

## Runtime, memory, and limitations

- CV workflow: 4381.1 seconds
- Total experiment before report write: 4635.5 seconds
- Largest prepared fold pair: 2203.6 MiB
- Only four curated configurations were tested. Threshold 0.50 is a reference only.
- No calibration, SHAP, rules, ensembling, or holdout evaluation is performed.
