# LightGBM Temporal-CV Tuning

## Methodology and leakage protections

Three expanding folds are built inside TRAIN. Boundaries align to complete timestamps and
each fit period strictly precedes evaluation. Frequency maps, medians, scaling, historical
amount state, and graph state are rebuilt using each fold's fit history only. Fold evaluation
drives early stopping with a 50-round patience. External VALIDATION is not transformed or
scored until both configurations are selected. The final holdout is untouched.

| Fold | Fit rows | Evaluation rows | Fit DT max | Evaluation DT min | Evaluation DT max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 289364 | 41338 | 7135483 | 7135609 | 8121450 |
| 2 | 330702 | 41338 | 8121450 | 8121466 | 9247568 |
| 3 | 372040 | 41338 | 9247568 | 9247583 | 10437996 |

## Configurations tested

Eight curated parameter configurations were applied to both feature candidates (16 CV runs).

1. `{"class_weight": "balanced", "colsample_bytree": 1.0, "learning_rate": 0.1, "max_depth": -1, "metric": "average_precision", "min_child_samples": 20, "n_estimators": 500, "n_jobs": -1, "num_leaves": 63, "objective": "binary", "random_state": 42, "reg_alpha": 0.0, "reg_lambda": 0.0, "subsample": 1.0, "subsample_freq": 0, "verbosity": -1}`
2. `{"class_weight": "balanced", "colsample_bytree": 0.9, "learning_rate": 0.05, "max_depth": -1, "metric": "average_precision", "min_child_samples": 30, "n_estimators": 800, "n_jobs": -1, "num_leaves": 31, "objective": "binary", "random_state": 42, "reg_alpha": 0.1, "reg_lambda": 0.5, "subsample": 0.9, "subsample_freq": 1, "verbosity": -1}`
3. `{"class_weight": "balanced", "colsample_bytree": 0.9, "learning_rate": 0.05, "max_depth": -1, "metric": "average_precision", "min_child_samples": 30, "n_estimators": 800, "n_jobs": -1, "num_leaves": 63, "objective": "binary", "random_state": 42, "reg_alpha": 0.1, "reg_lambda": 0.5, "subsample": 0.9, "subsample_freq": 1, "verbosity": -1}`
4. `{"class_weight": "balanced", "colsample_bytree": 0.8, "learning_rate": 0.03, "max_depth": -1, "metric": "average_precision", "min_child_samples": 50, "n_estimators": 1200, "n_jobs": -1, "num_leaves": 31, "objective": "binary", "random_state": 42, "reg_alpha": 0.5, "reg_lambda": 1.0, "subsample": 0.8, "subsample_freq": 1, "verbosity": -1}`
5. `{"class_weight": "balanced", "colsample_bytree": 0.8, "learning_rate": 0.03, "max_depth": 8, "metric": "average_precision", "min_child_samples": 40, "n_estimators": 1000, "n_jobs": -1, "num_leaves": 127, "objective": "binary", "random_state": 42, "reg_alpha": 0.5, "reg_lambda": 1.0, "subsample": 0.8, "subsample_freq": 1, "verbosity": -1}`
6. `{"class_weight": "balanced", "colsample_bytree": 0.9, "learning_rate": 0.03, "max_depth": 8, "metric": "average_precision", "min_child_samples": 50, "n_estimators": 1200, "n_jobs": -1, "num_leaves": 63, "objective": "binary", "random_state": 42, "reg_alpha": 1.0, "reg_lambda": 2.0, "subsample": 0.8, "subsample_freq": 1, "verbosity": -1}`
7. `{"class_weight": "balanced", "colsample_bytree": 0.8, "learning_rate": 0.1, "max_depth": 6, "metric": "average_precision", "min_child_samples": 40, "n_estimators": 500, "n_jobs": -1, "num_leaves": 31, "objective": "binary", "random_state": 42, "reg_alpha": 0.0, "reg_lambda": 1.0, "subsample": 1.0, "subsample_freq": 0, "verbosity": -1}`
8. `{"class_weight": "balanced", "colsample_bytree": 0.85, "learning_rate": 0.07, "max_depth": 8, "metric": "average_precision", "min_child_samples": 30, "n_estimators": 700, "n_jobs": -1, "num_leaves": 63, "objective": "binary", "random_state": 42, "reg_alpha": 0.1, "reg_lambda": 1.0, "subsample": 0.9, "subsample_freq": 1, "verbosity": -1}`

## TRAIN-only temporal CV results

| Feature set | Config | Fold PR-AUC | Mean | Std | Mean ROC-AUC | Iterations |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| candidate_a | 3 | 0.565115 / 0.576368 / 0.636654 | 0.592712 | 0.031409 | 0.915088 | 539 / 563 / 800 |
| candidate_a | 6 | 0.565500 / 0.564920 / 0.628697 | 0.586372 | 0.029929 | 0.916726 | 1104 / 920 / 1194 |
| candidate_a | 5 | 0.570789 / 0.557697 / 0.623708 | 0.584065 | 0.028537 | 0.914372 | 990 / 796 / 854 |
| candidate_a | 8 | 0.549840 / 0.553882 / 0.621096 | 0.574939 | 0.032679 | 0.913039 | 210 / 287 / 475 |
| candidate_a | 4 | 0.537472 / 0.555007 / 0.621840 | 0.571440 | 0.036350 | 0.916148 | 701 / 858 / 967 |
| candidate_a | 2 | 0.534405 / 0.553608 / 0.623383 | 0.570466 | 0.038231 | 0.914998 | 430 / 665 / 800 |
| candidate_a | 1 | 0.544507 / 0.545635 / 0.604548 | 0.564897 | 0.028042 | 0.911508 | 192 / 142 / 113 |
| candidate_a | 7 | 0.527028 / 0.532877 / 0.622750 | 0.560885 | 0.043810 | 0.908583 | 155 / 337 / 499 |
| candidate_b_graph | 3 | 0.549036 / 0.568634 / 0.636428 | 0.584700 | 0.037443 | 0.917309 | 239 / 455 / 800 |
| candidate_b_graph | 1 | 0.562835 / 0.554963 / 0.626716 | 0.581505 | 0.032130 | 0.909486 | 170 / 250 / 499 |
| candidate_b_graph | 5 | 0.559518 / 0.560141 / 0.622494 | 0.580718 | 0.029541 | 0.914669 | 872 / 680 / 999 |
| candidate_b_graph | 6 | 0.543569 / 0.562328 / 0.617363 | 0.574420 | 0.031316 | 0.917228 | 389 / 964 / 727 |
| candidate_b_graph | 8 | 0.548139 / 0.543867 / 0.626501 | 0.572836 | 0.037987 | 0.910370 | 480 / 366 / 631 |
| candidate_b_graph | 2 | 0.538069 / 0.544414 / 0.619298 | 0.567260 | 0.036887 | 0.914136 | 437 / 380 / 670 |
| candidate_b_graph | 4 | 0.533456 / 0.553771 / 0.605514 | 0.564247 | 0.030336 | 0.915947 | 555 / 964 / 573 |
| candidate_b_graph | 7 | 0.515265 / 0.524840 / 0.610775 | 0.550293 | 0.042945 | 0.907087 | 202 / 282 / 368 |

- Candidate A best configuration: **3**
- Candidate A CV parameters: `{"class_weight": "balanced", "colsample_bytree": 0.9, "learning_rate": 0.05, "max_depth": -1, "metric": "average_precision", "min_child_samples": 30, "n_estimators": 800, "n_jobs": -1, "num_leaves": 63, "objective": "binary", "random_state": 42, "reg_alpha": 0.1, "reg_lambda": 0.5, "subsample": 0.9, "subsample_freq": 1, "verbosity": -1}`
- Candidate A fold iterations: **(539, 563, 800)**; full-TRAIN refit uses
  **563** estimators.
- Candidate B best configuration: **3**
- Candidate B CV parameters: `{"class_weight": "balanced", "colsample_bytree": 0.9, "learning_rate": 0.05, "max_depth": -1, "metric": "average_precision", "min_child_samples": 30, "n_estimators": 800, "n_jobs": -1, "num_leaves": 63, "objective": "binary", "random_state": 42, "reg_alpha": 0.1, "reg_lambda": 0.5, "subsample": 0.9, "subsample_freq": 1, "verbosity": -1}`
- Candidate B fold iterations: **(239, 455, 800)**; full-TRAIN refit uses
  **455** estimators.

## One-time external validation results

| Candidate | Features | PR-AUC | ROC-AUC | Precision | Recall | F1 | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| candidate_a | 443 | 0.593452 | 0.927279 | 0.470946 | 0.610125 | 0.531577 | `((83454, 2085), (1186, 1856))` |
| candidate_b_graph | 451 | 0.588683 | 0.926108 | 0.414598 | 0.636752 | 0.502204 | `((82804, 2735), (1105, 1937))` |

Previous untuned results were `0.593938` for Candidate A and `0.594611` for Candidate B.
The tuned winner changes the previous overall best by **-0.001159**.

## Simple versus graph trade-off

Graph features do not improve tuned validation PR-AUC; the simpler 443-feature model is preferable on both ranking performance and operational complexity.

Graph feature generation previously required 60.8 seconds and sparse relationship state;
Candidate A avoids that operational dependency.

## MLflow

- Experiment: `fraud-detection-lightgbm-tuning`
- Candidate A winning CV run: `3a9efa1f48b44b8f9fcd14769843acfe`
- Candidate B winning CV run: `abf85d6a1913406ca39103dc6f478c83`
- Candidate A final-refit run: `7cea2e213e5b40a39622f1fe259f3152`
- Candidate B final-refit run: `6da1a5946cf244f2bc68ceeb2011f229`

## Runtime and memory

- Candidate A CV workflow: **994.8 seconds**
- Candidate B CV workflow: **768.3 seconds**
- Total experiment before report write: **1905.1 seconds**
- Largest simultaneously prepared matrix pair: **863.6 MiB**
- Matrices are prepared one fold at a time and scaling statistics are learned in batches.

No calibration, threshold optimization, SHAP, rules, registry promotion, or holdout evaluation
is performed.
