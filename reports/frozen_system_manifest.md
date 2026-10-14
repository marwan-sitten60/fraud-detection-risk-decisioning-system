# Frozen system manifest — pre-HOLDOUT lock

This is the exact system for any later HOLDOUT evaluation. This phase has not accessed HOLDOUT.

## Primary frozen evaluation policy

The primary final HOLDOUT evaluation must use the **rules-disabled** champion policy:

- 467-feature pipeline.
- 80% LightGBM + 20% XGBoost ensemble.
- Platt probability calibration.
- Frozen Balanced expected-cost decisioning.
- The illustrative Rules Engine is implemented and frozen, but **DISABLED BY DEFAULT** for the primary final HOLDOUT evaluation.

On TRAIN OOF, enabling the illustrative rules increased modeled fraud transaction capture from 69.62% to 72.03%, while increasing review workload and illustrative modeled expected cost from 169,003.97 to 171,824.46. The rules engine is therefore retained as an optional operational overlay, outside the primary champion policy.

Rules may be reported separately as an optional post-hoc operational overlay if needed. HOLDOUT results must not be used to retune the rules or their thresholds. The frozen rule definitions, thresholds, model, calibration, features, ensemble, and business-cost configuration remain unchanged.

## Final model fit scope

- LightGBM: rebuild and fit on **TRAIN only**.
- XGBoost: rebuild and fit on **TRAIN only**.
- Feature-state fitting: **TRAIN only**, following the existing frozen temporal inference semantics.
- Platt calibrator: use the existing frozen **TRAIN-OOF-derived** artifact; do not retrain or replace it.
- Ensemble weights: **0.8 LightGBM / 0.2 XGBoost**, unchanged.
- Balanced expected-cost business decisioning: unchanged.
- Illustrative Rules Engine: frozen and **disabled by default** for the primary HOLDOUT evaluation.

Do not refit boosters or feature state on TRAIN + VALIDATION for this terminal evaluation. Preserve all frozen hyperparameters, calibration, rules, thresholds, and business costs. No further VALIDATION-based tuning is permitted.

## Features and preprocessing

- Count: **467**, selected in `notebooks/17_rules_engine.ipynb` from `prepare_engineered_matrices` using the established feature-name selection.
- Retained families: raw transaction/identity, base frequency encodings, five historical amount features, 20 composite identity frequency/history features, and four composite temporal target encodings. No graph features. No SHAP inputs.
- Preprocessing: fold-local frequency maps, strict past-only history, training-only tabular preprocessing and StandardScaler. TRAIN full-fit scaling batch size = len(TRAIN). Full TRAIN reference for validation composite frequency.
- Four target-encoding entities: card1_card2_addr1, card1_addr1_p_emaildomain, card1_card2_addr1_productcd, card1_addr1_deviceinfo. Alpha 50; neutral prior 0.5; historical FIT labels only; same-timestamp labels excluded; frozen final-model label delay **0 days**. The separate label-delay study made no model selection.

## Models, blend, and calibration

- LightGBM `LGBMClassifier(learning_rate=0.1, n_estimators=400, num_leaves=63, class_weight='balanced', random_state=42, n_jobs=-1, verbosity=-1)`.
- XGBoost `XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=300, tree_method='hist', eval_metric='aucpr', random_state=42, n_jobs=-1, scale_pos_weight=TRAIN_negative/TRAIN_positive)`. Full TRAIN scale_pos_weight = 27.434310084.
- Probability blend: **0.8 LightGBM + 0.2 XGBoost**. External VALIDATION PR-AUC 0.615265; ROC-AUC 0.916224.
- Platt calibrator: `models/probability_calibrator.joblib`; SHA-256 `3503bf66161b5152e9de4d638b9e7932d581fc371abd6a45b6969ba6c8134f6e`. TRAIN OOF selection in `reports/probability_calibration_summary.md`.
- Boosters are rebuilt with fixed parameters and the established feature pipeline, not serialized as model artifacts. Rebuild code is in this executed notebook and matches notebook 16.

## Business decision and illustrative rules

- Frozen Balanced scenario: {'fraud_loss_multiplier': 1.0, 'manual_review_cost': 3.0, 'review_fraud_detection_rate': 0.8, 'legitimate_review_friction_cost': 1.0, 'false_decline_cost': 25.0, 'block_operational_cost': 0.5}. Expected-cost formulas and decision tie order APPROVE, REVIEW, BLOCK are in `src/fraud_detection/decisioning/costs.py` and `policy.py`.
- Frozen rules: R01 Extreme risk and high amount, R02 Component disagreement, R03 Rare composite identity. Predicates, priorities, actions, reasons, and thresholds are in `reports/rules_frozen_config.json`; SHA-256 `3a4380831e177828c82a1a118259f83e354221fa52205b41d38aa21b58febb94`. Implementation: `src/fraud_detection/decisioning/rules.py`.
- HARD BLOCK > SOFT REVIEW > base. All matches recorded. Calibrated score unchanged.
- Primary comparison: uncapped REVIEW. Fixed 5% capacity sensitivity: rules first, all REVIEW candidates ranked by existing economic benefit and TransactionID, overflow to cheaper non-review action, HARD BLOCK retained. No cap tuned on VALIDATION.

## Software and artifact paths

- Python 3.11.9; packages: {'numpy': '2.4.6', 'pandas': '2.3.3', 'scikit-learn': '1.9.1', 'lightgbm': '4.7.0', 'xgboost': '3.2.0', 'mlflow': '3.16.1'}.
- TRAIN OOF `reports/ensemble_oof_predictions.csv`; raw sources `data/raw/train_transaction.csv`, `data/raw/train_identity.csv` bounded to TRAIN/VALIDATION; calibrator `models/probability_calibrator.joblib`; rules `reports/rules_frozen_config.json`; executed notebook `notebooks/17_rules_engine.ipynb`.
- MLflow tracking URI `sqlite:///./model_comparison.db`, experiment `fraud-detection-rules-engine`, run `b1144bf4a3c442238987416521639a07`.

Future HOLDOUT evaluation must use this lock without revising features, models, calibration, costs, rules, thresholds, or capacity behavior.
