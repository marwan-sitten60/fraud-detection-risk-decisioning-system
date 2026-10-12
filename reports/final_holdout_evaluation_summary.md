# Final untouched HOLDOUT evaluation

## Target-encoding memory optimization and pre-HOLDOUT parity gate

The only frozen implementation change projects the encoder's existing required columns before copying its query and history frames. Formula, keys, missing values, temporal order, same-timestamp isolation, label delay, names, order, and dtypes did not change. The earlier full-frame copy failed on 413,378 TRAIN rows while requesting a 1.23 GiB wide-frame allocation. The optimized feature preparation completed; sampled worker resident memory reached approximately 6.6 GiB, which is not a measured peak.

Saved old-encoder outputs matched exactly on synthetic wide frames with missing and unseen identities, repeated keys, same-timestamp batches, chronological updates, composite encodings, and zero-day and seven-day delays. The original committed encoder and optimized encoder also matched exactly on 2,000 TRAIN rows and 500 later TRAIN rows for all ten features, row indices, names, order, dtypes, and availability statistics at both delays. Before HOLDOUT prediction, the TRAIN-only 467-feature VALIDATION rebuild reproduced PR-AUC **0.615264501963** and ROC-AUC **0.916223956316**; Balanced calibrated decisions reproduced exactly at 73,908 APPROVE, 13,033 REVIEW, and 1,640 BLOCK. No earlier row-level VALIDATION prediction artifact existed for direct score comparison. See `reports/target_encoder_validation_parity.json` and `reports/target_encoder_real_train_parity.json`. No HOLDOUT labels were read during optimization or this gate.

## Pre-HOLDOUT lock verification

The committed working tree was clean at the start of this phase. The manifest was amended **before any HOLDOUT access** to explicitly lock the final model fit scope to TRAIN only. Its 467-feature pipeline, 0.8/0.2 blend, zero-day target-encoding label delay, frozen TRAIN-OOF-derived Platt calibrator, Balanced business costs, and rules-disabled primary policy were verified. The frozen rule-config SHA-256 is `3a4380831e177828c82a1a118259f83e354221fa52205b41d38aa21b58febb94`; the Platt artifact SHA-256 is `3503bf66161b5152e9de4d638b9e7932d581fc371abd6a45b6969ba6c8134f6e`. No HOLDOUT-derived artifact existed before this evaluation.

## Exact fit scope and chronological integrity

LightGBM, XGBoost, preprocessing, and label-based feature state were fitted on **TRAIN only**, using the locked parameters and temporal semantics. No TRAIN+VALIDATION refit occurred. The existing Platt artifact was reused unchanged. The HOLDOUT contains 88,581 rows, matching the original 88,581-row split. TransactionID is unique and disjoint from all TRAIN and VALIDATION IDs; its IDs and TransactionDT follow the development period. The feature schema matches TRAIN after excluding its label. No HOLDOUT label was available to feature construction, model scoring, calibration, business decisioning, or optional rule evaluation. Non-label history updates within the evaluation frame follow the existing feature transform implementation; label-based target encoding uses only matured TRAIN history.

## Prediction lock before label evaluation

The label-free `reports/holdout_predictions_frozen.csv` was written first with TransactionID, TransactionDT, TransactionAmt, both component scores, raw ensemble score, calibrated probability, and primary Balanced decision. It also records separately the fixed 5% capacity action and optional frozen-rule action. SHA-256: `0d7f5d117436cd7227d682388c5cceddbc72b689143984a7d186d9ec03c1391f`. `reports/holdout_prediction_lock.json` records the hash, row count, fit scope, and rules-disabled primary policy. Only after verifying this saved file and its checksum were HOLDOUT labels read and aligned by TransactionID.

## Final ranking and calibration

Primary PR-AUC 0.536900; ROC-AUC 0.878637. Versus frozen VALIDATION PR-AUC 0.615265 and ROC-AUC 0.916224, the absolute changes are -0.078365 and -0.037587; relative changes are -12.74% and -4.10%. PR-AUC is the primary ranking metric.

Frozen Platt calibration: Brier 0.021823; Log Loss 0.094295; ten equal-count-bin ECE 0.005559; mean probability 0.037656; observed fraud prevalence 0.034804. Descriptive VALIDATION references: Brier 0.019493, Log Loss 0.082039, ECE 0.005434. No recalibration occurred. The reliability figure is `reports/figures/holdout_calibration_curve.png`.

At calibrated probability >= 0.50: tp=1104, fp=276, fn=1979, tn=85222, precision=0.800000, recall=0.358093, f1=0.494734. **0.50 is a diagnostic cutoff, not the production business policy.**

## PRIMARY: Balanced expected-cost policy, rules disabled

| policy | approve_count | review_count | block_count | approve_rate | review_rate | block_rate | fraud_capture_rate | fraud_amount_capture_rate | fraud_transactions_approved | fraudulent_amount_approved | legitimate_approve_rate | legitimate_review_rate | legitimate_block_rate | modeled_total_cost | modeled_cost_per_transaction | approve_all_modeled_cost | modeled_cost_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary_balanced_rules_disabled | 73351 | 13595 | 1635 | 0.828067 | 0.153475 | 0.018458 | 0.622511 | 0.764860 | 945 | 64852.544000 | 0.846874 | 0.146214 | 0.006912 | 183099.159483 | 2.067025 | 487761.123419 | 304661.963936 |

APPROVE/REVIEW/BLOCK counts and rates, modeled fraud transaction and amount capture, approved fraud count and amount, legitimate action rates, review workload, blocked count, illustrative total and per-transaction cost, approve-all cost, and modeled reduction versus approve-all are in the table. Fraud capture assumes the frozen illustrative REVIEW detection rate of 0.8; no actual review outcomes are observed. Modeled cost differences are not real-world savings.

### Fixed 5% review-capacity sensitivity

| policy | approve_count | review_count | block_count | approve_rate | review_rate | block_rate | fraud_capture_rate | fraud_amount_capture_rate | fraud_transactions_approved | fraudulent_amount_approved | legitimate_approve_rate | legitimate_review_rate | legitimate_block_rate | modeled_total_cost | modeled_cost_per_transaction | approve_all_modeled_cost | modeled_cost_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary_5pct_review_capacity | 81470 | 4429 | 2682 | 0.919723 | 0.049999 | 0.030277 | 0.526370 | 0.683765 | 1391 | 124548.952000 | 0.936618 | 0.047756 | 0.015626 | 209901.366845 | 2.369598 | 487761.123419 | 277859.756574 |

The uncapped primary and the already-established 5% cap use the same Balanced expected-cost formulas. REVIEW candidates are ranked by economic benefit and TransactionID; overflow receives the cheaper non-review action. No capacity was chosen from HOLDOUT outcomes.

## SECONDARY: OPTIONAL FROZEN OPERATIONAL OVERLAY

The rules were loaded unchanged from `reports/rules_frozen_config.json` and evaluated once. They are **not** part of the primary champion policy. Total triggered 2,984/88,581 (3.369%); base-action overrides 2,974. Per-rule triggers: R01=10 (0.011%), R02=2,440 (2.755%), R03=597 (0.674%).

| policy | approve_count | review_count | block_count | approve_rate | review_rate | block_rate | fraud_capture_rate | fraud_amount_capture_rate | fraud_transactions_approved | fraudulent_amount_approved | legitimate_approve_rate | legitimate_review_rate | legitimate_block_rate | modeled_total_cost | modeled_cost_per_transaction | approve_all_modeled_cost | modeled_cost_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optional_frozen_rules_overlay | 70377 | 16569 | 1635 | 0.794493 | 0.187049 | 0.018458 | 0.664548 | 0.774638 | 783 | 59112.817000 | 0.813984 | 0.179104 | 0.006912 | 188293.525767 | 2.125665 | 487761.123419 | 299467.597652 |

The overlay did not change calibrated probabilities. Its retrospective outcomes were not used to revise the rules or primary policy.

## VALIDATION-to-HOLDOUT generalization and distribution checks

| split | fraud_prevalence | pr_auc | roc_auc | brier | log_loss | ece | mean_calibrated_probability | approve_rate | review_rate | block_rate | fraud_capture_rate | fraud_amount_capture_rate | legitimate_review_rate | legitimate_block_rate | modeled_cost_per_transaction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VALIDATION | 0.034341 | 0.615265 | 0.916224 | 0.019493 | 0.082039 | 0.005434 | 0.036660 | 0.834355 | 0.147131 | 0.018514 | 0.690138 | 0.817614 | 0.138709 | 0.005553 | 2.025542 |
| HOLDOUT | 0.034804 | 0.536900 | 0.878637 | 0.021823 | 0.094295 | 0.005559 | 0.037656 | 0.828067 | 0.153475 | 0.018458 | 0.622511 | 0.764860 | 0.146214 | 0.006912 | 2.067025 |

Transaction-amount distribution (existing quantity, no new feature):

| split | mean_amount | p50_amount | p90_amount | p99_amount |
| --- | --- | --- | --- | --- |
| TRAIN | 134.604892 | 68.950000 | 280.000000 | 1104.000000 |
| VALIDATION | 134.876482 | 67.950000 | 268.000000 | 1104.000000 |
| HOLDOUT | 137.148533 | 68.500000 | 264.860000 | 1226.894000 |

Calibrated-risk distribution compared with the earlier validation diagnostic bands:

| calibrated_risk_band | holdout_count | holdout_share | validation_share |
| --- | --- | --- | --- |
| 0-1% | 44659 | 0.504160 | 0.504341 |
| 1-2% | 15024 | 0.169607 | 0.172791 |
| 2-5% | 16667 | 0.188155 | 0.189341 |
| 5-10% | 6296 | 0.071076 | 0.070342 |
| 10-20% | 2901 | 0.032750 | 0.031305 |
| 20-40% | 1325 | 0.014958 | 0.013852 |
| 40%+ | 1709 | 0.019293 | 0.018029 |

These amount, risk, prevalence, and decision shifts are descriptive only. No drift-triggered system change was made.

## TRAIN-defined amount-band fraud outcomes

Bands use TRAIN amount quantiles only: [0.0, 42.95, 68.95, 125.0, 280.0, inf]. Captured fraud count and amount include BLOCK plus 0.8 times REVIEW, following the frozen illustrative assumption.

| amount_band | transactions | fraud_count | fraud_amount | modeled_fraud_transaction_capture | modeled_fraud_amount_capture | approved_fraud_amount |
| --- | --- | --- | --- | --- | --- | --- |
| [0.00, 42.95) | 21580 | 1034 | 24248.296000 | 0.538104 | 0.571175 | 9014.097000 |
| [42.95, 68.95) | 22756 | 544 | 29447.046000 | 0.536397 | 0.535757 | 11831.929000 |
| [68.95, 125.00) | 22521 | 658 | 64661.190000 | 0.600608 | 0.590071 | 22132.896000 |
| [125.00, 280.00) | 13163 | 412 | 76661.923000 | 0.755825 | 0.755391 | 12041.352000 |
| [280.00, +inf) | 8561 | 435 | 274590.066000 | 0.837701 | 0.850336 | 9832.270000 |

## Runtime, memory, and tracking

Runtime 207.0 seconds. Prepared feature-matrix estimate 905.7 MiB. Kernel resident-memory snapshot at report time 2413.8 MiB (not a peak). MLflow experiment `fraud-detection-final-holdout`; run `447dbfb1edac4b8d9db70850b6c4b9bb`; tracking URI `sqlite:///C:/fraud-detection-system/model_comparison.db`. The run logs frozen identifiers, terminal metrics, policy comparisons, capacity sensitivity, optional-rule results, runtime, and figures.

## Final verification

The executed notebook has all code cells completed without errors. The saved prediction artifact contains no `isFraud` column, has 88,581 rows, and its SHA-256 matches the pre-label lock. The complete suite passed (`pytest -q`: 181 passed); `ruff check .` and `ruff format --check .` both passed. The frozen model parameters, feature definitions and order, Platt artifact, 0.8/0.2 ensemble, Balanced costs, rule definitions and thresholds, and rules-disabled primary policy were not changed. The encoder column projection was verified as numerically identical before HOLDOUT prediction. No HOLDOUT result was used to tune the system.

## Terminal result lock

**The HOLDOUT result is terminal.** No change to the model, features, ensemble, calibration, rules, thresholds, or business policy may be justified using HOLDOUT performance. Future improvements require a new experimental cycle and a new future evaluation set. No HOLDOUT-driven tuning occurred in this phase.
