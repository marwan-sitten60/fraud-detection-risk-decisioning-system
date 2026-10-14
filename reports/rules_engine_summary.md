# Deterministic Fraud Rules Engine

**ILLUSTRATIVE OPERATIONAL RULES.** IEEE-CIS supplies no confirmed blacklist, compromised-device, merchant, sanctions, or company-rule feeds. Model risk, Balanced expected-cost base action, and deterministic rule action are separate layers. Rules never change the score.

## Frozen upstream system and TRAIN-only development

467 features, 80% LightGBM / 20% XGBoost, frozen Platt calibrator, and frozen Balanced cost scenario. TRAIN OOF analysis uses 82,676 chronological Fold 2/3 predictions, each calibrated from earlier folds. Composite frequency comes from each fold's strictly earlier TRAIN reference. Thresholds were saved to `reports/rules_frozen_config.json` before external VALIDATION access. Fraud labels, VALIDATION results, and SHAP values did not select rules or cutoffs.

## Exact illustrative rules and TRAIN-derived thresholds

- R01 **HARD BLOCK**, priority 100, reason `EXTREME_RISK_HIGH_AMOUNT`: TRAIN p99 calibrated risk and TRAIN p95 transaction amount.
- R02 **SOFT REVIEW**, priority 60, reason `MODEL_DISAGREEMENT`: TRAIN p95 component disagreement with risk above TRAIN p75.
- R03 **SOFT REVIEW**, priority 50, reason `RARE_IDENTITY_ELEVATED_RISK`: Rare TRAIN reference composite frequency, elevated risk, and nontrivial amount.

R01 uses calibrated p99 and amount p95. R02 uses disagreement p95 and calibrated p75. R03 uses fold-local composite frequency p10, calibrated p75, and amount median. All comparisons are inclusive; unseen identity frequency is zero. TRAIN p99 amount is context only, not an active cutoff.

- high_amount: 444.950000000
- extreme_amount: 1104.000000000
- very_high_risk: 0.883205387
- elevated_risk: 0.028448886
- high_disagreement: 0.163710277
- rare_composite_frequency: 0.000000000
- nontrivial_amount: 68.950000000

HARD BLOCK > SOFT REVIEW > base. Within a type, higher priority wins, then rule ID. All triggered rule IDs, names, and reasons are retained. Soft rules only escalate APPROVE. No rule lowers BLOCK or changes calibrated probability.

## TRAIN OOF rule behavior

Triggered 1,710/82,676 (2.0683%); overrides 1,661; APPROVE -> REVIEW 1,661; APPROVE/REVIEW -> BLOCK 0.

| rule_id | trigger_count | trigger_rate | fraud_prevalence | transaction_amount | fraudulent_amount | base_decisions_overridden | resulting_approve | resulting_review | resulting_block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | 49 | 0.000593 | 0.959184 | 36539.000000 | 35560.000000 | 0 | 0 | 0 | 49 |
| R02 | 1311 | 0.015857 | 0.069413 | 54967.986000 | 2611.248000 | 1311 | 0 | 1311 | 0 |
| R03 | 373 | 0.004512 | 0.026810 | 38316.216000 | 953.000000 | 373 | 0 | 373 | 0 |

Overlap: exactly one=1,687; multiple=23. Common combinations: ('R02',): 1288, ('R03',): 350, ('R01',): 49, ('R02', 'R03'): 23.

Base-to-final decisions:

| base_decision | APPROVE | BLOCK | REVIEW |
| --- | --- | --- | --- |
| APPROVE | 66279 | 0 | 1661 |
| BLOCK | 0 | 2053 | 0 |
| REVIEW | 0 | 0 | 12683 |

### Frozen Balanced base versus Balanced plus rules

| policy | approve_count | review_count | block_count | approve_rate | review_rate | block_rate | fraud_capture_rate | fraud_amount_capture_rate | fraudulent_amount_approved | legitimate_review_rate | legitimate_block_rate | modeled_total_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 67940 | 12683 | 2053 | 0.821762 | 0.153406 | 0.024832 | 0.696248 | 0.806320 | 64356.267000 | 0.146095 | 0.007451 | 169003.973452 |
| base_plus_rules | 66279 | 14344 | 2053 | 0.801672 | 0.173497 | 0.024832 | 0.720310 | 0.811252 | 60792.019000 | 0.165763 | 0.007451 | 171824.456071 |

Capture assumes the existing illustrative 0.8 REVIEW fraud-detection rate. Modeled cost uses the same Balanced expected-cost formula. Its difference is not a measured saving.

## Review capacity

Order: base expected-cost decision, rules, collect REVIEW candidates, then existing economic benefit ranking and TransactionID tie break. Primary comparison is uncapped to match earlier Balanced analysis. Fixed 5% sensitivity: uncapped REVIEW 14,344, capped REVIEW 4,133, overflow 10,211. Overflow becomes the cheaper non-review action. HARD BLOCK remains BLOCK.

## External VALIDATION: one application of frozen rules

The champion reproduced PR-AUC 0.615265 and ROC-AUC 0.916224. Persisted thresholds were checked unchanged. Triggered 2,737/88,581 (3.0898%); overrides 2,708; APPROVE -> REVIEW 2,708; APPROVE/REVIEW -> BLOCK 0. Rules were not revised after these outcomes.

| rule_id | trigger_count | trigger_rate | fraud_prevalence | transaction_amount | fraudulent_amount | base_decisions_overridden | resulting_approve | resulting_review | resulting_block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | 29 | 0.000327 | 0.965517 | 27105.000000 | 26474.000000 | 0 | 0 | 0 | 29 |
| R02 | 2259 | 0.025502 | 0.054449 | 107425.187000 | 4097.329000 | 2259 | 0 | 2259 | 0 |
| R03 | 490 | 0.005532 | 0.014286 | 50326.403000 | 634.389000 | 490 | 0 | 490 | 0 |

Overlap: exactly one=2,696; multiple=41. Common combinations: ('R02',): 2218, ('R03',): 449, ('R02', 'R03'): 41, ('R01',): 29.

Base-to-final decisions:

| base_decision | APPROVE | BLOCK | REVIEW |
| --- | --- | --- | --- |
| APPROVE | 71200 | 0 | 2708 |
| BLOCK | 0 | 1640 | 0 |
| REVIEW | 0 | 0 | 13033 |

| policy | approve_count | review_count | block_count | approve_rate | review_rate | block_rate | fraud_capture_rate | fraud_amount_capture_rate | fraudulent_amount_approved | legitimate_review_rate | legitimate_block_rate | modeled_total_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 73908 | 13033 | 1640 | 0.834355 | 0.147131 | 0.018514 | 0.690138 | 0.817614 | 49739.884000 | 0.138709 | 0.005553 | 179424.542932 |
| base_plus_rules | 71200 | 15741 | 1640 | 0.803784 | 0.177702 | 0.018514 | 0.724326 | 0.825232 | 45008.166000 | 0.168847 | 0.005553 | 184237.729612 |

Fraud transaction/amount capture, fraudulent amount approved, legitimate REVIEW/BLOCK rates, and modeled cost appear above. Incremental customer friction is the difference in legitimate intervention rates. Triggered-row fraud prevalence is retrospective only. Figures are under `reports/figures/`.

## Limits and extension

These rules illustrate architecture, not real company policy. Disagreement signals uncertainty, not confirmed fraud. REVIEW effectiveness and costs are assumptions. Genuine blacklist, device, merchant, or regulatory predicates can be registered without changing model code when verified feeds and ownership exist.

## Runtime, memory, and MLflow

Notebook runtime: 217.5 seconds. Prepared full TRAIN+VALIDATION feature matrices: 905.7 MiB. MLflow experiment `fraud-detection-rules-engine`; run `b1144bf4a3c442238987416521639a07`; tracking URI `sqlite:///./model_comparison.db`. Thresholds, trigger/override rates, decision rates, capture, legitimate impact, modeled costs, config, and three figures are logged.

No model, feature, ensemble weight, calibrator, cost scenario, or risk score changed. HOLDOUT was not loaded, transformed, predicted, or evaluated.

## Verification gates

The executed notebook contains eight completed code cells and no error outputs. The MLflow run has status `FINISHED`; its persisted metrics, exact thresholds, and rule artifacts were verified. `pytest -q`: 175 passed (8 scikit-learn deprecation warnings). `ruff check .`: passed. `ruff format --check .`: passed (93 files already formatted).

## Observed policy trade-off

On TRAIN OOF, the rules raised modeled fraud transaction capture by 2.406 percentage points and legitimate REVIEW rate by 1.967 points; modeled expected cost increased by 2,820.483. On external VALIDATION, modeled fraud transaction capture rose by 3.419 points and legitimate REVIEW rate by 3.014 points; modeled expected cost increased by 4,813.187. R01 added no BLOCK overrides in either split because the base Balanced policy already blocked its matched transactions. These are retrospective, assumption-dependent outcomes; no rule was changed in response.
