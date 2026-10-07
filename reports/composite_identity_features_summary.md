# Composite Entity / Identity Features

## Composite identities

- **card1_card2_addr1:** card1 + card2 + addr1
- **card1_addr1_p_emaildomain:** card1 + addr1 + P_emaildomain
- **card1_card2_addr1_productcd:** card1 + card2 + addr1 + ProductCD
- **card1_addr1_deviceinfo:** card1 + addr1 + DeviceInfo

Each component uses `M` when missing. Nonmissing values use `V<length>:<value>` before
joining, so a real value cannot collide with the missing token or component delimiters.

## New features

- `card1_card2_addr1_composite_frequency`
- `card1_addr1_p_emaildomain_composite_frequency`
- `card1_card2_addr1_productcd_composite_frequency`
- `card1_addr1_deviceinfo_composite_frequency`
- `card1_card2_addr1_previous_tx_count`
- `card1_card2_addr1_previous_avg_amount`
- `card1_card2_addr1_previous_max_amount`
- `card1_card2_addr1_amount_vs_previous_avg`
- `card1_addr1_p_emaildomain_previous_tx_count`
- `card1_addr1_p_emaildomain_previous_avg_amount`
- `card1_addr1_p_emaildomain_previous_max_amount`
- `card1_addr1_p_emaildomain_amount_vs_previous_avg`
- `card1_card2_addr1_productcd_previous_tx_count`
- `card1_card2_addr1_productcd_previous_avg_amount`
- `card1_card2_addr1_productcd_previous_max_amount`
- `card1_card2_addr1_productcd_amount_vs_previous_avg`
- `card1_addr1_deviceinfo_previous_tx_count`
- `card1_addr1_deviceinfo_previous_avg_amount`
- `card1_addr1_deviceinfo_previous_max_amount`
- `card1_addr1_deviceinfo_amount_vs_previous_avg`

There are **20** controlled composite features:
4 fold-local frequencies and
16 past-only history features.
Historical batches use only `TransactionDT < T`; equal-time transactions never see each other.
No target statistic is used.

## Temporal CV

Three expanding folds inside TRAIN use fold-local frequency maps, historical state, imputation,
and preprocessing. External VALIDATION is not used for feature selection.

| Feature set | Features | Fold PR-AUCs | Mean | Std | Mean ROC-AUC |
| --- | ---: | --- | ---: | ---: | ---: |
| baseline_plus_all_composite | 463 | 0.554229 / 0.574851 / 0.625544 | 0.584875 | 0.029965 | 0.906202 |
| baseline_plus_composite_frequency | 447 | 0.547756 / 0.569923 / 0.625902 | 0.581194 | 0.032883 | 0.908516 |
| baseline_plus_composite_history | 459 | 0.548773 / 0.552403 / 0.618456 | 0.573211 | 0.032028 | 0.901608 |
| baseline | 443 | 0.544091 / 0.542344 / 0.621051 | 0.569162 | 0.036698 | 0.906386 |

- TRAIN-CV winner: **baseline_plus_all_composite**
- Winner feature count: **463**

## External validation

- PR-AUC: **0.595387**
- ROC-AUC: **0.926655**
- Precision at 0.50: **0.538343**
- Recall at 0.50: **0.572321**
- F1 at 0.50: **0.554812**
- Confusion matrix: **((84046, 1493), (1301, 1741))**
- Previous preferred LightGBM PR-AUC: **0.593938**
- Absolute improvement: **+0.001449**
- Relative improvement: **+0.24%**

## Runtime, memory, and serving complexity

- Fold feature-preparation seconds: **43.5, 44.6, 48.1**
- Final TRAIN/VALIDATION feature preparation: **60.9 seconds**
- Peak prepared fold estimate: **924.9 MiB**
- Final winner fit/evaluation workflow: **96.3 seconds**
- Total notebook time before report write: **622.8 seconds**
- Serving requires deterministic composite normalization, four reference maps when selected,
  and keyed count/sum/max state for selected history features. This is moderate stateful complexity.

## MLflow

- Experiment: `fraud-detection-composite-features`
- **baseline:** `6784c22734834c22b7a1803e83dfdc74`
- **baseline_plus_composite_frequency:** `d0e6018a4c8b4dc5902dcdbe1dc5b40a`
- **baseline_plus_composite_history:** `0d55432e7a2c459a83204032def3c87c`
- **baseline_plus_all_composite:** `04db3f18cef642f0bc7e524a119e4e22`
- **Final winner:** `a7cc25894a484b078d476e063eeb0007`

## Scope confirmation

External VALIDATION was evaluated once after TRAIN-CV selection. HOLDOUT was never assigned,
transformed, predicted, summarized, or evaluated. No graph features, CatBoost, target encoding,
calibration, threshold optimization, SHAP, or rules are included.
