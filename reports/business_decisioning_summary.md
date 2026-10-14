# Business cost modeling and risk decisioning

## MODEL/DATA MEASUREMENTS

The frozen 467-feature champion is 80% LightGBM and 20% XGBoost, followed by the previously selected Platt calibrator. TRAIN OOF policy analysis uses only Fold 2 calibrated from Fold 1 and Fold 3 calibrated from Folds 1+2, giving 82,676 leakage-safe rows. TransactionAmt and labels were aligned by unique TransactionID, and OOF labels were checked against TRAIN labels. External VALIDATION was accessed only after all policies and scenarios were frozen. HOLDOUT was not loaded or evaluated.

## ILLUSTRATIVE BUSINESS COST ASSUMPTIONS

These values are **ILLUSTRATIVE / DEMONSTRATION ASSUMPTIONS**, not company values. IEEE-CIS does not supply actual review cost, recovery, false-decline cost, or customer-friction cost. Replace these inputs with measured, approved values before operational use. No scenario was tuned on TRAIN or external VALIDATION outcomes.

- **Balanced:** {'fraud_loss_multiplier': 1.0, 'manual_review_cost': 3.0, 'review_fraud_detection_rate': 0.8, 'legitimate_review_friction_cost': 1.0, 'false_decline_cost': 25.0, 'block_operational_cost': 0.5}
- **Fraud-loss-sensitive:** {'fraud_loss_multiplier': 2.0, 'manual_review_cost': 3.0, 'review_fraud_detection_rate': 0.8, 'legitimate_review_friction_cost': 1.0, 'false_decline_cost': 20.0, 'block_operational_cost': 0.5}
- **Customer-friction-sensitive:** {'fraud_loss_multiplier': 1.0, 'manual_review_cost': 5.0, 'review_fraud_detection_rate': 0.75, 'legitimate_review_friction_cost': 5.0, 'false_decline_cost': 75.0, 'block_operational_cost': 0.5}

The Balanced profile uses a unit fraud loss multiplier and moderate review and false-decline costs as a neutral demonstration. Fraud-loss-sensitive doubles the modeled fraud exposure and lowers false-decline cost, which favors intervention. Customer-friction-sensitive raises review friction and false-decline cost, which favors approval. The review detection rates are illustrative effectiveness assumptions, not measurements.

## Expected-cost equations and action rule

For p = calibrated fraud probability, A = TransactionAmt, m = fraud loss multiplier, d = review detection rate:

- APPROVE = p × A × m. Fraud exposure is probability times amount times loss multiplier.
- REVIEW = manual review cost + p × (1 − d) × A × m + (1 − p) × legitimate review friction cost. Residual fraud loss remains when review fails.
- BLOCK = (1 − p) × false decline cost + block operational cost. Legitimate customer loss is modeled probabilistically.

Choose minimum expected cost. Ties follow APPROVE, REVIEW, BLOCK. Optional review capacity ranks candidates by expected benefit over the cheaper non-review action; ties use TransactionID. Overflow goes explicitly to the cheaper of APPROVE or BLOCK. No fixed probability threshold is used.

## TRAIN OOF policy analysis

| scenario | approve_rate | review_rate | block_rate | fraud_capture_rate | fraud_amount_capture_rate | legitimate_review_rate | legitimate_block_rate | false_decline_count | approve_all_modeled_cost | modeled_total_cost | modeled_cost_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Balanced | 0.8218 | 0.1534 | 0.0248 | 0.6962 | 0.8063 | 0.1461 | 0.0075 | 591 | 526890.3912 | 169003.9735 | 357886.4177 |
| Fraud-loss-sensitive | 0.7347 | 0.2210 | 0.0443 | 0.7759 | 0.8741 | 0.2183 | 0.0229 | 1816 | 1053780.7824 | 246525.0846 | 807255.6978 |
| Customer-friction-sensitive | 0.9067 | 0.0771 | 0.0161 | 0.5739 | 0.6831 | 0.0671 | 0.0025 | 199 | 526890.3912 | 245184.3651 | 281706.0261 |

Detailed counts, fraud amounts, legitimate impact, and per-transaction costs:

| scenario | approve_count | review_count | block_count | fraud_transactions_approved | fraud_transactions_reviewed | fraud_transactions_blocked | fraudulent_amount_approved | fraudulent_amount_reviewed | fraudulent_amount_blocked | legitimate_approve_rate | legitimate_review_rate | legitimate_block_rate | modeled_cost_per_transaction | approve_all_modeled_cost_per_transaction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Balanced | 67940 | 12683 | 2053 | 801 | 1095 | 1462 | 64356.2670 | 238045.6260 | 275692.5890 | 0.8465 | 0.1461 | 0.0075 | 2.0442 | 6.3730 |
| Fraud-loss-sensitive | 60746 | 18271 | 3659 | 562 | 953 | 1843 | 37168.2310 | 178136.8220 | 362789.4290 | 0.7588 | 0.2183 | 0.0229 | 2.9818 | 12.7459 |
| Customer-friction-sensitive | 74964 | 6377 | 1335 | 1167 | 1055 | 1136 | 121169.3230 | 248188.5750 | 208736.5840 | 0.9304 | 0.0671 | 0.0025 | 2.9656 | 6.3730 |

Fraud capture includes blocked fraud plus reviewed fraud times the illustrative review detection rate. Fraud amount capture applies the same rule to fraudulent TransactionAmt. These are modeled outcomes, not observed reviewer performance. Detailed fraud approved/reviewed/blocked counts and amounts and legitimate rates are in the executed notebook. Approve-all misses all 3,358 fraud transactions and approves 578094.48 fraudulent amount. The approve-all modeled reference varies by scenario multiplier. Cost reductions are **ILLUSTRATIVE SCENARIO OUTPUTS**, not actual savings.

## Balanced review capacity sensitivity (TRAIN OOF)

| capacity | review_rate | fraud_capture_rate | fraud_amount_capture_rate | false_decline_count | modeled_total_cost |
| --- | --- | --- | --- | --- | --- |
| unlimited | 0.1534 | 0.6962 | 0.8063 | 591 | 169003.9735 |
| 1% | 0.0100 | 0.5921 | 0.7010 | 2452 | 228245.6737 |
| 3% | 0.0300 | 0.6009 | 0.7135 | 1645 | 208507.7400 |
| 5% | 0.0500 | 0.6148 | 0.7343 | 1197 | 193907.9978 |

Capacities are sensitivity settings, not optimized values. Review overflow counts are in the notebook.

## Representative amount decision boundaries

Low, median, high amounts are TRAIN TransactionAmt quantiles 10%, 50%, 90%: {0.1: 25.95, 0.5: 68.95, 0.9: 280.0}. Regions are approximate to a 0.001 probability grid for Balanced assumptions.

| amount_quantile | TransactionAmt | decision | p_min | p_max |
| --- | --- | --- | --- | --- |
| 0.1000 | 25.9500 | APPROVE | 0.0000 | 0.1830 |
| 0.1000 | 25.9500 | REVIEW | 0.1840 | 0.7360 |
| 0.1000 | 25.9500 | BLOCK | 0.7370 | 1.0000 |
| 0.5000 | 68.9500 | APPROVE | 0.0000 | 0.0710 |
| 0.5000 | 68.9500 | REVIEW | 0.0720 | 0.5680 |
| 0.5000 | 68.9500 | BLOCK | 0.5690 | 1.0000 |
| 0.9000 | 280.0000 | APPROVE | 0.0000 | 0.0170 |
| 0.9000 | 280.0000 | REVIEW | 0.0180 | 0.2680 |
| 0.9000 | 280.0000 | BLOCK | 0.2690 | 1.0000 |

See the three decisioning figures in reports/figures/.

## External VALIDATION: one assessment of frozen scenarios

| scenario | approve_rate | review_rate | block_rate | fraud_capture_rate | fraud_amount_capture_rate | legitimate_review_rate | legitimate_block_rate | false_decline_count | approve_all_modeled_cost | modeled_total_cost | modeled_cost_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Balanced | 0.8344 | 0.1471 | 0.0185 | 0.6901 | 0.8176 | 0.1387 | 0.0056 | 475 | 506794.6542 | 179424.5429 | 327370.1112 |
| Fraud-loss-sensitive | 0.7445 | 0.2180 | 0.0375 | 0.7679 | 0.8776 | 0.2143 | 0.0207 | 1772 | 1013589.3083 | 264301.8418 | 749287.4666 |
| Customer-friction-sensitive | 0.9164 | 0.0730 | 0.0107 | 0.5435 | 0.6861 | 0.0627 | 0.0014 | 117 | 506794.6542 | 256077.4404 | 250717.2138 |

Detailed external VALIDATION outcomes:

| scenario | approve_count | review_count | block_count | fraud_transactions_approved | fraud_transactions_reviewed | fraud_transactions_blocked | fraudulent_amount_approved | fraudulent_amount_reviewed | fraudulent_amount_blocked | legitimate_approve_rate | legitimate_review_rate | legitimate_block_rate | modeled_cost_per_transaction | approve_all_modeled_cost_per_transaction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Balanced | 73908 | 13033 | 1640 | 709 | 1168 | 1165 | 49739.8840 | 204445.9500 | 242721.7540 | 0.8557 | 0.1387 | 0.0056 | 2.0255 | 5.7213 |
| Fraud-loss-sensitive | 65946 | 19315 | 3320 | 509 | 985 | 1548 | 32136.5050 | 143375.3540 | 321395.7290 | 0.7650 | 0.2143 | 0.0207 | 2.9837 | 11.4425 |
| Customer-friction-sensitive | 81173 | 6462 | 946 | 1114 | 1099 | 829 | 100249.0730 | 222889.9070 | 173768.6080 | 0.9359 | 0.0627 | 0.0014 | 2.8909 | 5.7213 |

Approve-all on external VALIDATION misses all 3,042 fraud transactions and approves 496907.59 fraudulent amount. Decisions use only Platt-calibrated probabilities. Detailed validation fraud counts and fraudulent amount totals, and legitimate approval/review/block rates, are in the notebook. The capture trade-offs depend on illustrative costs and assumed review effectiveness; they are not measured business savings.

## Operational limits and production implications

Real review detection, recovery, chargeback, customer lifetime value, and friction costs are unknown. Expected cost is a scenario calculation using calibrated probabilities, not realized net revenue. Review capacity changes the action mix and can increase modeled cost. Controlled use requires approved cost inputs and matured outcome labels. No API, extra rules, SHAP, or holdout evaluation was performed.

## Runtime, memory, tracking

Notebook runtime: 197.3 seconds. Prepared feature matrices: 905.7 MiB; the notebook rebuild required several GiB of process memory, so TRAIN OOF arrays were released before model preparation. MLflow experiment: fraud-detection-business-decisioning; tracking URI: sqlite:///./model_comparison.db; runs: {"Balanced": "c5a1857447de486ba0cf4c8d5067d16f", "Fraud-loss-sensitive": "a18fd95b2c9b4d7980da7d36987d0fea", "Customer-friction-sensitive": "ef32f3c78caf4e5c87efe4a43289b120", "capacity-unlimited": "459a16f80c634a249fb82ad97b10865f", "capacity-1%": "e7f7c66d5c294d46948c5f56c54d29b1", "capacity-3%": "e5d20f555af84f2b8ae714402c87f7fb", "capacity-5%": "0a538ff8be084dd484f88688fd5195e7"}.
