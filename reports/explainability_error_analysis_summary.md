# Explainability and error analysis of the frozen system

## Frozen system and validation reproduction

The analyzed system uses 467 existing predictors, frozen LightGBM and XGBoost parameters, an 80/20 probability blend, the TRAIN-OOF-fitted Platt calibrator, and the previously frozen Balanced illustrative decision profile. Rebuilt external VALIDATION PR-AUC = 0.615265; ROC-AUC = 0.916224. These reproduce the champion reference. Diagnostics did not change any model, feature, calibrator, ensemble weight, scenario, or policy. HOLDOUT was not loaded, predicted, explained, or evaluated.

## Component TreeSHAP scope and sample

Native LightGBM and XGBoost tree-contribution APIs calculate TreeSHAP separately in **raw-margin/log-odds space**, with additivity checked against each model's raw margin. The global sample is 5,000 uniformly sampled VALIDATION rows without replacement using NumPy seed 42; fraud was not oversampled. The final score is component probabilities → 80/20 blend → Platt calibration → expected-cost action. These SHAP values are **COMPONENT MODEL EXPLANATIONS**, not an exact explanation of the calibrated ensemble or business decision. SHAP importance is not causal inference.

## Top LightGBM features

| rank | feature | family | mean_abs_shap |
| --- | --- | --- | --- |
| 1 | TransactionDT | raw_transaction | 0.48073 |
| 2 | card1_addr1_p_emaildomain_target_encoded_fraud_rate | composite_temporal_te | 0.45201 |
| 3 | D1 | raw_transaction | 0.27580 |
| 4 | C13 | raw_transaction | 0.25891 |
| 5 | card1_card2_addr1_productcd_target_encoded_fraud_rate | composite_temporal_te | 0.22966 |
| 6 | card1_addr1_deviceinfo_target_encoded_fraud_rate | composite_temporal_te | 0.20280 |
| 7 | V307 | raw_transaction | 0.15560 |
| 8 | amount_vs_addr1_previous_avg | historical_amount | 0.14339 |
| 9 | C14 | raw_transaction | 0.13848 |
| 10 | card1_addr1_p_emaildomain_previous_tx_count | composite_identity | 0.13166 |
| 11 | card1_frequency | frequency_encoding | 0.12994 |
| 12 | card1_addr1_deviceinfo_composite_frequency | composite_identity | 0.12946 |
| 13 | card1_card2_addr1_composite_frequency | composite_identity | 0.12815 |
| 14 | P_emaildomain | raw_transaction | 0.12369 |
| 15 | card2_frequency | frequency_encoding | 0.11670 |

## Top XGBoost features

| rank | feature | family | mean_abs_shap |
| --- | --- | --- | --- |
| 1 | card1_addr1_p_emaildomain_target_encoded_fraud_rate | composite_temporal_te | 0.64649 |
| 2 | card1_card2_addr1_productcd_target_encoded_fraud_rate | composite_temporal_te | 0.34646 |
| 3 | D1 | raw_transaction | 0.26521 |
| 4 | C13 | raw_transaction | 0.21863 |
| 5 | TransactionDT | raw_transaction | 0.21035 |
| 6 | C14 | raw_transaction | 0.20255 |
| 7 | C1 | raw_transaction | 0.18553 |
| 8 | amount_vs_addr1_previous_avg | historical_amount | 0.17650 |
| 9 | card1_addr1_deviceinfo_target_encoded_fraud_rate | composite_temporal_te | 0.16749 |
| 10 | card1_addr1_p_emaildomain_previous_tx_count | composite_identity | 0.12869 |
| 11 | card1_frequency | frequency_encoding | 0.12323 |
| 12 | V258 | raw_transaction | 0.11888 |
| 13 | V91 | raw_transaction | 0.10787 |
| 14 | V307 | raw_transaction | 0.10041 |
| 15 | card1_addr1_deviceinfo_composite_frequency | composite_identity | 0.09658 |

Features in both top 20: ['C1', 'C13', 'C14', 'D1', 'P_emaildomain', 'TransactionAmt', 'TransactionDT', 'V307', 'amount_vs_addr1_previous_avg', 'card1_addr1_deviceinfo_composite_frequency', 'card1_addr1_deviceinfo_target_encoded_fraud_rate', 'card1_addr1_p_emaildomain_previous_tx_count', 'card1_addr1_p_emaildomain_target_encoded_fraud_rate', 'card1_card2_addr1_productcd_target_encoded_fraud_rate', 'card1_frequency']. LightGBM-only top 20: ['C5', 'card1_addr1_p_emaildomain_composite_frequency', 'card1_card2_addr1_composite_frequency', 'card1_previous_avg_amount', 'card2_frequency']. XGBoost-only top 20: ['C11', 'D4', 'V258', 'V70', 'V91']. Signed SHAP direction versus standardized feature value was summarized with sample Spearman correlations, not causal effects:

| model | feature | feature_value_shap_spearman |
| --- | --- | --- |
| LightGBM | TransactionDT | 0.01484 |
| LightGBM | card1_addr1_p_emaildomain_target_encoded_fraud_rate | 0.94903 |
| LightGBM | D1 | -0.86260 |
| LightGBM | C13 | -0.93545 |
| LightGBM | card1_card2_addr1_productcd_target_encoded_fraud_rate | 0.91500 |
| LightGBM | card1_addr1_deviceinfo_target_encoded_fraud_rate | 0.79063 |
| LightGBM | V307 | -0.75965 |
| LightGBM | amount_vs_addr1_previous_avg | 0.89736 |
| LightGBM | C14 | -0.80156 |
| LightGBM | card1_addr1_p_emaildomain_previous_tx_count | -0.48375 |
| LightGBM | card1_frequency | 0.37581 |
| LightGBM | card1_addr1_deviceinfo_composite_frequency | 0.79935 |
| LightGBM | card1_card2_addr1_composite_frequency | 0.82125 |
| LightGBM | P_emaildomain | 0.60696 |
| LightGBM | card2_frequency | 0.52457 |
| XGBoost | card1_addr1_p_emaildomain_target_encoded_fraud_rate | 0.86590 |
| XGBoost | card1_card2_addr1_productcd_target_encoded_fraud_rate | 0.81400 |
| XGBoost | D1 | -0.87276 |
| XGBoost | C13 | -0.95135 |
| XGBoost | TransactionDT | 0.01139 |
| XGBoost | C14 | -0.65400 |
| XGBoost | C1 | 0.70823 |
| XGBoost | amount_vs_addr1_previous_avg | 0.90473 |
| XGBoost | card1_addr1_deviceinfo_target_encoded_fraud_rate | 0.51018 |
| XGBoost | card1_addr1_p_emaildomain_previous_tx_count | -0.43887 |
| XGBoost | card1_frequency | 0.46032 |
| XGBoost | V258 | 0.26831 |
| XGBoost | V91 | -0.82965 |
| XGBoost | V307 | -0.58972 |
| XGBoost | card1_addr1_deviceinfo_composite_frequency | 0.75990 |

## Feature-family and temporal target-encoding importance

| family | mean_abs_shap_lightgbm | share_lightgbm | rank_lightgbm | mean_abs_shap_xgboost | share_xgboost | rank_xgboost |
| --- | --- | --- | --- | --- | --- | --- |
| raw_transaction | 4.96812 | 0.62423 | 1 | 4.39776 | 0.60295 | 1 |
| composite_identity | 1.03786 | 0.13040 | 2 | 0.88064 | 0.12074 | 3 |
| composite_temporal_te | 0.92860 | 0.11668 | 3 | 1.19982 | 0.16450 | 2 |
| historical_amount | 0.37206 | 0.04675 | 4 | 0.35825 | 0.04912 | 4 |
| raw_identity | 0.36480 | 0.04584 | 5 | 0.23954 | 0.03284 | 5 |
| frequency_encoding | 0.28730 | 0.03610 | 6 | 0.21769 | 0.02985 | 6 |

The four retained composite temporal target-encoding features are **matured historical fraud-risk signals**; no current-row fraud label enters their validation values. Individual raw-margin contributions and global ranks:

| model | feature | rank | mean_abs_shap | p05_shap | median_shap | p95_shap | positive_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | card1_card2_addr1_target_encoded_fraud_rate | 55 | 0.04414 | -0.08694 | 0.01114 | 0.11161 | 0.62240 |
| LightGBM | card1_addr1_p_emaildomain_target_encoded_fraud_rate | 2 | 0.45201 | -0.53052 | -0.03328 | 1.67540 | 0.46000 |
| LightGBM | card1_card2_addr1_productcd_target_encoded_fraud_rate | 5 | 0.22966 | -0.33856 | -0.02699 | 0.60604 | 0.43980 |
| LightGBM | card1_addr1_deviceinfo_target_encoded_fraud_rate | 6 | 0.20280 | -0.17916 | -0.03262 | 0.74988 | 0.39880 |
| XGBoost | card1_card2_addr1_target_encoded_fraud_rate | 52 | 0.03938 | -0.11348 | 0.00098 | 0.06521 | 0.51400 |
| XGBoost | card1_addr1_p_emaildomain_target_encoded_fraud_rate | 1 | 0.64649 | -1.10446 | -0.53621 | 1.11651 | 0.23740 |
| XGBoost | card1_card2_addr1_productcd_target_encoded_fraud_rate | 2 | 0.34646 | -0.59813 | -0.29937 | 0.47786 | 0.23260 |
| XGBoost | card1_addr1_deviceinfo_target_encoded_fraud_rate | 9 | 0.16749 | -0.29332 | -0.11514 | 0.28661 | 0.25660 |

## Diagnostic 0.50 error classes and transaction amount

The 0.50 **calibrated** probability cutoff is for TP/FP/FN/TN diagnostics only, not a production threshold. The actual business policy is analyzed separately.

| diagnostic_class | count | mean | median | sum |
| --- | --- | --- | --- | --- |
| FN | 1875 | 185.30847 | 88.50000 | 347453.38600 |
| FP | 179 | 124.60764 | 59.14400 | 22304.76800 |
| TN | 85360 | 133.88333 | 67.95000 | 11428281.33800 |
| TP | 1167 | 128.06701 | 64.55400 | 149454.20200 |

Amount-band boundaries come from TRAIN quantiles [0.0, 42.95, 68.95, 125.0, 280.0, inf]. Counts, error rates, and missed fraud amount:

| band | amount_min | amount_max | count | fraud_prevalence | TP | FP | FN | TN | recall | precision | false_positive_rate | false_negative_rate | fraud_amount | missed_fraud_amount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.00000 | 42.95000 | 21712 | 0.03708 | 357 | 63 | 448 | 20844 | 0.44348 | 0.85000 | 0.00301 | 0.55652 | 19913.58900 | 10945.84700 |
| 1 | 42.95000 | 68.95000 | 23107 | 0.02571 | 245 | 30 | 349 | 22483 | 0.41246 | 0.89091 | 0.00133 | 0.58754 | 32030.95400 | 19092.10900 |
| 2 | 68.95000 | 125.00000 | 22127 | 0.03145 | 312 | 39 | 384 | 21392 | 0.44828 | 0.88889 | 0.00182 | 0.55172 | 65896.50700 | 37251.53200 |
| 3 | 125.00000 | 280.00000 | 12976 | 0.03815 | 145 | 32 | 350 | 12449 | 0.29293 | 0.81921 | 0.00256 | 0.70707 | 91153.16100 | 66248.78700 |
| 4 | 280.00000 | inf | 8659 | 0.05220 | 108 | 15 | 344 | 8192 | 0.23894 | 0.87805 | 0.00183 | 0.76106 | 287913.37700 | 213915.11100 |

## High-value fraud misses and customer friction

False-negative details:

- false_negative_count: 1875
- fraudulent_amount_missed: 347453.386
- missed_amount_share: 0.6992313951140549
- median_missed_amount: 88.5
- mean_missed_amount: 185.30847253333334
- p90_missed_amount: 445.0
- p95_missed_amount: 678.3849999999978
- max_missed_amount: 3822.95
- median_fn_probability: 0.10147685687336051
- p90_fn_probability: 0.3754382583472086
- high_value_threshold: 280.0
- high_value_fn_count: 344
- high_value_fn_amount: 213915.111
- high_value_fn_approve_count: 21
- fraud_approved_count: 709
- fraud_approved_amount: 49739.884

Balanced decisions among diagnostic false negatives: {'REVIEW': 1004, 'APPROVE': 707, 'BLOCK': 164}. A model false negative can still enter REVIEW or BLOCK because decisioning uses amount-dependent expected costs. High-value uses the TRAIN 90th-percentile amount cutoff. Legitimate customer friction:

- legitimate_review_count: 11865
- legitimate_review_rate: 0.13870865920808054
- legitimate_block_count: 475
- legitimate_block_rate: 0.00555302259787933
- legitimate_high_risk_count: 179
- intervened_amount_median: 226.0
- intervened_amount_p90: 884.0
- intervened_probability_median: 0.05388258993015413
- intervened_probability_p90: 0.17589332598226706

Features most prominent within the representative sample's legitimate REVIEW/BLOCK rows (descriptive, noncausal): LightGBM ['card1_addr1_p_emaildomain_target_encoded_fraud_rate', 'TransactionDT', 'C13', 'card1_addr1_deviceinfo_target_encoded_fraud_rate', 'card1_card2_addr1_productcd_target_encoded_fraud_rate']; XGBoost ['card1_addr1_p_emaildomain_target_encoded_fraud_rate', 'card1_card2_addr1_productcd_target_encoded_fraud_rate', 'C14', 'amount_vs_addr1_previous_avg', 'C1'].

## Calibrated risk bands and supported slices

| risk_band | count | fraud_count | observed_fraud_rate | mean_calibrated_probability | transaction_amount | fraud_amount | approve_rate | review_rate | block_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0–1% | 44675 | 159 | 0.00356 | 0.00385 | 4859626.77000 | 22135.59000 | 0.99369 | 0.00631 | 0.00000 |
| 1–2% | 15306 | 151 | 0.00987 | 0.01446 | 2073109.09700 | 26670.53200 | 0.92872 | 0.07128 | 0.00000 |
| 2–5% | 16772 | 331 | 0.01974 | 0.03135 | 2866084.95700 | 62591.31400 | 0.73581 | 0.26312 | 0.00107 |
| 5–10% | 6231 | 291 | 0.04670 | 0.06880 | 1191822.70100 | 61867.71400 | 0.37362 | 0.61579 | 0.01059 |
| 10–20% | 2773 | 386 | 0.13920 | 0.13728 | 540845.35600 | 83120.49000 | 0.18283 | 0.76560 | 0.05157 |
| 20–40% | 1227 | 410 | 0.33415 | 0.27646 | 204093.47100 | 66120.70000 | 0.08802 | 0.74735 | 0.16463 |
| 40%+ | 1597 | 1314 | 0.82279 | 0.73868 | 211911.34200 | 174401.24800 | 0.01002 | 0.23168 | 0.75830 |

Only category groups with at least 500 validation transactions are shown. Selected supported groups (highest count first):

| column | category | count | fraud_prevalence | mean_calibrated_risk | recall_at_0_50 | legitimate_intervention_rate | fraud_amount_intervention_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeviceType | <MISSING> | 73350 | 0.02117 | 0.02461 | 0.19639 | 0.14389 | 0.88798 |
| ProductCD | W | 72291 | 0.02054 | 0.02388 | 0.19394 | 0.14363 | 0.88820 |
| card6 | debit | 69738 | 0.02329 | 0.02711 | 0.34298 | 0.10617 | 0.84892 |
| card4 | visa | 57747 | 0.03436 | 0.03592 | 0.37550 | 0.14077 | 0.89422 |
| P_emaildomain | gmail.com | 35005 | 0.04259 | 0.04475 | 0.43260 | 0.15620 | 0.88749 |
| card4 | mastercard | 29116 | 0.03115 | 0.03624 | 0.41786 | 0.13882 | 0.87942 |
| card6 | credit | 18840 | 0.07527 | 0.07202 | 0.43018 | 0.29319 | 0.94214 |
| P_emaildomain | yahoo.com | 15674 | 0.01927 | 0.02580 | 0.29470 | 0.14722 | 0.91226 |
| P_emaildomain | <MISSING> | 14130 | 0.02951 | 0.02749 | 0.25180 | 0.11092 | 0.84103 |
| ProductCD | C | 9125 | 0.12603 | 0.11832 | 0.59130 | 0.10583 | 0.90254 |
| DeviceType | desktop | 8741 | 0.09713 | 0.09808 | 0.63369 | 0.15902 | 0.93614 |
| DeviceType | mobile | 6490 | 0.09861 | 0.09014 | 0.50625 | 0.12889 | 0.93318 |
| P_emaildomain | hotmail.com | 6322 | 0.04239 | 0.05245 | 0.48507 | 0.12785 | 0.81699 |
| P_emaildomain | anonymous.com | 5010 | 0.01996 | 0.02927 | 0.26000 | 0.18228 | 0.84615 |
| P_emaildomain | aol.com | 4269 | 0.02998 | 0.03114 | 0.45312 | 0.14127 | 0.94906 |
| ProductCD | R | 3365 | 0.04755 | 0.05902 | 0.56250 | 0.27488 | 0.98272 |
| ProductCD | H | 2350 | 0.05957 | 0.07047 | 0.45714 | 0.17330 | 0.93646 |
| ProductCD | S | 1450 | 0.07379 | 0.05349 | 0.42056 | 0.04617 | 0.84571 |
| P_emaildomain | comcast.net | 1144 | 0.07430 | 0.04127 | 0.15294 | 0.20113 | 0.97320 |
| P_emaildomain | icloud.com | 999 | 0.03804 | 0.03402 | 0.42105 | 0.13319 | 0.86897 |

The complete supported-slice table is reports/supported_categorical_slices.csv. Slice differences did not change the policy.

## Component disagreement

- pearson: 0.8967599430353742
- spearman: 0.9061092065511711
- mean_absolute_gap: 0.050877491442805345
- mean_absolute_rank_gap: 0.09132422756820806
- p90_absolute_rank_gap: 0.20791140312256573
- rank_gap_ge_0_20_rate: 0.11064449486910286
- median_absolute_gap: 0.020174876197656105
- p90_absolute_gap: 0.14170550866866305
- high_gap_threshold: 0.14170550866866305
- high_gap_count: 8859
- high_gap_fraud_rate: 0.08432102946156451
- high_gap_false_negative_rate: 0.08149904052376114
- high_gap_fraud_amount: 148272.75900000002
- xgboost_higher_on_lgb_low_fraud_rate: 0.8349885408708938

High disagreement means the top 10% of absolute LightGBM–XGBoost score gaps on VALIDATION. The XGBoost-higher rate among fraud rows scored below 0.50 by LightGBM checks whether XGBoost adds signal there; it does not establish a causal or independently deployable benefit. Earlier OOF ensemble evidence selected the frozen 80/20 blend; no weight is reconsidered here.

## Anonymized local cases

Case IDs are report-local aliases; no original TransactionID is displayed. The model columns list separate signed raw-margin component SHAP drivers. The final four cost/action columns explain the Balanced business decision.

| case | kind | true_fraud | lightgbm_score | xgboost_score | raw_ensemble_score | calibrated_probability | TransactionAmt | ProductCD | DeviceType | card4 | card6 | lightgbm_drivers | xgboost_drivers | expected_approve_cost | expected_review_cost | expected_block_cost | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | high_risk_tp | 1 | 0.99992 | 0.99987 | 0.99991 | 0.99720 | 25.26800 | C | mobile | visa | debit | card1_addr1_p_emaildomain_target_encoded_fraud_rate (+1.759), V258 (+1.603), V45 (+1.363), C1 (+1.248), C12 (+0.817) | V258 (+1.370), C1 (+1.095), card1_addr1_p_emaildomain_target_encoded_fraud_rate (+0.868), V45 (+0.858), card1_card2_addr1_productcd_target_encoded_fraud_rate (+0.506) | 25.19718 | 8.04224 | 0.57007 | BLOCK |
| C02 | false_positive | 0 | 0.99929 | 0.99831 | 0.99909 | 0.98241 | 176.44800 | C | desktop | mastercard | credit | V258 (+1.807), C1 (+1.454), card1_addr1_p_emaildomain_target_encoded_fraud_rate (+1.240), C13 (+1.233), V187 (+0.581) | V258 (+1.559), C1 (+1.132), card1_addr1_p_emaildomain_target_encoded_fraud_rate (+0.756), C13 (+0.583), V187 (+0.461) | 173.34407 | 37.68641 | 0.93978 | BLOCK |
| C03 | false_negative | 1 | 0.00038 | 0.00100 | 0.00050 | 0.00060 | 35.95000 | W | nan | mastercard | debit | C5 (-0.577), TransactionDT (-0.473), card1_addr1_p_emaildomain_target_encoded_fraud_rate (-0.452), card1_card2_addr1_productcd_target_encoded_fraud_rate (-0.281), V82 (-0.246) | card1_addr1_p_emaildomain_target_encoded_fraud_rate (-1.035), card1_card2_addr1_productcd_target_encoded_fraud_rate (-0.704), C5 (-0.544), C13 (-0.420), TransactionDT (-0.280) | 0.02154 | 4.00371 | 25.48502 | APPROVE |
| C04 | low_risk_tn | 0 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 30.92400 | C | desktop | mastercard | credit | id_17 (-1.336), V55 (-1.153), D13 (-1.098), V258 (+0.659), TransactionDT (-0.634) | card3 (-1.087), D13 (-0.811), id_17 (-0.686), card1_addr1_p_emaildomain_target_encoded_fraud_rate (-0.672), V55 (-0.572) | 0.00014 | 4.00002 | 25.49989 | APPROVE |
| C05 | review | 0 | 0.97942 | 0.99184 | 0.98190 | 0.84100 | 6.39400 | C | desktop | visa | debit | card1_addr1_p_emaildomain_target_encoded_fraud_rate (+1.488), C1 (+1.465), V258 (+0.818), card1_addr1_deviceinfo_target_encoded_fraud_rate (+0.700), C13 (+0.663) | C1 (+1.119), card1_addr1_p_emaildomain_target_encoded_fraud_rate (+1.040), V258 (+0.855), card1_card2_addr1_productcd_target_encoded_fraud_rate (+0.618), C11 (+0.450) | 5.37732 | 4.23447 | 4.47512 | REVIEW |
| C06 | block | 1 | 0.99992 | 0.99987 | 0.99991 | 0.99720 | 25.26800 | C | mobile | visa | debit | card1_addr1_p_emaildomain_target_encoded_fraud_rate (+1.759), V258 (+1.603), V45 (+1.363), C1 (+1.248), C12 (+0.817) | V258 (+1.370), C1 (+1.095), card1_addr1_p_emaildomain_target_encoded_fraud_rate (+0.868), V45 (+0.858), card1_card2_addr1_productcd_target_encoded_fraud_rate (+0.506) | 25.19718 | 8.04224 | 0.57007 | BLOCK |
| C07 | high_value_fraud_miss | 1 | 0.25086 | 0.25022 | 0.25073 | 0.08932 | 3822.95000 | W | nan | visa | credit | amount_vs_addr1_previous_avg (+0.440), card1_card2_addr1_productcd_target_encoded_fraud_rate (+0.323), TransactionDT (-0.321), C13 (+0.295), card1_addr1_p_emaildomain_previous_tx_count (+0.279) | amount_vs_addr1_previous_avg (+0.309), card1_addr1_p_emaildomain_previous_tx_count (+0.280), card6 (+0.269), V91 (-0.256), V70 (-0.251) | 341.46842 | 72.20436 | 23.26698 | BLOCK |

## Limitations and future monitoring implications

TreeSHAP samples model behavior, not fraud causality. Component raw-margin SHAP cannot be added across boosters to explain the calibrated blend. A uniform 5,000-row sample supports global diagnostics but rare segments and local examples have higher variance. REVIEW capture uses illustrative effectiveness; actual reviewer outcomes are unavailable. The 0.50 error cutoff is diagnostic only. Future monitoring should watch calibration by risk and amount band, high-value misses, legitimate interventions, category support, and component disagreement. No monitoring system, model change, or new business rule was implemented.

## Runtime, memory, MLflow

Notebook runtime: 220.4 seconds. Prepared feature-matrix estimate: 905.7 MiB; SHAP matrices contain 5,000 × 467 values per model. MLflow experiment: fraud-detection-explainability-error-analysis; tracking URI: sqlite:///C:/fraud-detection-system/model_comparison.db; run ID: 94961ff053e04c7a8fbe782567b96430. Seven figures are under reports/figures/ and logged as run artifacts.

## Verification gates

The executed notebook contains 13 completed code cells and no error outputs. Native LightGBM and XGBoost raw-margin SHAP additivity assertions passed on the deterministic sample. MLflow run `94961ff053e04c7a8fbe782567b96430` has status `FINISHED`, records the reproduced validation metrics, and contains the seven figure artifacts.

- `pytest`: 163 passed (8 existing scikit-learn deprecation warnings).
- `ruff check .`: passed.
- `ruff format --check .`: passed (88 files already formatted).
