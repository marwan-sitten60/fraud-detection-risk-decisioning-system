# Project reports

Use the root [README](../README.md) for the short project overview. The reports below provide the supporting evidence.

## Research and modeling

- [Frozen system manifest](frozen_system_manifest.md) — canonical description of the selected model, feature set, calibration, policy, and fit scope.
- [Model comparison](model_comparison_summary.md) — Logistic Regression, LightGBM, and XGBoost baselines on chronological development data.
- [Feature engineering](feature_engineering_phase1_summary.md) — frequency encoding and missingness/time transformations.
- [Behavioral features](behavioral_features_summary.md) — past-only entity and transaction history experiments.
- [Composite identity features](composite_identity_features_summary.md) — chronological composite-frequency features.
- [Temporal target encoding](temporal_target_encoding_summary.md) — fold-safe past-only target statistics and selected 467-feature pipeline.
- [Graph features](graph_features_summary.md) — historical entity-link experiments and their small, mixed validation effects.
- [Time-decay weighting](time_decay_weighting_summary.md) — comparison that retained unweighted fitting.
- [CatBoost challenger](catboost_challenge_summary.md) — native-categorical challenger and reason it was not selected.
- [Ensemble challenge](ensemble_challenge_summary.md) — out-of-fold 80/20 LightGBM/XGBoost blend selection.

## Evaluation and decisioning

- [Final HOLDOUT evaluation](final_holdout_evaluation_summary.md) — locked terminal evaluation, ranking, calibration, and policy outcomes.
- [Probability calibration](probability_calibration_summary.md) — TRAIN-OOF Platt selection and validation checks.
- [Business decisioning](business_decisioning_summary.md) — illustrative expected-cost assumptions and action policy.
- [Optional rules](rules_engine_summary.md) — rules overlay comparisons and disabled-by-default decision.
- [Explainability and error analysis](explainability_error_analysis_summary.md) — component-model TreeSHAP and validation diagnostics.

## Productionization

- [Frozen inference bundle](frozen_inference_bundle_summary.md) — artifact contents, checksums, input contract, and runtime parity.
- [FastAPI serving](fastapi_serving_summary.md) — request contract, ordering, readiness, and local runtime behavior.
- [PostgreSQL persistence](postgresql_persistence_summary.md) — durable requests, idempotency, atomic writes, and recovery design.
- [Live PostgreSQL verification](postgresql_live_verification.md) — Phase 3.5 live integration results.
- [Docker deployment](docker_deployment_summary.md) — Phase 8 Compose build, health, persistence, restart, and parity verification.

## Feedback, monitoring, and operations

- [Delayed labels and feedback](delayed_labels_feedback_summary.md) — timestamp/maturity contract and stored-prediction evaluation.
- [Production monitoring](production_monitoring_summary.md) — fixed reference, drift metrics, alerts, and mature-label requirements.
- [Traffic simulator](traffic_simulator_summary.md) — chronological HTTP replay, scheduling, delayed labels, and local throughput limits.
- [FraudGuard dashboard](dashboard_summary.md) — six pages, API-only access, and display/privacy boundaries.

## Final release

- [Final project release summary](final_project_release_summary.md) — project results, repository/security audit, verification, limitations, and release recommendation.

Raw execution logs, temporary simulator outputs, and local database files are intentionally excluded from this index.

