# Interview story: Fraud Detection & Risk Decisioning System

## A. 30-second explanation

I built a portfolio fraud risk decisioning system using the IEEE-CIS dataset. I treated it as a temporal problem: I engineered past-only behavioral and identity features, compared models with chronological evaluation, selected an 80/20 LightGBM/XGBoost blend, calibrated it with TRAIN out-of-fold Platt scaling, and mapped risk to APPROVE, REVIEW, or BLOCK using explicit illustrative costs. I then connected the frozen bundle to FastAPI, PostgreSQL, delayed-label feedback, monitoring, a chronological simulator, FraudGuard, and Docker Compose. The later HOLDOUT PR-AUC was lower than validation, so I report the temporal generalization gap rather than tuning against it.

## B. Two-minute explanation

Fraud is rare and patterns change over time, so random splits and accuracy can give misleading results. I used the IEEE-CIS labeled transaction data, ordered by TransactionDT, with 413,378 TRAIN rows, 88,581 development VALIDATION rows, and 88,581 later terminal HOLDOUT rows.

The final feature set has 467 columns. It combines the raw tabular signal with frequency encoding, past-only amount and composite-identity history, and temporally safe target-encoded statistics. Preprocessing and statistics are rebuilt inside expanding TRAIN folds. Same-timestamp rows do not see each other. The final system blends LightGBM and XGBoost at 80/20; that blend improved out-of-fold PR-AUC by about 0.0031 over LightGBM. A Platt calibrator selected on TRAIN OOF predictions makes the risk estimate usable in expected-cost calculations.

The decision policy compares assumed fraud, review, and false-decline costs per transaction amount. On the terminal HOLDOUT the risk ranking reached PR-AUC 0.5369 and ROC-AUC 0.8786, below development validation at 0.6153 and 0.9162. The drop is evidence of temporal distribution shift and is part of the result. Reported fraud capture under the Balanced rules-disabled policy is modeled and assumes 80% review effectiveness.

The software side includes one-worker FastAPI, PostgreSQL-backed request and prediction history, idempotent retries, atomic batch writes, restart replay, delayed labels, drift and health monitoring, a simulator, and a read-only Streamlit dashboard. Docker Compose starts the database, migration job, API, and dashboard. It is a local portfolio demonstration, not a bank deployment or horizontally scalable service.

## C. Deep technical explanation

### Data and temporal validation

The full labeled transaction file has 590,540 rows, around 3.50% fraud. The chronological partitions are 413,378 TRAIN, 88,581 VALIDATION, and 88,581 HOLDOUT. TRAIN itself has 14,538 fraud rows (3.52%). All fit-derived preprocessing, frequency maps, model fitting, calibration fitting, and historical state use data available before the row being evaluated. Expanding temporal folds preserve future direction. The final HOLDOUT was scored and locked before its labels were opened for terminal evaluation; that result was not fed back into feature or policy selection.

### Selected model and decision

The production bundle has 467 features. LightGBM and XGBoost share that matrix and are combined with weights 0.8 and 0.2. OOF blend comparison showed a small PR-AUC increase over LightGBM alone. Platt calibration is fitted on TRAIN OOF outputs, not HOLDOUT. The decision layer is a separate expected-cost calculation: calibrated probability and transaction amount feed expected costs for approve, review, and block. The minimum-cost action is selected using illustrative fraud-loss, review-effectiveness, review-cost, false-decline, and operational-cost assumptions. It is not a learned 0.5 cutoff.

### Experiments and diagnostics

Development experiments covered a Logistic Regression baseline, boosting models, frequency and missingness/time features, past-only behavior, composite identities, graph relations, temporal target encoding, CatBoost, sample weighting, ensembling, calibration, rules, and TreeSHAP. Historical amount features and composite identities improved earlier chronological validation; adding composite temporal target encoding raised the candidate PR-AUC by 0.0168 over its preceding configuration. The 80/20 blend gave a further roughly 0.0031 OOF PR-AUC gain over LightGBM.

Not every experiment was kept. CatBoost scored 0.5664 validation PR-AUC versus the LightGBM reference at 0.5935 and required a heavier native-categorical path. Sample weighting selected unweighted fitting; the 90-day half-life mean CV PR-AUC (0.5826) was below unweighted (0.5849). Graph ablations were mixed: the best tested set added only 0.000673 PR-AUC over the immediately preceding feature candidate, while adding state complexity; it was not part of the final bundle. These are development findings, not reasons to revisit the locked HOLDOUT.

TreeSHAP explains each booster in raw-margin space. It is used for model interpretation, error analysis, and diagnosis; it is neither causal analysis nor an exact attribution of the calibrated policy action.

### Serving and state

A prediction contains exactly the frozen 433 raw input columns and no label. Requests must advance by (TransactionDT, TransactionID). The runtime computes derived features, component scores, blend, calibrated probability, and decision in one ordered process. PostgreSQL stores canonical label-free payloads, request sequence and batch boundaries, predictions, and audit events. A successful batch is committed atomically.

An X-Idempotency-Key maps the canonical request to its stored result. An identical retry is returned without inference or state mutation; a different payload with the same key returns 409. Reusing a previously scored TransactionID under another key returns 409.

At startup the service verifies the bundle and Alembic schema, loads the runtime, and replays only successful requests for that bundle in durable order with original batch boundaries. Recovery replays raw label-free transaction payloads and does not read fraud_labels. The readiness endpoint stays unavailable until the database check and recovery finish.

The API intentionally runs one worker. Each process has mutable chronological history; multiple workers could independently score the same next transaction with different state. PostgreSQL provides restart recovery but not live cross-worker state coordination.

### Labels and monitoring

A trusted label source can later send an observed outcome and its matured_at timestamp. The feedback service joins mature labels to the original stored prediction. Ingestion does not update frozen model state, calibration, or policy.

Before labels mature, the system can report input PSI, categorical Jensen-Shannon divergence, score and decision drift, entity novelty, service errors and latency, request volume, and label coverage. Mature labels are needed for PR-AUC, ROC-AUC, Brier score, and other outcome measures. Drift is a distribution change, not proof of failure.

### Current operating limits and future scale

The current service is local/single-process, without authentication or authorization, and has no automatic retraining or production SLA. Feature generation is the dominant local single-request cost.

Safe scaling would need deterministic ordered feature-state updates shared between workers, such as a stateful stream processor or external feature/state service with well-defined key partitioning and replay. Simply increasing Uvicorn workers or replicas is unsafe.

A future retraining process could use sufficiently mature labels to form a candidate dataset, then perform chronological training and validation, calibration, champion/challenger comparison, shadow or canary evaluation, and controlled promotion. The current project does not implement this process.

## D. Likely interview questions

### Why PR-AUC?

Fraud is rare, so precision-recall performance focuses on the positive class and the precision/recall trade-off. The no-skill PR reference is close to prevalence. ROC-AUC is reported as a secondary ranking view, but PR-AUC is the primary comparison metric.

### Why a temporal split?

In service, we predict future transactions from past history. A random split can leak future distributions or entity history into an earlier example. Chronological splits and expanding folds better reflect deployment and expose temporal drift.

### How did you avoid leakage?

Preprocessors and target-derived statistics are fit inside each training fold. Stateful features use strictly earlier rows; same-timestamp batches are scored before state advances. The raw API contract excludes the target. HOLDOUT labels were not available to model, feature, calibration, or policy selection.

### Why historical features?

The same card, address, device, or composite identity can recur. Past counts, recency, and amount behavior can express novelty and deviation that a single row cannot. They are updated in order and never use later transactions for an earlier prediction.

### Why target encoding?

A smoothed historical fraud rate can summarize risk for high-cardinality identities. It is computed fold-locally from prior labels with chronological and same-timestamp isolation. At inference, incoming transactions do not update that label state.

### Why LightGBM and XGBoost?

Both work well on tabular data, while their errors are not identical. The out-of-fold blend search measured whether combination helped. It selected 80/20, which improved PR-AUC by approximately 0.0031 over LightGBM alone on those development OOF predictions.

### Why an 80/20 ensemble?

The weight was selected on aligned, chronological OOF probabilities rather than HOLDOUT. It gave a small but measurable OOF gain; the final external validation assessment reproduced the selected champion. The added second model also costs more to serve and version.

### Why calibrate probabilities?

A ranking score orders examples but need not represent the chance of fraud. The expected-cost layer needs meaningful probabilities to calculate expected fraud exposure and expected false-decline or review trade-offs. Platt calibration was selected from TRAIN OOF comparisons.

### Why not use a 0.5 probability cutoff?

The action depends on transaction amount and assumed fraud, review, and false-decline costs. A 0.5 cutoff ignores that different amounts imply different expected losses. The 0.5 confusion matrix is a diagnostic only.

### Why expected-cost decisioning?

It makes business assumptions explicit and compares expected costs of APPROVE, REVIEW, and BLOCK. The costs and review effectiveness are illustrative because IEEE-CIS does not provide actual bank costs or review outcomes. Therefore modeled savings are not claimed as real savings.

### Why did HOLDOUT PR-AUC drop?

The later period is temporally different from development validation. PR-AUC changed from 0.615265 to 0.536900, and ROC-AUC from 0.916224 to 0.878637. This is the reason for future-time evaluation. The locked result was not used to retune the model.

### Why do fraud labels arrive later?

A transaction may only be confirmed through a later chargeback, report, or analyst resolution. Prediction is immediate; trusted ground truth and its maturity time arrive later. The system stores labels separately and scores only once.

### What can you monitor before labels arrive?

Input distributions, missingness, numeric PSI, categorical Jensen-Shannon divergence, score distributions, decision mix, entity novelty, API error/latency, readiness, and label coverage. These signals detect operational or distribution changes, not confirmed accuracy loss.

### What requires mature labels?

Outcome-based metrics such as PR-AUC, ROC-AUC, Brier score, fraud capture, and decision outcomes. The service keeps them unavailable or marks support insufficient until the maturity and sample-size rules are met.

### Why one FastAPI worker?

Chronological feature history is mutable process-local state. Different workers would hold different histories and may score a transaction inconsistently. One worker preserves order in this demonstration.

### How does restart recovery work?

The service verifies the immutable bundle and database schema, then replays successful requests for that bundle ordered by a durable request sequence. It restores original batch boundaries and transaction order from stored label-free payloads. Recovery writes no duplicate predictions and does not query labels. Readiness waits until this finishes.

### Why PostgreSQL?

It provides transactional persistence and constraints for unique transactions, idempotency keys, request boundaries, predictions, and audit events. Successful batch persistence is atomic. The same durable request stream also supports recovery after a process restart.

### How does idempotency work?

The client sends X-Idempotency-Key. The service hashes the canonical ordered scoring request and stores the result. A retry with the same key and same request returns the stored response without rescoring. Reusing a key with changed payload or endpoint is a 409 conflict.

### Why reject CatBoost, graph features, or time-decay weighting?

CatBoost did not beat the preferred LightGBM candidate and required a heavier categorical serving path. Graph results varied; their best validation gain was only about 0.0007 over the prior candidate and the extra state was not retained in the final bundle. Unweighted fitting won the tested time-decay comparison, so weighting was not used. Each choice followed chronological development evidence, not HOLDOUT outcomes.

### Why are the rules disabled by default?

Rules were evaluated as an overlay. They improved some fraud-capture measures but increased modeled expected cost and review burden under the illustrative assumptions. They remain available for explicit analysis, while the primary frozen policy uses the rules-disabled result.

### How would you scale this?

First define a shared deterministic ordering and state-update contract. Then an ordered state service or stream processor could partition compatible entity state, with persisted offsets and replay. A shared external store and idempotent updates could support multiple API workers only after parity and concurrency tests. Simply adding replicas is unsafe today.

### How would retraining work later?

Only after labels mature and are quality-controlled would a candidate training cohort be built. A separate temporal train/validation cycle would compare candidates, fit calibration, and evaluate decision assumptions. Candidates could then run in champion/challenger and shadow or canary mode before a governed promotion. There is no automatic retraining in this project.

