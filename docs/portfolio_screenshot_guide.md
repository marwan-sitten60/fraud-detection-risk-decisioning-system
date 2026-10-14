# Portfolio screenshot guide

Screenshots are optional. Capture them manually only from a clean local or synthetic demo, and store reviewed images under docs/images/. The current demo has no authentication; never capture secrets or private desktop content.

Suggested targets:

1. **FraudGuard Overview:** show status marked LOCAL / SIMULATED / DEVELOPMENT REPLAY and aggregate cards or charts.
2. **Monitoring:** show drift and alerts with their selected time window and sample count.
3. **Transactions:** use a clearly synthetic TransactionID and show risk score separately from the business action.
4. **Compose health:** show only service names and healthy state; crop terminal paths, environment variables, and command history.

Before adding an image, verify that it contains no passwords, .env values, credential-bearing database URLs, tokens, raw 433-field payloads, real transaction identifiers, user profile paths, personal desktop data, or HOLDOUT labels. Do not present simulator traffic or alert counts as model performance.

