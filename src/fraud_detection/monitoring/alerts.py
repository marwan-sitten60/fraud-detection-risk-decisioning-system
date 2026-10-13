"""Deterministic alert rules; non-reference cutoffs are explicit operational defaults."""

MIN_PERFORMANCE_ROWS = 200
MIN_PERFORMANCE_PER_CLASS = 20
PSI_WARNING = 0.20
PSI_CRITICAL = 0.30
JS_WARNING = 0.10
JS_CRITICAL = 0.20
ERROR_WARNING = 0.05
ERROR_CRITICAL = 0.15
P95_LATENCY_WARNING_MS = 2000.0
P95_LATENCY_CRITICAL_MS = 5000.0
LOW_COVERAGE_WARNING = 0.10
MIN_DRIFT_ROWS = 30


def make_alert(alert_type, severity, metric, value, threshold, message):
    return {
        "alert_type": alert_type,
        "severity": severity,
        "metric_name": metric,
        "observed_value": None if value is None else float(value),
        "threshold_reference": str(threshold),
        "message": message,
    }


def severity(value, warning, critical):
    if value is None:
        return None
    if value >= critical:
        return "CRITICAL"
    if value >= warning:
        return "WARNING"
    return None


def performance_supported(sample_size: int, fraud_count: int, legitimate_count: int) -> bool:
    return (
        sample_size >= MIN_PERFORMANCE_ROWS
        and fraud_count >= MIN_PERFORMANCE_PER_CLASS
        and legitimate_count >= MIN_PERFORMANCE_PER_CLASS
    )


def build_alerts(data: dict, reference: dict) -> list[dict]:
    alerts = []
    drift_supported = data["system"]["transactions_scored"] >= MIN_DRIFT_ROWS
    for metric, result in data["drift"]["inputs"].items() if drift_supported else ():
        if "psi" in result:
            value = result["psi"]
            alert_type = "INPUT_DRIFT"
        else:
            value = result["js_divergence"]
            alert_type = "INPUT_DRIFT"
        level = (
            severity(value, PSI_WARNING, PSI_CRITICAL)
            if "psi" in result
            else severity(value, JS_WARNING, JS_CRITICAL)
        )
        if level:
            alerts.append(
                make_alert(
                    alert_type,
                    level,
                    metric,
                    value,
                    "PSI .20/.30" if "psi" in result else "JS .10/.20",
                    f"Input distribution changed for {metric}",
                )
            )
    for metric, result in data["drift"]["scores"].items() if drift_supported else ():
        level = severity(result["psi"], PSI_WARNING, PSI_CRITICAL)
        if level:
            alerts.append(
                make_alert(
                    "SCORE_DRIFT",
                    level,
                    metric,
                    result["psi"],
                    "PSI .20/.30",
                    f"Score distribution changed for {metric}",
                )
            )
    decision_result = data["drift"]["decisions"]
    if decision_result and drift_supported:
        level = severity(decision_result["js_divergence"], JS_WARNING, JS_CRITICAL)
        if level:
            alerts.append(
                make_alert(
                    "DECISION_DRIFT",
                    level,
                    "decision_distribution",
                    decision_result["js_divergence"],
                    "JS .10/.20",
                    "Decision mix changed from the TRAIN OOF reference",
                )
            )
    health = data["system"]
    level = severity(health["error_rate"], ERROR_WARNING, ERROR_CRITICAL)
    if level:
        alerts.append(
            make_alert(
                "HIGH_ERROR_RATE",
                level,
                "error_rate",
                health["error_rate"],
                "operational default .05/.15",
                "Request error rate exceeded the operational default",
            )
        )
    latency = health["latency_ms"]["p95"]
    level = severity(latency, P95_LATENCY_WARNING_MS, P95_LATENCY_CRITICAL_MS)
    if level:
        alerts.append(
            make_alert(
                "HIGH_LATENCY",
                level,
                "latency_p95_ms",
                latency,
                "operational default 2000/5000 ms",
                "P95 request latency exceeded the operational default",
            )
        )
    feedback = data["feedback"]
    if (
        health["transactions_scored"] >= 50
        and feedback["mature_label_coverage"] is not None
        and feedback["mature_label_coverage"] < LOW_COVERAGE_WARNING
    ):
        alerts.append(
            make_alert(
                "LOW_LABEL_COVERAGE",
                "WARNING",
                "mature_label_coverage",
                feedback["mature_label_coverage"],
                "operational default < .10",
                "Mature label coverage is below the operational default",
            )
        )
    metrics = feedback["metrics"]
    if metrics["sample_size"]:
        fraud_count = round(metrics["fraud_prevalence"] * metrics["sample_size"])
        legitimate_count = metrics["sample_size"] - fraud_count
        if performance_supported(metrics["sample_size"], fraud_count, legitimate_count):
            baseline = reference.get("performance_reference", {})
            reference_pr = baseline.get("pr_auc")
            if reference_pr and metrics["pr_auc"] < reference_pr * 0.75:
                alerts.append(
                    make_alert(
                        "PERFORMANCE_DEGRADATION",
                        "WARNING",
                        "pr_auc",
                        metrics["pr_auc"],
                        f"< 75% of TRAIN OOF PR-AUC {reference_pr:.4f}",
                        "Mature-label PR-AUC fell below the operational comparison bound",
                    )
                )
            reference_brier = baseline.get("brier")
            if reference_brier is not None and metrics["brier"] > reference_brier + 0.03:
                alerts.append(
                    make_alert(
                        "PERFORMANCE_DEGRADATION",
                        "WARNING",
                        "brier",
                        metrics["brier"],
                        f"> TRAIN OOF Brier + .03 ({reference_brier:.4f})",
                        "Mature-label Brier score exceeded the operational comparison bound",
                    )
                )
    return alerts
