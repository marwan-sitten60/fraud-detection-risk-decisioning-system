"""Delayed-label coverage and matured-label performance."""

from __future__ import annotations

from dashboard.client import APIClient
from dashboard.components import select_window, show_api_error


def _metric(st, label: str, metrics: dict, key: str, percent: bool = False):
    value = metrics.get(key)
    if value is None:
        shown = "N/A"
    elif percent:
        shown = f"{value:.1%}"
    else:
        shown = f"{value:.4f}"
    st.metric(label, shown)


def render(api=None):
    import streamlit as st

    api = api or APIClient()
    st.title("Feedback")
    st.caption(
        "Only backend-defined mature-label semantics are used. Local label delays are synthetic."
    )
    window = select_window(st, "feedback_window")
    result = api.get("/feedback/summary", window)
    if show_api_error(st, result):
        return
    data = result.data or {}
    metrics = data.get("metrics") or {}
    columns = st.columns(6)
    columns[0].metric("Predictions", data.get("total_predictions", 0))
    columns[1].metric("Labels received", data.get("labels_received", 0))
    columns[2].metric("Mature labels", data.get("mature_labels", 0))
    columns[3].metric("Immature labels", data.get("immature_labels", 0))
    coverage = data.get("mature_label_coverage")
    columns[4].metric("Mature coverage", f"{coverage:.1%}" if coverage is not None else "N/A")
    prevalence = metrics.get("fraud_prevalence")
    columns[5].metric(
        "Fraud prevalence · mature",
        f"{prevalence:.1%}" if prevalence is not None else "N/A",
    )
    status = data.get("status", "unavailable")
    if status == "insufficient_labels":
        st.warning(
            "Performance metrics are unavailable: insufficient_labels "
            f"({data.get('mature_labels', 0)} mature labels). Metrics below are omitted."
        )
    elif status != "available":
        st.info(f"Performance status: {status}.")
    else:
        st.subheader("Matured-label performance")
        layout = st.columns(4)
        for column, label, key in zip(
            layout,
            ("PR-AUC", "ROC-AUC", "Brier score", "Log loss"),
            ("pr_auc", "roc_auc", "brier", "log_loss"),
            strict=True,
        ):
            with column:
                _metric(st, label, metrics, key)
        layout = st.columns(4)
        for column, label, key in zip(
            layout,
            ("Precision", "Recall", "F1", "Fraud transaction capture"),
            (
                "diagnostic_precision",
                "diagnostic_recall",
                "diagnostic_f1",
                "fraud_transaction_capture",
            ),
            strict=True,
        ):
            with column:
                _metric(st, label, metrics, key, percent=True)
        layout = st.columns(3)
        for column, label, key in zip(
            layout,
            ("Fraud amount capture", "Legitimate REVIEW rate", "Legitimate BLOCK rate"),
            ("fraud_amount_capture", "legitimate_review_rate", "legitimate_block_rate"),
            strict=True,
        ):
            with column:
                _metric(st, label, metrics, key, percent=True)
    st.subheader("Label delay")
    delays = data.get("label_delays") or {}
    left, right = st.columns(2)
    for column, label, key in (
        (left, "Prediction → observed", "prediction_to_observed"),
        (right, "Observed → maturity", "observed_to_maturity"),
    ):
        delay = delays.get(key) or {}
        column.metric(
            f"{label} median",
            f"{delay['median_seconds']:,.2f} sec"
            if delay.get("median_seconds") is not None
            else "N/A",
        )
        column.caption(f"p90 {delay.get('p90_seconds', 'N/A')} sec · n={delay.get('count', 0)}")
    if delays.get("transaction_to_observed_reason"):
        st.caption(delays["transaction_to_observed_reason"])
    st.caption(
        "Synthetic and development replay delays do not represent a financial institution’s "
        "reporting delay."
    )
