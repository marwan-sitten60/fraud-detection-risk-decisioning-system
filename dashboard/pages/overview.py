"""Executive overview of monitoring and service activity."""

from __future__ import annotations

import pandas as pd

from dashboard.client import APIClient
from dashboard.components import pct, select_window, show_api_error
from dashboard.formatting import format_risk, sort_alerts


def render(api=None):
    import streamlit as st

    api = api or APIClient()
    st.title("Operational overview")
    st.caption("Traffic reflects local software tests, synthetic inputs, and development replay.")
    window = select_window(st)
    left, right = st.columns([1, 3])
    with left:
        st.caption(f"Monitoring window: {window['window']} UTC")
    summary_result = api.get("/monitoring/summary", window)
    alerts_result = api.get("/monitoring/alerts", {"limit": 100})
    feedback_result = api.get("/feedback/summary")
    if show_api_error(st, summary_result):
        return
    summary = summary_result.data or {}
    system = summary.get("system", {})
    output = system.get("model_output", {})
    distribution = output.get("decision_distribution", {})
    feedback = feedback_result.data if feedback_result.ok else {}
    coverage = feedback.get("mature_label_coverage") if feedback else None
    alert_items = (alerts_result.data or {}).get("alerts", []) if alerts_result.ok else []
    warning_count = sum(item.get("severity") == "WARNING" for item in alert_items)
    critical_count = sum(item.get("severity") == "CRITICAL" for item in alert_items)

    columns = st.columns(4)
    columns[0].metric("Transactions", f"{system.get('transactions_scored', 0):,}")
    columns[1].metric("Mean calibrated risk", format_risk(output.get("mean_calibrated_risk")))
    columns[2].metric("Mature label coverage", pct(coverage))
    columns[3].metric("Persisted warning / critical", f"{warning_count} / {critical_count}")
    st.caption(
        "Alert counts use the latest 100 persisted alerts; the API does not expose "
        "an active/cleared state."
    )
    st.caption(
        "Risk is the calibrated fraud probability. The decision also reflects expected-cost policy."
    )

    decision_cols = st.columns(3)
    for col, action in zip(decision_cols, ("APPROVE", "REVIEW", "BLOCK"), strict=True):
        action_value = distribution.get(action, {})
        col.metric(f"{action} rate", pct(action_value.get("rate")))
    chart_col, latency_col = st.columns(2)
    with chart_col:
        st.subheader("Decision mix")
        decision_df = pd.DataFrame(
            [
                {"Decision": action, "Transactions": values.get("count", 0)}
                for action, values in distribution.items()
            ]
        )
        if decision_df.empty:
            st.info("No scored transactions in this window.")
        else:
            st.bar_chart(decision_df.set_index("Decision"), y="Transactions", color="#3276a8")
    with latency_col:
        st.subheader("Request activity and latency")
        latency = system.get("latency_ms") or {}
        st.metric("Requests", f"{system.get('request_count', 0):,}")
        latency_df = pd.DataFrame(
            [
                {"Percentile": key.upper(), "Milliseconds": val}
                for key, val in latency.items()
                if val is not None
            ]
        )
        if latency_df.empty:
            st.info("Latency metrics are unavailable for this window.")
        else:
            st.bar_chart(latency_df.set_index("Percentile"), y="Milliseconds", color="#6b8e70")
    risk_result = api.get("/predictions", {"limit": 200, **window})
    st.subheader("Recent calibrated risk distribution")
    if risk_result.ok:
        items = (risk_result.data or {}).get("items", [])
        if items:
            histogram = pd.cut(
                [row["calibrated_probability"] for row in items],
                bins=[index / 10 for index in range(11)],
                include_lowest=True,
            )
            counts = pd.Series(histogram).value_counts(sort=False)
            risk_df = pd.DataFrame(
                {"Transactions": counts.to_numpy()}, index=counts.index.astype(str)
            )
            risk_df.index.name = "Calibrated probability range"
            st.bar_chart(risk_df, y="Transactions", color="#c98542")
        else:
            st.info("No prediction records in this window.")
    else:
        st.caption("Recent risk distribution is unavailable from the API.")
    st.subheader("Recent alerts")
    if alerts_result.ok and alert_items:
        top = sort_alerts(alert_items)[:8]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Severity": row.get("severity"),
                        "Type": row.get("alert_type"),
                        "Metric": row.get("metric_name"),
                        "Observed": row.get("observed_value"),
                        "Reference": row.get("threshold_reference"),
                        "Time": row.get("created_at"),
                    }
                    for row in top
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    elif alerts_result.ok:
        st.info("No persisted alerts are available.")
    if feedback:
        st.caption(
            f"Feedback status: {feedback.get('status', 'unavailable')} · "
            f"{feedback.get('mature_labels', 0)} mature labels"
        )
