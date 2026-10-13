"""Input, score, decision, novelty, and system monitoring views."""

from __future__ import annotations

import pandas as pd

from dashboard.client import APIClient
from dashboard.components import select_window, show_api_error
from dashboard.formatting import drift_status, sort_alerts

INPUTS = (
    "TransactionAmt",
    "ProductCD",
    "card4",
    "card6",
    "DeviceType",
    "P_emaildomain",
    "R_emaildomain",
)


def render(api=None):
    import streamlit as st

    api = api or APIClient()
    st.title("Monitoring")
    params = select_window(st, "monitoring_page_window")
    result = api.get("/monitoring/summary", params)
    drift_result = api.get("/monitoring/drift", params)
    alerts_result = api.get("/monitoring/alerts", {"limit": 100})
    if show_api_error(st, result):
        return
    summary = result.data or {}
    drift = (
        (drift_result.data or {}).get("drift", summary.get("drift", {}))
        if drift_result.ok
        else summary.get("drift", {})
    )
    selected_alerts = summary.get("alerts", [])
    st.caption(
        f"Window: {summary.get('window_start', '—')} to {summary.get('window_end', '—')} (UTC)"
    )

    st.subheader("Input drift")
    inputs = drift.get("inputs", {})
    rows = []
    for name in INPUTS:
        metric = inputs.get(name, {})
        alert = next(
            (
                item
                for item in selected_alerts
                if item.get("metric_name") in {name, f"missingness:{name}"}
            ),
            {},
        )
        rows.append(
            {
                "Variable": name,
                "Metric": "PSI" if "psi" in metric else "JS divergence",
                "Observed": metric.get("psi", metric.get("js_divergence")),
                "Sample": metric.get("sample_count"),
                "Unseen rate": metric.get("unseen_rate"),
                "Status": alert.get("severity", drift_status(drift.get("input_status"))),
                "Reference / threshold": alert.get(
                    "threshold_reference", "TRAIN OOF fixed reference"
                ),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        plot_rows = [row for row in rows if row["Observed"] is not None]
        if plot_rows:
            st.bar_chart(
                pd.DataFrame(plot_rows).set_index("Variable"), y="Observed", color="#c98542"
            )
    else:
        st.info("Input drift metrics are unavailable.")
    st.caption(
        f"Input status: **{drift_status(drift.get('input_status'))}**. Drift values use "
        "the fixed monitoring reference and backend thresholds."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Score drift")
        scores = drift.get("scores", {})
        score_rows = [
            {"Score": name, "PSI": item.get("psi"), "Sample": item.get("sample_count")}
            for name, item in scores.items()
        ]
        if score_rows:
            st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)
            values = pd.DataFrame(score_rows).dropna(subset=["PSI"])
            if not values.empty:
                st.bar_chart(values.set_index("Score"), y="PSI", color="#3276a8")
        else:
            st.info("Score drift metrics are unavailable.")
        st.caption(f"Score status: **{drift_status(drift.get('score_status'))}**")
    with right:
        st.subheader("Decision drift")
        decisions = drift.get("decisions", {})
        st.metric(
            "Jensen–Shannon divergence",
            f"{decisions.get('js_divergence'):.4f}"
            if decisions.get("js_divergence") is not None
            else "N/A",
        )
        st.caption(f"Decision status: **{drift_status(drift.get('decision_status'))}**")
        frequencies = decisions.get("frequencies", {})
        if frequencies:
            st.bar_chart(pd.DataFrame({"Share": frequencies}), color="#6b8e70")

    st.subheader("Entity novelty")
    novelty = drift.get("entity_novelty_rate", {})
    if novelty:
        st.dataframe(
            pd.DataFrame(
                [{"Entity": key, "Novelty rate": value} for key, value in novelty.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Entity novelty metrics are unavailable.")

    st.subheader("System health metrics")
    system = summary.get("system", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requests", system.get("request_count", "N/A"))
    c2.metric("Transactions scored", system.get("transactions_scored", "N/A"))
    c3.metric(
        "Success rate",
        f"{system['success_rate']:.1%}" if system.get("success_rate") is not None else "N/A",
    )
    p95 = (system.get("latency_ms") or {}).get("p95")
    c4.metric("P95 latency", f"{p95:,.0f} ms" if p95 is not None else "N/A")
    st.caption(
        f"Database ready: {system.get('database_ready', 'Unavailable')} · "
        f"Runtime recovered: {system.get('state_recovered', 'Unavailable')}"
    )

    st.subheader("Alerts")
    alerts = (alerts_result.data or {}).get("alerts", []) if alerts_result.ok else []
    if alerts:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Severity": item.get("severity"),
                        "Type": item.get("alert_type"),
                        "Metric": item.get("metric_name"),
                        "Observed": item.get("observed_value"),
                        "Threshold / reference": item.get("threshold_reference"),
                        "Time": item.get("created_at"),
                    }
                    for item in sort_alerts(alerts)
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    elif alerts_result.ok:
        st.info("No persisted alerts are available.")
    else:
        st.caption("Alerts are unavailable from the API.")
