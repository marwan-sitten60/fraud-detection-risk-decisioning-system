"""Service, recovery, database, and frozen bundle status."""

from __future__ import annotations

from dashboard.client import APIClient
from dashboard.components import show_api_error


def render(api=None):
    import streamlit as st

    api = api or APIClient()
    st.title("System health")
    health = api.get("/health")
    readiness = api.get("/ready")
    version = api.get("/version")
    columns = st.columns(3)
    columns[0].metric("Liveness", "Healthy" if health.ok else "Unavailable")
    ready_data = readiness.data or {} if readiness.ok else {}
    columns[1].metric("Service readiness", "Ready" if ready_data.get("ready") else "Not ready")
    columns[2].metric(
        "Database readiness", "Ready" if ready_data.get("database_ready") else "Not ready"
    )
    if not health.ok:
        show_api_error(st, health)
    if not readiness.ok:
        show_api_error(st, readiness)
    if not version.ok:
        show_api_error(st, version)
    if readiness.ok:
        st.subheader("Runtime recovery")
        st.metric("Recovered from PostgreSQL", "Yes" if ready_data.get("state_recovered") else "No")
        st.metric("Replayed transactions", ready_data.get("replayed_transaction_count", "N/A"))
    if version.ok:
        data = version.data or {}
        st.subheader("Frozen bundle")
        left, right = st.columns(2)
        left.write(f"**Bundle:** {data.get('bundle_version', 'N/A')}")
        left.write(f"**Ensemble:** {data.get('ensemble', 'N/A')}")
        left.write(f"**Features:** {data.get('feature_count', 'N/A')}")
        right.write(f"**Calibration:** {data.get('calibration_method', 'N/A')}")
        right.write(f"**Decision policy:** {data.get('primary_policy', 'N/A')}")
        right.write(f"**Rules:** {'Enabled' if data.get('rules_enabled') else 'Disabled'}")
        st.caption(
            "Model architecture: 80% LightGBM + 20% XGBoost · Platt calibration · "
            "467 features · Balanced decision policy · rules default disabled"
        )
