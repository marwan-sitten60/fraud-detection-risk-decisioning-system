"""FraudGuard local operational dashboard."""

from __future__ import annotations

import streamlit as st

from dashboard.client import APIClient
from dashboard.pages import feedback, health, monitoring, overview, simulation, transactions

st.set_page_config(page_title="FraudGuard Operations", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; max-width: 1500px;}
      [data-testid="stMetric"] {
        background: #f5f7f9; border: 1px solid #e4e9ee;
        padding: .85rem; border-radius: .55rem;
      }
      .fraudguard-banner {
        padding: .7rem 1rem; background: #eef4f7;
        border-left: 4px solid #3276a8; border-radius: .25rem;
        margin-bottom: .8rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
api = APIClient()
st.markdown(
    '<div class="fraudguard-banner"><b>LOCAL / SIMULATED / DEVELOPMENT REPLAY</b> '
    "\u00b7 Portfolio demonstration; not a live financial institution system.</div>",
    unsafe_allow_html=True,
)
with st.sidebar:
    st.title("FraudGuard")
    st.caption("Risk operations console")
    if st.button("Refresh data", use_container_width=True):
        st.rerun()
    ready = api.get("/ready")
    version = api.get("/version") if ready.ok else None
    ready_data = ready.data or {} if ready.ok else {}
    st.caption(f"Service: {'Ready' if ready_data.get('ready') else 'Unavailable'}")
    st.caption(f"Database: {'Ready' if ready_data.get('database_ready') else 'Unavailable'}")
    bundle_name = (
        (version.data or {}).get("bundle_version", "Unavailable")
        if version and version.ok
        else "Unavailable"
    )
    st.caption(f"Bundle: {bundle_name}")
    st.divider()
    st.caption(
        "Authentication and authorization are required before exposing this dashboard "
        "to sensitive production traffic."
    )

pages = [
    st.Page(overview.render, title="Overview", icon="📊", default=True, url_path="overview"),
    st.Page(transactions.render, title="Transactions", icon="🔎", url_path="transactions"),
    st.Page(monitoring.render, title="Monitoring", icon="📈", url_path="monitoring"),
    st.Page(feedback.render, title="Feedback", icon="🏷️", url_path="feedback"),
    st.Page(simulation.render, title="Simulation", icon="🧪", url_path="simulation"),
    st.Page(health.render, title="System Health", icon="🩺", url_path="health"),
]
st.navigation(pages).run()
