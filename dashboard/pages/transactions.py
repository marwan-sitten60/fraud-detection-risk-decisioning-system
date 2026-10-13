"""Audit-safe prediction lookup and recent transaction table."""

from __future__ import annotations

import pandas as pd

from dashboard.client import APIClient
from dashboard.components import select_window, show_api_error
from dashboard.formatting import (
    format_decision,
    format_risk,
    risk_decision_note,
    safe_transaction_view,
)


def _display_row(row: dict) -> dict:
    row = safe_transaction_view(row)
    row["Calibrated fraud probability"] = format_risk(row.pop("calibrated_probability", None))
    row["Actual decision"] = format_decision(
        row.pop("decision", row.get("final_decision") or row.get("base_decision"))
    )
    row.pop("base_decision", None)
    row.pop("final_decision", None)
    row["Triggered rules"] = ", ".join(row.pop("triggered_rule_ids", []) or []) or "None"
    return row


def render(api=None):
    import streamlit as st

    api = api or APIClient()
    st.title("Transactions")
    st.caption("Prediction audit view. Raw transaction payloads and feature vectors are not shown.")
    transaction_id = st.text_input("Search TransactionID", placeholder="Enter an integer ID")
    if transaction_id.strip():
        try:
            parsed_id = int(transaction_id.strip())
            result = api.get(f"/predictions/{parsed_id}")
            if result.ok:
                record = result.data or {}
                st.metric(
                    "Calibrated fraud probability",
                    format_risk(record.get("calibrated_probability")),
                )
                st.info(risk_decision_note())
                final_action = record.get("final_decision") or record.get("base_decision")
                st.metric("Actual decision", format_decision(final_action))
                safe = _display_row(record)
                st.dataframe(pd.DataFrame([safe]), use_container_width=True, hide_index=True)
            elif result.status_code == 404:
                st.info("No stored prediction was found for that TransactionID.")
            else:
                show_api_error(st, result)
        except ValueError:
            st.warning("Enter a whole-number TransactionID.")
    st.divider()
    st.subheader("Recent predictions")
    window = select_window(st, "transactions_window")
    decision = st.selectbox("Decision filter", ["All", "APPROVE", "REVIEW", "BLOCK"])
    page_col, offset_col = st.columns(2)
    limit = page_col.selectbox("Rows per page", [25, 50, 100])
    offset = offset_col.number_input("Row offset", min_value=0, max_value=100_000, step=limit)
    params = {"limit": limit, "offset": offset, **window}
    if decision != "All":
        params["decision"] = decision
    result = api.get("/predictions", params)
    if show_api_error(st, result):
        return
    items = (result.data or {}).get("items", [])
    if not items:
        st.info("No predictions match this window and decision filter.")
    else:
        st.caption(f"Showing {len(items)} records starting at offset {offset}.")
        st.dataframe(
            pd.DataFrame([_display_row(item) for item in items]),
            use_container_width=True,
            hide_index=True,
        )
