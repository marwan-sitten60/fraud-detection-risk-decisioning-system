"""Safe summary of CLI-generated traffic simulation runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dashboard.formatting import load_simulation_summaries

ROOT = Path(__file__).resolve().parents[2]


def render(api=None):
    import streamlit as st

    st.title("Simulation")
    st.caption(
        "Portfolio demonstration using simulated/development replay traffic. "
        "Not a live financial institution system."
    )
    runs, malformed = load_simulation_summaries(ROOT / "reports")
    if malformed:
        st.caption(f"Skipped {malformed} unreadable or invalid report file(s).")
    if not runs:
        st.info("No simulation summaries found. Run the CLI command below to create a run.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Run ID": row.get("simulation_run_id"),
                        "Scenario": str(row.get("scenario", "unknown")).upper(),
                        "State": "DRIFTED" if row.get("scenario") == "drifted" else "HEALTHY",
                        "Configured": row.get("configured_transaction_count"),
                        "Scored": row.get("successfully_scored_count"),
                        "Failures": row.get("failed_count"),
                        "Target RPS": row.get("target_rps"),
                        "Achieved RPS": row.get("achieved_rps"),
                        "Replay lag (sec)": row.get("replay_lag_seconds"),
                        "Latency p50 (ms)": row.get("request_latency_p50_ms"),
                        "Latency p95 (ms)": row.get("request_latency_p95_ms"),
                        "Labels delivered": row.get("labels_delivered"),
                        "Mature labels": row.get("mature_label_count"),
                        "Alerts": row.get("run_alert_count", row.get("active_alert_count")),
                        "Bundle": row.get("bundle_version"),
                    }
                    for row in runs
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Run status comes from the configured scenario. It is not a model quality assessment."
        )
    st.subheader("Run with the CLI")
    st.code(
        ".venv\\Scripts\\python.exe scripts\\run_traffic_simulation.py --help",
        language="powershell",
    )
    st.caption(
        "Configure the CLI’s simulation source and scenario in its existing options; "
        "the dashboard does not start simulator processes."
    )
