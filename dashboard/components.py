"""Small shared Streamlit presentation helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def select_window(st, key: str = "monitoring_window") -> dict:
    choice = st.selectbox("Window", ["Last hour", "Last 24 hours", "Last 7 days"], key=key)
    token = {"Last hour": "1h", "Last 24 hours": "24h", "Last 7 days": "7d"}[choice]
    return time_window_params(token)


def time_window_params(token: str, now: datetime | None = None) -> dict:
    duration = {"1h": timedelta(hours=1), "24h": timedelta(hours=24), "7d": timedelta(days=7)}[
        token
    ]
    end = now or datetime.now(UTC)
    end = end.astimezone(UTC)
    return {"window": token, "from": (end - duration).isoformat(), "to": end.isoformat()}


def show_api_error(st, result) -> bool:
    if result.ok:
        return False
    st.warning(f"Data unavailable: {result.error}. Check that FastAPI is running and ready.")
    return True


def number(value, digits: int = 0) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def pct(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def metric_or_unavailable(st, label: str, value, formatter=str):
    st.metric(label, formatter(value) if value is not None else "N/A")


def iso_bounds(start: datetime, end: datetime) -> dict:
    return {"from": start.astimezone(UTC).isoformat(), "to": end.astimezone(UTC).isoformat()}
