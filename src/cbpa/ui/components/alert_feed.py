"""Scrollable alert feed with severity icons."""

from __future__ import annotations

import streamlit as st

from cbpa.layer4_execution.monitor import MonitorAlert
from cbpa.ui.theme import SEVERITY_COLORS

_PLAIN_ALERT_TEMPLATES = {
    "fatigue": {
        "violation": "Worker fatigue exceeds safe limit — reduce robot speed or take a break.",
        "warning": "Worker fatigue approaching limit — consider slowing down.",
    },
    "noise": {
        "violation": "Noise exceeds 80 dB limit — check robot speeds.",
        "warning": "Noise level rising — consider reducing speed.",
    },
}


def _plain_alert_message(alert: MonitorAlert) -> str:
    """Convert a technical alert to plain language."""
    msg = alert.message.lower()
    for keyword, templates in _PLAIN_ALERT_TEMPLATES.items():
        if keyword in msg:
            return templates.get(alert.severity, alert.message)
    return alert.message


def alert_item(alert: MonitorAlert) -> None:
    """Render a single alert entry."""
    from cbpa.ui.state import is_dev_mode

    color = SEVERITY_COLORS.get(alert.severity, "#757575")
    icon = "🔴" if alert.severity == "violation" else "🟡"
    message = alert.message if is_dev_mode() else _plain_alert_message(alert)

    st.markdown(
        f'{icon} <span style="color:{color};font-weight:bold;">'
        f"[{alert.severity.upper()}]</span> {message}",
        unsafe_allow_html=True,
    )


def alert_feed(alerts: list[MonitorAlert], max_items: int = 20) -> None:
    """Render a scrollable list of alerts, newest first."""
    if not alerts:
        st.info("No alerts.")
        return

    display = list(reversed(alerts[-max_items:]))
    with st.container(height=300):
        for a in display:
            alert_item(a)
            st.markdown(
                '<hr style="margin:4px 0;border:none;border-top:1px solid #eee;">',
                unsafe_allow_html=True,
            )
