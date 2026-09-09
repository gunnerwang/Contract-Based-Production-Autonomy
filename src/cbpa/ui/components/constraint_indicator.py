"""Constraint margin-to-limit bars (green/yellow/red)."""

from __future__ import annotations

import streamlit as st

from cbpa.ui.theme import CONSTRAINT_OK, CONSTRAINT_VIOLATION, CONSTRAINT_WARNING


def constraint_bar(
    name: str,
    current_value: float,
    limit: float,
    unit: str = "",
) -> None:
    """Render a single constraint compliance bar."""
    from cbpa.ui.state import is_dev_mode

    if limit == 0:
        margin_pct = 0.0
    else:
        margin = limit - current_value
        margin_pct = (margin / limit) * 100

    if margin_pct < 0:
        color = CONSTRAINT_VIOLATION
        status = "VIOLATION"
    elif margin_pct < 10:
        color = CONSTRAINT_WARNING
        status = "WARNING"
    else:
        color = CONSTRAINT_OK
        status = "OK"

    # Progress as fraction of limit (capped at 1.5 for display)
    progress = min(current_value / limit, 1.5) if limit > 0 else 0

    if is_dev_mode():
        st.markdown(f"**{name}** ({current_value:.2f} / {limit:.1f} {unit}) — {status}")
    else:
        _suffix = f" {unit}" if unit else ""
        _plain_status = {
            "OK": f"Within safe range (limit: {limit:g}{_suffix})",
            "WARNING": f"Approaching limit of {limit:g}{_suffix}",
            "VIOLATION": f"OVER LIMIT of {limit:g}{_suffix} — action needed",
        }
        st.markdown(f"**{name}:** {_plain_status[status]}")

    st.progress(min(progress, 1.0))
    st.markdown(
        f'<div style="height:3px;background:{color};border-radius:2px;margin-top:-10px;"></div>',
        unsafe_allow_html=True,
    )


def constraint_panel(constraints: list[dict]) -> None:
    """Render a panel of constraint bars.

    Each dict: {"name": str, "current_value": float, "limit": float, "unit": str}
    """
    for c in constraints:
        constraint_bar(
            name=c["name"],
            current_value=c["current_value"],
            limit=c["limit"],
            unit=c.get("unit", ""),
        )
        st.markdown("")
