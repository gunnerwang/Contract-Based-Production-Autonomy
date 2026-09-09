"""Horizontal 5-phase progress indicator."""

from __future__ import annotations

import streamlit as st

from cbpa.ui.theme import PHASE_COLORS, PHASE_LABELS


def phase_timeline(current_phase: int, completed_phases: list[int] | None = None) -> None:
    """Render a horizontal 5-phase progress indicator.

    Circles: gray=pending, blue/colored=current, green-check=completed, red=error.
    """
    completed = set(completed_phases or [])

    cols = st.columns(5)
    for i, col in enumerate(cols, start=1):
        with col:
            if i in completed:
                icon = "✅"
                bg = PHASE_COLORS.get(i, "#4CAF50")
                opacity = "1.0"
            elif i == current_phase:
                icon = "🔄"
                bg = PHASE_COLORS.get(i, "#2196F3")
                opacity = "1.0"
            else:
                icon = f"**{i}**"
                bg = "#E0E0E0"
                opacity = "0.5"

            label = PHASE_LABELS.get(i, f"Phase {i}")
            st.markdown(
                f'<div style="text-align:center;opacity:{opacity};">'
                f'<div style="width:40px;height:40px;border-radius:50%;'
                f"background:{bg};color:white;display:inline-flex;"
                f'align-items:center;justify-content:center;font-size:18px;">'
                f"{icon}</div>"
                f"<br><small>{label}</small></div>",
                unsafe_allow_html=True,
            )
