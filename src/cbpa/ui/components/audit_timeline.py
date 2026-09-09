"""Vertical decision timeline for audit entries."""

from __future__ import annotations

import streamlit as st

from cbpa.models.escalation import AuditEntry
from cbpa.ui.theme import PHASE_COLORS


def audit_timeline(entries: list[AuditEntry], filter_phase: int | None = None) -> None:
    """Render a vertical timeline of audit entries with phase-colored dots."""
    if filter_phase is not None:
        entries = [e for e in entries if e.phase == filter_phase]

    if not entries:
        st.info("No audit entries to display.")
        return

    for entry in entries:
        color = PHASE_COLORS.get(entry.phase, "#757575")
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;margin-bottom:12px;">'
            f'<div style="width:12px;height:12px;border-radius:50%;background:{color};'
            f'margin-top:5px;margin-right:12px;flex-shrink:0;"></div>'
            f"<div>"
            f'<div style="font-size:0.85em;color:#888;">Phase {entry.phase} | {entry.layer} | {entry.timestamp[:19]}</div>'
            f"<div><strong>{entry.action}</strong></div>"
            f'<div style="font-size:0.9em;color:#555;">{entry.contract_state}</div>'
            f"</div></div>",
            unsafe_allow_html=True,
        )


def audit_table(entries: list[AuditEntry]) -> None:
    """Render audit entries as a table with expandable JSON details."""
    import json

    for i, entry in enumerate(entries):
        color = PHASE_COLORS.get(entry.phase, "#757575")
        cols = st.columns([0.5, 1, 1.5, 2, 2, 1])
        with cols[0]:
            st.markdown(
                f'<span style="color:{color};font-weight:bold;">P{entry.phase}</span>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.text(entry.layer)
        with cols[2]:
            st.text(entry.action)
        with cols[3]:
            st.text(entry.contract_state)
        with cols[4]:
            st.text(entry.timestamp[:19])
        with cols[5]:
            if entry.details:
                with st.popover("Details"):
                    st.json(entry.details)
