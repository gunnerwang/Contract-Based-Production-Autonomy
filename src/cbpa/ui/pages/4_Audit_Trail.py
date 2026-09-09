"""Page 4: Audit Trail — Timeline + table views."""

from __future__ import annotations

import streamlit as st

from cbpa.ui.components.audit_timeline import audit_table, audit_timeline
from cbpa.ui.state import get_experiment_service, get_factory_experiment_service, is_dev_mode, is_factory_mode
from cbpa.ui.theme import PHASE_LABELS


def render() -> None:
    """Main render function for the Audit Trail page."""
    if not is_dev_mode():
        st.info("The Audit Trail is available in **Developer Mode**. "
                "Enable it from the sidebar toggle.")
        return

    st.header("Audit Trail")

    # Get audit entries from the right experiment based on mode
    entries = []
    if is_factory_mode():
        fsvc = get_factory_experiment_service()
        result = st.session_state.get("factory_experiment_result")
        if fsvc._experiment is not None:
            entries = fsvc._experiment.audit.entries
    else:
        svc = get_experiment_service()
        result = st.session_state.get("experiment_result")
        if result is not None and hasattr(result, "audit"):
            entries = result.audit.entries
        elif svc._experiment is not None:
            entries = svc._experiment.audit.entries

    if not entries:
        st.info("No audit entries. Run an experiment first.")
        return

    st.markdown(f"**Total entries:** {len(entries)}")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        all_phases = sorted(set(e.phase for e in entries))
        phase_filter = st.selectbox(
            "Filter by Phase",
            [None] + all_phases,
            format_func=lambda x: "All Phases" if x is None else f"Round {x}: {PHASE_LABELS.get(x, '')}",
        )
    with col2:
        layer_options = sorted(set(e.layer for e in entries))
        layer_filter = st.selectbox(
            "Filter by Layer",
            [None] + layer_options,
            format_func=lambda x: "All Layers" if x is None else x,
        )

    filtered = entries
    if phase_filter is not None:
        filtered = [e for e in filtered if e.phase == phase_filter]
    if layer_filter is not None:
        filtered = [e for e in filtered if e.layer == layer_filter]

    # View toggle
    tab1, tab2, tab3 = st.tabs(["Timeline", "Table", "Escalation Provenance"])

    with tab1:
        audit_timeline(filtered)

    with tab2:
        audit_table(filtered)

    with tab3:
        st.subheader("Round 4: Escalation Decision Provenance")
        p4_entries = [e for e in entries if e.phase == 4]
        if not p4_entries:
            st.info("Escalation round has not been executed yet.")
        else:
            audit_timeline(p4_entries)
            st.markdown("---")
            st.markdown("**Decision Details**")
            for e in p4_entries:
                if e.action == "manager_decision":
                    st.json(e.details)
