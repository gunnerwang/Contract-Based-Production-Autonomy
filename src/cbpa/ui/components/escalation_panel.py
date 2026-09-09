"""Interactive 3-option escalation chooser for Phase 4."""

from __future__ import annotations

import streamlit as st

from cbpa.models.escalation import EscalationQuery, ManagerDecision, RemediationOption


def _option_card(option: RemediationOption, selected: bool) -> None:
    """Render a single escalation option card."""
    border_color = "#2196F3" if selected else "#E0E0E0"
    bg = "#E3F2FD" if selected else "#FAFAFA"
    constraint_icon = "✅" if option.preserves_human_constraints else "⚠️"

    st.markdown(
        f'<div style="border:2px solid {border_color};border-radius:8px;'
        f'padding:12px;margin-bottom:8px;background:{bg};">'
        f"<strong>{option.label}</strong><br>"
        f"{option.description}<br>"
        f'<small style="color:#666;">'
        f"{constraint_icon} Human constraints: "
        f"{'Preserved' if option.preserves_human_constraints else 'Relaxed'}"
        f"</small>"
        f"</div>",
        unsafe_allow_html=True,
    )


def escalation_panel(
    query: EscalationQuery,
    key_prefix: str = "escalation",
) -> ManagerDecision | None:
    """Render the Phase 4 escalation UI with 3 option cards.

    Returns a ManagerDecision when the user clicks "Confirm Decision",
    otherwise returns None.
    """
    st.warning(f"**Escalation Required:** {query.conflict_summary}")
    st.markdown(f"**Violated constraints:** {', '.join(query.violated_constraints)}")
    st.markdown(f"**Demand increase:** {query.demand_increase_pct:.0f}%")

    st.markdown("---")
    st.subheader("Remediation Options")

    # Radio selector
    option_labels = [f"{o.label}: {o.description}" for o in query.options]
    selected_idx = st.radio(
        "Select an option",
        range(len(query.options)),
        format_func=lambda i: option_labels[i],
        key=f"{key_prefix}_radio",
    )

    # Display option cards
    cols = st.columns(len(query.options))
    for i, (col, opt) in enumerate(zip(cols, query.options)):
        with col:
            _option_card(opt, selected=i == selected_idx)

    # Rationale input
    rationale = st.text_area(
        "Rationale (optional)",
        value="Prioritizing human well-being over deadline",
        key=f"{key_prefix}_rationale",
    )

    if st.button("Confirm Decision", key=f"{key_prefix}_confirm", type="primary"):
        selected_option = query.options[selected_idx]  # type: ignore[index]
        return ManagerDecision(
            selected_option=selected_option.label,
            rationale=rationale,
        )

    return None
