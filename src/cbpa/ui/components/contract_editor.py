"""OutcomeContract form widget for editing KPI targets and constraints."""

from __future__ import annotations

import streamlit as st

from cbpa.models.contract import (
    HardConstraint,
    KPIDirection,
    KPITarget,
    OutcomeContract,
    PriorityLevel,
)


def contract_editor(contract: OutcomeContract, key_prefix: str = "contract") -> OutcomeContract | None:
    """Render an editable form for an OutcomeContract.

    Returns the updated contract if "Apply" is clicked, else None.
    """
    st.subheader(f"Contract: {contract.name}")

    with st.form(f"{key_prefix}_form"):
        # KPI targets
        st.markdown("**KPI Targets**")
        kpi_targets = []
        for i, kpi in enumerate(contract.kpi_targets):
            cols = st.columns([2, 2, 1])
            with cols[0]:
                name = st.text_input("Name", value=kpi.name, key=f"{key_prefix}_kpi_name_{i}")
            with cols[1]:
                direction = st.selectbox(
                    "Direction",
                    [d.value for d in KPIDirection],
                    index=[d.value for d in KPIDirection].index(kpi.direction.value),
                    key=f"{key_prefix}_kpi_dir_{i}",
                )
            with cols[2]:
                threshold = st.number_input(
                    "Threshold",
                    value=kpi.threshold if kpi.threshold is not None else 0.0,
                    key=f"{key_prefix}_kpi_thresh_{i}",
                    format="%.4f",
                )
            kpi_targets.append(
                KPITarget(
                    name=name,
                    direction=KPIDirection(direction),
                    threshold=threshold if threshold != 0 else None,
                    unit=kpi.unit,
                )
            )

        st.markdown("---")

        # Hard constraints K
        st.markdown("**Hard Constraints (K)**")
        hard_constraints = []
        for i, hc in enumerate(contract.hard_constraints):
            cols = st.columns([2, 1, 1])
            with cols[0]:
                hc_name = st.text_input(
                    "Constraint", value=hc.name, key=f"{key_prefix}_hc_name_{i}"
                )
            with cols[1]:
                hc_op = st.selectbox(
                    "Op", ["<=", ">=", "==", "<", ">"],
                    index=["<=", ">=", "==", "<", ">"].index(hc.operator),
                    key=f"{key_prefix}_hc_op_{i}",
                )
            with cols[2]:
                hc_limit = st.number_input(
                    "Limit", value=hc.limit, key=f"{key_prefix}_hc_limit_{i}",
                    format="%.2f",
                )
            hard_constraints.append(
                HardConstraint(name=hc_name, operator=hc_op, limit=hc_limit, unit=hc.unit)
            )

        st.markdown("---")

        # Priority order
        st.markdown("**Priority Order (P)**")
        priority_str = st.text_input(
            "Priority (comma-separated)",
            value=", ".join(p.value for p in contract.priority_order),
            key=f"{key_prefix}_priority",
        )

        submitted = st.form_submit_button("Apply Contract Changes")

    if submitted:
        priority_order = []
        for p in priority_str.split(","):
            p = p.strip()
            try:
                priority_order.append(PriorityLevel(p))
            except ValueError:
                st.warning(f"Unknown priority level: {p}")

        updated = OutcomeContract(
            name=contract.name,
            kpi_targets=kpi_targets,
            hard_constraints=hard_constraints,
            priority_order=priority_order,
            context=contract.context,
            assumptions=contract.assumptions,
        )
        return updated

    return None
