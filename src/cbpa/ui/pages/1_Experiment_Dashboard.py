"""Page 1: Experiment Dashboard — Config + Phase Execution + Contract Editor."""

from __future__ import annotations

import time

import streamlit as st

from cbpa.config.scenario import ScenarioConfig
from cbpa.models.contract import make_c1
from cbpa.models.escalation import ManagerDecision
from cbpa.service.config_service import ConfigService, PRESETS
from cbpa.runner.experiment import MANAGER_INTENT
from cbpa.service.experiment_service import ExperimentService, ExperimentState
from cbpa.ui.components.contract_editor import contract_editor
from cbpa.ui.components.escalation_panel import escalation_panel
from cbpa.ui.components.phase_timeline import phase_timeline
from cbpa.ui.components.schedule_card import schedule_card
from cbpa.ui.state import get_experiment_service, get_monitoring_service, is_dev_mode
import logging

_logger = logging.getLogger(__name__)
from cbpa.ui.theme import (
    PHASE_COLORS,
    OPTION_DISPLAY_NAMES,
    SCHEDULE_DISPLAY_NAMES,
    display_phase,
)


def _config_tab() -> None:
    """Tab 1: Configuration panel with sliders."""
    st.subheader("Scenario Configuration")

    config: ScenarioConfig = st.session_state.config

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Cell Parameters**")
        r1_cycle = st.slider("R1 Cycle Time (s)", 30.0, 120.0, config.cell.r1_cycle_time_s, 1.0)
        r2_cycle = st.slider("R2 Cycle Time (s)", 30.0, 120.0, config.cell.r2_cycle_time_s, 1.0)
        human_ins = st.slider("Human Insertion (s)", 30.0, 120.0, config.cell.human_insertion_time_s, 1.0)
        shift_h = st.slider("Shift Hours", 4.0, 12.0, config.cell.shift_hours, 0.5)
        demand = st.slider("Base Demand (u/h)", 20.0, 100.0, config.cell.demand_base_uph, 1.0)

        st.markdown("**Demand Spike**")
        spike_pct = st.slider("Spike %", 0.0, 50.0, config.demand_spike_pct, 1.0)

    with col_right:
        st.markdown("**Schedule Parameters**")

        st.caption("S1 (Baseline)")
        s1c1, s1c2, s1c3, s1c4 = st.columns(4)
        s1_r1 = s1c1.number_input("R1 Spd", value=config.schedules.s1_r1_speed, format="%.2f", key="s1r1")
        s1_r2 = s1c2.number_input("R2 Spd", value=config.schedules.s1_r2_speed, format="%.2f", key="s1r2")
        s1_hr = s1c3.number_input("H Rate", value=config.schedules.s1_human_rate, format="%.2f", key="s1hr")
        s1_buf = s1c4.number_input("Buffer", value=config.schedules.s1_buffer_s, format="%.1f", key="s1buf")

        st.caption("S2 (Aggressive)")
        s2c1, s2c2, s2c3, s2c4 = st.columns(4)
        s2_r1 = s2c1.number_input("R1 Spd", value=config.schedules.s2_r1_speed, format="%.2f", key="s2r1")
        s2_r2 = s2c2.number_input("R2 Spd", value=config.schedules.s2_r2_speed, format="%.2f", key="s2r2")
        s2_hr = s2c3.number_input("H Rate", value=config.schedules.s2_human_rate, format="%.2f", key="s2hr")
        s2_buf = s2c4.number_input("Buffer", value=config.schedules.s2_buffer_s, format="%.1f", key="s2buf")

        st.caption("S3 (Balanced)")
        s3c1, s3c2, s3c3, s3c4 = st.columns(4)
        s3_r1 = s3c1.number_input("R1 Spd", value=config.schedules.s3_r1_speed, format="%.2f", key="s3r1")
        s3_r2 = s3c2.number_input("R2 Spd", value=config.schedules.s3_r2_speed, format="%.2f", key="s3r2")
        s3_hr = s3c3.number_input("H Rate", value=config.schedules.s3_human_rate, format="%.2f", key="s3hr")
        s3_buf = s3c4.number_input("Buffer", value=config.schedules.s3_buffer_s, format="%.1f", key="s3buf")

        st.markdown("**V/R Weights**")
        st.caption("Value: [throughput, quality, flexibility]")
        vw = st.text_input(
            "Value weights",
            value=", ".join(f"{w:.2f}" for w in config.vr_value_weights),
            key="vw",
        )
        st.caption("Resource: [energy, downtime, labor, cost]")
        rw = st.text_input(
            "Resource weights",
            value=", ".join(f"{w:.2f}" for w in config.vr_resource_weights),
            key="rw",
        )

    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("Apply Configuration", type="primary"):
            try:
                vr_value = [float(x.strip()) for x in vw.split(",")]
                vr_resource = [float(x.strip()) for x in rw.split(",")]
            except ValueError:
                st.error("Invalid weight format.")
                return

            errors = ConfigService.validate_weights(vr_value, "Value weights")
            errors += ConfigService.validate_weights(vr_resource, "Resource weights")
            if errors:
                for e in errors:
                    st.error(e)
                return

            new_config = ConfigService.from_dict({
                "cell.r1_cycle_time_s": r1_cycle,
                "cell.r2_cycle_time_s": r2_cycle,
                "cell.human_insertion_time_s": human_ins,
                "cell.shift_hours": shift_h,
                "cell.demand_base_uph": demand,
                "schedules.s1_r1_speed": s1_r1,
                "schedules.s1_r2_speed": s1_r2,
                "schedules.s1_human_rate": s1_hr,
                "schedules.s1_buffer_s": s1_buf,
                "schedules.s2_r1_speed": s2_r1,
                "schedules.s2_r2_speed": s2_r2,
                "schedules.s2_human_rate": s2_hr,
                "schedules.s2_buffer_s": s2_buf,
                "schedules.s3_r1_speed": s3_r1,
                "schedules.s3_r2_speed": s3_r2,
                "schedules.s3_human_rate": s3_hr,
                "schedules.s3_buffer_s": s3_buf,
                "demand_spike_pct": spike_pct,
                "vr_value_weights": vr_value,
                "vr_resource_weights": vr_resource,
            })
            st.session_state.config = new_config
            svc = get_experiment_service()
            svc.reset(config=new_config)
            st.success("Configuration applied.")
            st.rerun()

    with bc2:
        if st.button("Reset to Defaults"):
            st.session_state.config = ScenarioConfig()
            svc = get_experiment_service()
            svc.reset(config=ScenarioConfig())
            st.success("Reset to defaults.")
            st.rerun()


def _contract_diff_block(phases: list) -> None:
    """Compare C1 vs C2 context, show unchanged constraints + priority."""
    c1 = c2 = None
    for pr in phases:
        if pr.contract is not None:
            if pr.contract.name == "C1" and c1 is None:
                c1 = pr.contract
            if pr.contract.name == "C2":
                c2 = pr.contract
    if c1 is None or c2 is None:
        return

    st.subheader("Contract Diff: C1 → C2")

    # Context changes
    changed = {}
    added = {}
    for k, v in c2.context.items():
        if k not in c1.context:
            added[k] = v
        elif c1.context[k] != v:
            changed[k] = (c1.context[k], v)

    if changed or added:
        st.markdown("**Context changes:**")
        for k, (old, new) in changed.items():
            st.markdown(f"- `{k}`: {old} → {new}")
        for k, v in added.items():
            st.markdown(f"- `{k}`: *(new)* {v}")
    else:
        st.markdown("**Context:** no changes")

    # Unchanged constraints
    st.markdown("**Hard constraints (unchanged):**")
    for hc in c1.hard_constraints:
        st.markdown(f"- {hc.name} {hc.operator} {hc.limit} {hc.unit}")

    # Priority order
    st.markdown(f"**Priority order (unchanged):** {' > '.join(p.value for p in c1.priority_order)}")


def _execution_tab() -> None:
    """Tab 2: Phase execution with progress indicator."""
    svc = get_experiment_service()
    status = svc.status
    completed_nums = [pr.phase for pr in status.completed_phases]

    # Manager intent
    with st.expander("Manager Intent", expanded=False):
        st.markdown(f"*{MANAGER_INTENT}*")

    # Progress indicator
    phase_timeline(
        current_phase=status.current_phase,
        completed_phases=completed_nums,
    )

    st.markdown("---")

    # Buttons
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        if st.button("Run All Phases", type="primary",
                      disabled=status.state not in (ExperimentState.IDLE, ExperimentState.COMPLETED)):
            svc.reset(config=st.session_state.config)
            result = svc.run_all()
            st.session_state.experiment_result = result
            st.rerun()

    with bc2:
        next_phase = max(completed_nums) + 1 if completed_nums else 1
        can_step = (
            status.state in (ExperimentState.IDLE, ExperimentState.RUNNING, ExperimentState.COMPLETED)
            and next_phase <= 5
        )
        if st.button(f"Step Phase {next_phase}", disabled=not can_step):
            if next_phase == 1 and status.state in (ExperimentState.IDLE, ExperimentState.COMPLETED):
                svc.reset(config=st.session_state.config)
            svc.run_phase(next_phase)
            if svc.status.state == ExperimentState.COMPLETED:
                st.session_state.experiment_result = svc.status.result
            st.rerun()

    with bc3:
        if st.button("Reset Experiment"):
            svc.reset(config=st.session_state.config)
            st.session_state.experiment_result = None
            st.rerun()

    # Escalation handshake
    if status.state == ExperimentState.AWAITING_ESCALATION:
        st.markdown("---")
        query = svc.get_escalation_query()
        if query is not None:
            decision = escalation_panel(query)
            if decision is not None:
                svc.resolve_escalation(decision)
                st.rerun()

    # Phase result cards
    if status.completed_phases:
        st.markdown("---")
        st.subheader("Phase Results")
        for pr in status.completed_phases:
            color = PHASE_COLORS.get(pr.phase, "#757575")
            with st.expander(f"Phase {pr.phase}: {pr.description}", expanded=pr.phase == len(status.completed_phases)):
                if pr.contract is not None:
                    prio = " > ".join(p.value for p in pr.contract.priority_order)
                    st.caption(f"Contract: **{pr.contract.name}** | Priority: {prio}")
                if pr.schedule:
                    schedule_card(pr.schedule, metrics=pr.metrics, vr_score=pr.vr_score)
                if pr.feasibility:
                    feas_color = "green" if pr.feasibility.is_feasible else "red"
                    st.markdown(
                        f"**Feasibility:** :{feas_color}[{'FEASIBLE' if pr.feasibility.is_feasible else 'REJECTED'}]"
                    )
                    if pr.feasibility.violations:
                        st.markdown(f"Violations: {', '.join(pr.feasibility.violations)}")
                if pr.decision:
                    st.markdown(f"**Decision:** {pr.decision.selected_option}")
                    st.markdown(f"*{pr.decision.rationale}*")

        # C1 → C2 diff block
        _contract_diff_block(status.completed_phases)


def _contract_tab() -> None:
    """Tab 3: Contract editor."""
    svc = get_experiment_service()
    # Use the contract from the latest phase, or default C1
    active_contract = None
    for pr in reversed(svc.status.completed_phases):
        if pr.contract is not None:
            active_contract = pr.contract
            break

    if active_contract is None:
        active_contract = make_c1()

    updated = contract_editor(active_contract)
    if updated is not None:
        st.success(f"Contract {updated.name} updated.")
        st.json(updated.model_dump())


# ── Operator-mode helpers ─────────────────────────────────────────────

def _fatigue_status(value: float) -> str:
    if value <= 0.30:
        return "OK"
    elif value <= 0.40:
        return "Caution"
    return "HIGH"


def _noise_status(value: float) -> str:
    if value <= 75.0:
        return "OK"
    elif value <= 80.0:
        return "Caution"
    return "TOO LOUD"


def _violation_to_plain(v: str) -> str:
    _map = {
        "fatigue_index": "Worker fatigue exceeds safe limit",
        "noise_db": "Noise level exceeds 80 dB limit",
    }
    for key, plain in _map.items():
        if key in v.lower():
            return plain
    return v


# ── Operator three-mode helpers ──────────────────────────────────────

_TRADEOFF_TABLE = {
    "Worker Well-being": {"throughput": "~42 u/h", "comfort": "High", "risk": "Low"},
    "Safety": {"throughput": "~43 u/h", "comfort": "High", "risk": "Low"},
    "Quality": {"throughput": "~45 u/h", "comfort": "Medium", "risk": "Low"},
    "Throughput": {"throughput": "~53 u/h", "comfort": "Low", "risk": "High"},
    "Cost": {"throughput": "~48 u/h", "comfort": "Medium", "risk": "Medium"},
    "Flexibility": {"throughput": "~46 u/h", "comfort": "Medium", "risk": "Low"},
}

_PRODUCT_MIX_OPTIONS = ["V_A", "V_B", "V_C"]

_ALL_PRIORITIES = [
    "Worker Well-being", "Safety", "Quality",
    "Throughput", "Cost", "Flexibility",
]

_PRIORITY_TO_ENUM = {
    "Worker Well-being": "HumanWellbeing",
    "Safety": "Safety",
    "Quality": "Quality",
    "Throughput": "Throughput",
    "Cost": "Cost",
    "Flexibility": "Flexibility",
}


def _tradeoff_preview(priority_order: list[str]) -> None:
    """Static estimate lookup based on top priority in the ordering."""
    if not priority_order:
        return
    top = priority_order[0]
    est = _TRADEOFF_TABLE.get(top)
    if est is None:
        return
    st.markdown(f"**Estimated trade-offs** (top priority: {top}):")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Throughput", est["throughput"])
    with c2:
        st.metric("Worker Comfort", est["comfort"])
    with c3:
        st.metric("Safety Risk", est["risk"])


def _apply_legislator_contract(choices: dict) -> ScenarioConfig:
    """Build a ScenarioConfig from legislator choices, optionally starting from a preset."""
    preset = choices.get("preset", "Standard Shift")
    base_config = ConfigService.from_preset(preset)

    overrides: dict = {}
    shift_hours = choices.get("shift_hours", 8)
    overrides["cell.shift_hours"] = float(shift_hours)

    demand_level = choices.get("demand_level", "Normal")
    if demand_level == "High (+20%)":
        overrides["demand_spike_pct"] = 20.0
    elif demand_level == "Surge (+30%)":
        overrides["demand_spike_pct"] = 30.0
    else:
        overrides["demand_spike_pct"] = 0.0

    # Map top priority to robot speed settings
    priority_order = choices.get("priority_order", ["Worker Well-being"])
    top_priority = priority_order[0] if priority_order else "Worker Well-being"
    if top_priority == "Throughput":
        overrides["schedules.s1_r1_speed"] = 0.80
        overrides["schedules.s1_r2_speed"] = 0.80
    elif top_priority in ("Worker Well-being", "Safety"):
        overrides["schedules.s1_r1_speed"] = 0.55
        overrides["schedules.s1_r2_speed"] = 0.55

    if overrides:
        config_dict = {}
        # Start from preset config, then overlay
        preset_cfg = PRESETS[preset].get("config", {})
        config_dict.update(preset_cfg)
        config_dict.update(overrides)
        return ConfigService.from_dict(config_dict) if config_dict else base_config
    return base_config


def _push_to_monitoring(result) -> None:
    """Push experiment results to monitoring service + integration bridges."""
    from cbpa.models.contract import make_c1
    mon = get_monitoring_service()
    mon.clear()
    contract = make_c1()
    for pr in result.phases:
        if pr.contract is not None:
            contract = pr.contract
    mon.set_contract(contract)
    for pr in result.phases:
        if pr.metrics is not None:
            mon.push_snapshot(pr.metrics, phase=pr.phase, layer="L4")
            # Push each phase's KPIs to BaSyx + OPC-UA
            _push_kpis_to_bridges(pr.metrics)


# ── Integration bridge helpers ──────────────────────────────────────


def _sync_contract_to_bridges(choices: dict) -> dict:
    """Push approved contract C^out to BaSyx AAS and OPC-UA. Returns status dict."""
    contract_data = {
        "name": "C1",
        "kpi_targets": [
            {"name": "DefectRate", "operator": "<=", "limit": choices.get("kpi_defect_rate_max", 0.01)},
            {"name": "DeadlineGap", "operator": "<=", "limit": choices.get("kpi_deadline_gap_max", 15.0)},
        ],
        "hard_constraints": [
            {"name": "FatigueIndex", "operator": "<=", "limit": choices.get("fatigue_limit", 0.4)},
            {"name": "Noise", "operator": "<=", "limit": choices.get("noise_limit", 80.0), "unit": "dB"},
            {"name": "CyberRiskLevel", "operator": "<=", "limit": choices.get("cyber_risk_limit", 2.0)},
        ],
        "priority_order": choices.get("priority_order_formal", []),
        "context": {
            "shift_hours": choices.get("shift_hours", 8),
            "demand_level": choices.get("demand_level", "Normal"),
            "product_mix": choices.get("product_mix", []),
        },
        "assumptions": choices.get("assumptions", {}),
    }

    status = {"basyx": None, "opcua": None}

    # BaSyx AAS sync
    basyx = st.session_state.get("basyx_bridge")
    if basyx is not None:
        result = basyx.sync_contract(contract_data)
        status["basyx"] = result

    # OPC-UA constraint limits
    opcua = st.session_state.get("opcua_bridge")
    if opcua is not None:
        opcua.update_constraints({
            "fatigue_limit": choices.get("fatigue_limit", 0.4),
            "noise_limit": choices.get("noise_limit", 80.0),
            "cyber_risk_limit": choices.get("cyber_risk_limit", 2.0),
        })
        opcua.update_contract_status("C1", "approved")
        status["opcua"] = {"updated": True, "stub": not opcua.is_connected}

    return status


def _verify_constraints_isaac(choices: dict) -> dict | None:
    """Pre-verify hard constraints K against Isaac Sim physics. Returns result dict."""
    isaac = st.session_state.get("isaac_bridge")
    if isaac is None:
        return None

    constraints = [
        {"name": "FatigueIndex", "operator": "<=", "limit": choices.get("fatigue_limit", 0.4)},
        {"name": "Noise", "operator": "<=", "limit": choices.get("noise_limit", 80.0)},
        {"name": "CyberRiskLevel", "operator": "<=", "limit": choices.get("cyber_risk_limit", 2.0)},
    ]
    return isaac.verify_constraints(constraints)


def _show_integration_status(sync_result: dict, verify_result: dict | None) -> None:
    """Display integration sync results after contract approval."""
    items = []

    # BaSyx status
    basyx_r = sync_result.get("basyx")
    if basyx_r is not None:
        if basyx_r.get("stub"):
            items.append("BaSyx AAS: synced (stub)")
        else:
            n = len(basyx_r.get("submodels_synced", []))
            items.append(f"BaSyx AAS: {n} submodels synced")

    # OPC-UA status
    opcua_r = sync_result.get("opcua")
    if opcua_r is not None:
        if opcua_r.get("stub"):
            items.append("OPC-UA: updated (stub)")
        else:
            items.append("OPC-UA: constraint limits published")

    # Isaac Sim status
    if verify_result is not None:
        vr = verify_result.get("verification_results", [])
        all_pass = verify_result.get("all_passed", True)
        if not all_pass:
            failed = [r["constraint"] for r in vr if not r["passed"]]
            st.warning(f"Isaac Sim: constraints {failed} may be violated.")
        else:
            margins = ", ".join(f"{r['constraint']}={r['margin_pct']}%" for r in vr)
            mode = "stub" if verify_result.get("stub") else "physics"
            items.append(f"Isaac Sim ({mode}): all K passed ({margins})")

    if items:
        st.caption("Integration: " + " | ".join(items))


def _push_kpis_to_bridges(metrics) -> None:
    """Push live KPI readings to BaSyx and OPC-UA after each phase."""
    kpi_data = {
        "throughput_uph": metrics.throughput_uph,
        "defect_rate": metrics.defect_rate,
        "noise_db": metrics.noise_db,
        "fatigue_index": metrics.fatigue_index,
        "energy_kwh": metrics.energy_kwh,
    }

    # BaSyx live KPI submodel
    basyx = st.session_state.get("basyx_bridge")
    if basyx is not None:
        basyx.update_kpi_submodel(kpi_data)

    # OPC-UA KPI nodes
    opcua = st.session_state.get("opcua_bridge")
    if opcua is not None:
        opcua.update_kpi({
            **kpi_data,
            "deadline_gap_pct": metrics.deadline_gap_pct,
        })


# ── Legislator tab (Mode 1) ─────────────────────────────────────────

def _legislator_tab() -> None:
    """Mode 1: Human as Legislator — guided contract elicitation."""
    st.subheader("Define Your Shift Contract")
    st.caption(
        "Set goals, safety limits, and shift context. "
        "The system will not execute until you approve. "
        "Contract structure: C = <KPI, K, P, X, A>"
    )

    # --- Priority ordering (maps to P) ---
    st.markdown("#### Priority Ordering  `P`")
    st.caption(
        "Order indicates priority — first item is most important. "
        "Remove and re-add items to reorder."
    )
    priority_order = st.multiselect(
        "Priority order (highest first)",
        _ALL_PRIORITIES,
        default=["Worker Well-being", "Safety", "Quality", "Throughput"],
        key="leg_priority_order",
        help="Defines the full priority ordering P in the outcome contract.",
    )
    if len(priority_order) < 2:
        st.warning("Select at least 2 priorities to define a meaningful ordering.")

    st.markdown("---")

    # --- KPI targets (maps to KPI) ---
    st.markdown("#### KPI Targets  `KPI`")
    st.caption("Set measurable performance targets for this shift.")
    kpi1, kpi2 = st.columns(2)
    with kpi1:
        defect_rate_max = st.slider(
            "Max Defect Rate",
            min_value=0.005, max_value=0.020, value=0.010, step=0.001,
            format="%.3f",
            key="leg_defect_rate",
            help="Upper bound on acceptable defect rate.",
        )
    with kpi2:
        deadline_gap_max = st.slider(
            "Max Acceptable Deadline Gap (%)",
            min_value=0.0, max_value=30.0, value=15.0, step=1.0,
            key="leg_deadline_gap",
            help="Maximum acceptable shortfall vs. demand target.",
        )

    st.markdown("---")

    # --- Hard constraints (maps to K) ---
    st.markdown("#### Hard Constraints  `K`")
    st.caption("Non-negotiable safety limits the system cannot override.")
    sl1, sl2, sl3 = st.columns(3)
    with sl1:
        fatigue_limit = st.slider(
            "Max Worker Fatigue",
            min_value=0.20, max_value=0.50, value=0.40, step=0.01,
            key="leg_fatigue",
            help="Hard constraint: fatigue index must stay below this.",
        )
    with sl2:
        noise_limit = st.slider(
            "Max Noise Level (dB)",
            min_value=70.0, max_value=85.0, value=80.0, step=0.5,
            key="leg_noise",
            help="Hard constraint: noise must stay below this.",
        )
    with sl3:
        cyber_risk_limit = st.slider(
            "Max Cyber Risk Level",
            min_value=1.0, max_value=4.0, value=2.0, step=0.5,
            key="leg_cyber",
            help="Hard constraint: cyber risk level (1=Low, 2=Medium, 3=High, 4=Critical).",
        )

    st.markdown("---")

    # --- Shift context (maps to X) ---
    st.markdown("#### Shift Context  `X`")
    st.caption("Operational context for this shift.")
    cx1, cx2, cx3 = st.columns(3)
    with cx1:
        shift_hours = st.select_slider(
            "Shift Duration (hours)",
            options=[6, 8, 10, 12],
            value=8,
            key="leg_shift_hours",
        )
    with cx2:
        demand_level = st.selectbox(
            "Demand Level",
            ["Normal", "High (+20%)", "Surge (+30%)"],
            key="leg_demand",
        )
    with cx3:
        product_mix = st.multiselect(
            "Product Mix",
            _PRODUCT_MIX_OPTIONS,
            default=["V_A", "V_B"],
            key="leg_mix",
        )

    st.markdown("---")

    # --- Assumptions (maps to A) ---
    st.markdown("#### Assumptions  `A`")
    st.caption("Environmental assumptions. If violated during execution, the system will escalate.")
    a1, a2, a3 = st.columns(3)
    with a1:
        demand_spike_risk = st.selectbox(
            "Demand Spike Risk",
            ["Low", "Medium", "High"],
            index=2,
            key="leg_spike_risk",
        )
    with a2:
        robot_max_speed = st.slider(
            "Robot Max Speed (m/s)",
            min_value=0.5, max_value=3.0, value=1.5, step=0.1,
            key="leg_robot_speed",
        )
    with a3:
        network_latency = st.slider(
            "Network Latency (ms)",
            min_value=5, max_value=100, value=20, step=5,
            key="leg_latency",
        )

    st.markdown("---")

    # --- Preset as starting point (collapsed) ---
    with st.expander("Start from a preset configuration", expanded=False):
        preset_names = ConfigService.preset_names()
        selected_preset = st.radio(
            "Configuration Preset",
            preset_names,
            key="leg_preset",
        )
        preset_info = PRESETS[selected_preset]
        st.info(preset_info["summary"])
        preview_cols = st.columns(len(preset_info["metrics_preview"]))
        for col, (k, v) in zip(preview_cols, preset_info["metrics_preview"].items()):
            with col:
                st.metric(k, v)

    st.markdown("---")

    # --- Trade-off preview ---
    _tradeoff_preview(priority_order)

    st.markdown("---")

    # --- Approve contract ---
    if st.button("Approve Contract & Begin Shift", type="primary"):
        choices = {
            "priority_order": priority_order,
            "priority_order_formal": [_PRIORITY_TO_ENUM[p] for p in priority_order],
            "kpi_defect_rate_max": defect_rate_max,
            "kpi_deadline_gap_max": deadline_gap_max,
            "fatigue_limit": fatigue_limit,
            "noise_limit": noise_limit,
            "cyber_risk_limit": cyber_risk_limit,
            "shift_hours": shift_hours,
            "demand_level": demand_level,
            "product_mix": product_mix,
            "assumptions": {
                "demand_spike_risk": demand_spike_risk,
                "r1_max_speed_mps": robot_max_speed,
                "network_latency_ms": network_latency,
            },
            "preset": st.session_state.get("leg_preset", "Standard Shift"),
        }
        new_config = _apply_legislator_contract(choices)
        st.session_state.config = new_config
        svc = get_experiment_service()
        svc.reset(config=new_config)
        st.session_state.legislator_choices = choices
        st.session_state.shift_contract_signed = True
        st.session_state.operator_feedback = []

        # ── Sync contract to BaSyx AAS ────────────────────────
        sync_result = _sync_contract_to_bridges(choices)

        # ── Verify constraints via Isaac Sim ──────────────────
        verify_result = _verify_constraints_isaac(choices)

        st.success("Contract approved. Proceed to the **Run & Monitor** tab.")

        # Show integration sync status
        _show_integration_status(sync_result, verify_result)
        st.rerun()

    # Show current contract status
    if st.session_state.get("shift_contract_signed"):
        st.info("Contract is currently **approved**. You can re-approve to update it.")


# ── Partner tab (Mode 2) ────────────────────────────────────────────

def _partner_tab() -> None:
    """Mode 2: Human as Partner — execution with real-time feedback."""
    st.subheader("Run & Monitor")

    # Gate check
    if not st.session_state.get("shift_contract_signed"):
        st.warning("No contract defined yet. Please go to the **Define Contract** tab first.")
        return

    svc = get_experiment_service()
    status = svc.status
    completed_nums = [pr.phase for pr in status.completed_phases]

    # Phase timeline (always visible)
    phase_timeline(
        current_phase=status.current_phase,
        completed_phases=completed_nums,
    )

    st.markdown("---")

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Shift", type="primary",
                      disabled=status.state not in (ExperimentState.IDLE, ExperimentState.COMPLETED)):
            svc.reset(config=st.session_state.config)
            # Deploy schedule to Isaac Sim before execution
            isaac = st.session_state.get("isaac_bridge")
            if isaac is not None:
                cfg = st.session_state.config
                isaac.deploy_schedule({
                    "schedule_name": "S1",
                    "r1_speed_fraction": cfg.schedules.s1_r1_speed,
                    "r2_speed_fraction": cfg.schedules.s1_r2_speed,
                })
            result = svc.run_all()
            st.session_state.experiment_result = result
            _push_to_monitoring(result)
            # Store how many KPI updates were pushed
            n_phases = sum(1 for pr in result.phases if pr.metrics is not None)
            st.session_state["_bridge_kpi_pushes"] = n_phases
            st.rerun()
    with col2:
        if st.button("Reset Shift"):
            svc.reset(config=st.session_state.config)
            st.session_state.experiment_result = None
            get_monitoring_service().clear()
            st.rerun()

    # Escalation inline
    if status.state == ExperimentState.AWAITING_ESCALATION:
        st.markdown("---")
        query = svc.get_escalation_query()
        if query is not None:
            _operator_escalation_inline(query, svc)

    # Live KPIs (after experiment runs)
    if status.completed_phases:
        st.markdown("---")
        _partner_live_kpis(status)

        # Integration sync indicator
        n_pushes = st.session_state.get("_bridge_kpi_pushes", 0)
        if n_pushes > 0:
            parts = []
            basyx = st.session_state.get("basyx_bridge")
            if basyx is not None and basyx.is_connected:
                parts.append(f"BaSyx AAS: {n_pushes} KPI updates")
            opcua = st.session_state.get("opcua_bridge")
            if opcua is not None and opcua.is_connected:
                parts.append(f"OPC-UA: {n_pushes} node updates")
            isaac = st.session_state.get("isaac_bridge")
            if isaac is not None:
                parts.append(f"Isaac Sim: schedule deployed ({isaac.mode})")
            if parts:
                st.caption("Integration: " + " | ".join(parts))

        st.markdown("---")
        _partner_feedback_controls()

        st.markdown("---")
        _partner_system_status(status)


def _partner_live_kpis(status) -> None:
    """Show 5 key metrics with contract-referenced limits."""
    st.subheader("Live KPIs")
    choices = st.session_state.get("legislator_choices", {})
    fatigue_limit = choices.get("fatigue_limit", 0.40)
    noise_limit = choices.get("noise_limit", 80.0)
    defect_limit = choices.get("kpi_defect_rate_max", 0.01)
    deadline_limit = choices.get("kpi_deadline_gap_max", 15.0)

    # Use the latest phase metrics
    latest_pr = status.completed_phases[-1]
    m = latest_pr.metrics
    if m is None:
        st.info("No metrics available yet.")
        return

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        st.metric("Production Rate", f"{m.throughput_uph:.1f} u/h")
    with mc2:
        delta = fatigue_limit - m.fatigue_index
        st.metric("Worker Fatigue `K`", f"{m.fatigue_index:.2f}", delta=f"{delta:+.2f} margin")
    with mc3:
        delta_n = noise_limit - m.noise_db
        st.metric("Noise Level `K`", f"{m.noise_db:.1f} dB", delta=f"{delta_n:+.1f} dB margin")
    with mc4:
        delta_d = defect_limit - m.defect_rate
        st.metric("Defect Rate `KPI`", f"{m.defect_rate:.4f}", delta=f"{delta_d:+.4f} margin")
    with mc5:
        delta_g = deadline_limit - m.deadline_gap_pct
        st.metric("Deadline Gap `KPI`", f"{m.deadline_gap_pct:.1f} %", delta=f"{delta_g:+.1f}% margin")

    # Progress bars for hard constraints K
    pb1, pb2 = st.columns(2)
    with pb1:
        fatigue_pct = min(m.fatigue_index / fatigue_limit, 1.0)
        st.markdown(f"**Fatigue `K`:** {m.fatigue_index:.2f} / {fatigue_limit:.2f}")
        st.progress(fatigue_pct)
    with pb2:
        noise_pct = min(m.noise_db / noise_limit, 1.0)
        st.markdown(f"**Noise `K`:** {m.noise_db:.1f} / {noise_limit:.1f} dB")
        st.progress(noise_pct)


def _partner_feedback_controls() -> None:
    """Three actionable feedback buttons for the operator."""
    st.subheader("Your Feedback")
    st.caption("Log your observations. These will be reviewed in the Shift Review tab.")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        if st.button("Request Slower Pace", key="fb_slower"):
            st.session_state.operator_feedback.append(
                {"action": "Request Slower Pace", "time": time.time()}
            )
            st.success("Logged: Request Slower Pace")
    with fc2:
        if st.button("Flag Discomfort", key="fb_discomfort"):
            st.session_state.operator_feedback.append(
                {"action": "Flag Discomfort", "time": time.time()}
            )
            st.success("Logged: Flag Discomfort")
    with fc3:
        if st.button("Request Break", key="fb_break"):
            st.session_state.operator_feedback.append(
                {"action": "Request Break", "time": time.time()}
            )
            st.success("Logged: Request Break")

    # Show feedback count
    fb_count = len(st.session_state.get("operator_feedback", []))
    if fb_count > 0:
        st.caption(f"{fb_count} feedback event(s) logged this shift.")


def _partner_system_status(status) -> None:
    """Show what happened at each phase."""
    st.subheader("System Status")

    for pr in status.completed_phases:
        phase_label = display_phase(pr.phase, dev_mode=False)
        with st.expander(f"Step {pr.phase}: {phase_label}", expanded=pr.phase == len(status.completed_phases)):
            if pr.schedule:
                sched_name = SCHEDULE_DISPLAY_NAMES.get(pr.schedule.name, pr.schedule.name)
                st.markdown(f"**Schedule:** {sched_name}")
            if pr.metrics is not None:
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("Production Rate", f"{pr.metrics.throughput_uph:.1f} u/h")
                with mc2:
                    st.metric("Worker Fatigue", _fatigue_status(pr.metrics.fatigue_index))
                with mc3:
                    st.metric("Noise", _noise_status(pr.metrics.noise_db))

                if pr.vr_score is not None:
                    st.metric("Efficiency Score", f"{pr.vr_score.vr_score:.3f}")

            if pr.feasibility:
                if pr.feasibility.is_feasible:
                    st.success("SAFE — all limits met")
                else:
                    violations = [_violation_to_plain(v) for v in pr.feasibility.violations]
                    st.error("NOT SAFE — " + "; ".join(violations))

            if pr.decision:
                opt_display = OPTION_DISPLAY_NAMES.get(pr.decision.selected_option, pr.decision.selected_option)
                st.info(f"Decision: **{opt_display}**")

    # C1 → C2 renegotiation banner
    c1_found = any(pr.contract and pr.contract.name == "C1" for pr in status.completed_phases)
    c2_found = any(pr.contract and pr.contract.name == "C2" for pr in status.completed_phases)
    if c1_found and c2_found:
        st.info(
            "Contract was renegotiated from C1 to C2. "
            "Safety limits remain unchanged; demand increase acknowledged."
        )


def _operator_escalation_inline(query, svc) -> None:
    """Operator-friendly inline escalation with constraint details."""
    st.warning("**The system needs your input** — a conflict was found it cannot resolve alone.")

    st.markdown("---")
    st.subheader("Choose an Action")

    options = query.options
    option_labels = []
    for opt in options:
        display = OPTION_DISPLAY_NAMES.get(opt.label, opt.label)
        badge = " (Recommended)" if opt.preserves_human_constraints else ""
        option_labels.append(f"{display}{badge}")

    selected_idx = st.radio(
        "Your choice",
        range(len(options)),
        format_func=lambda i: option_labels[i],
        key="operator_escalation_radio",
    )

    # Impact cards with constraint details
    cols = st.columns(len(options))
    for i, (col, opt) in enumerate(zip(cols, options)):
        with col:
            display = OPTION_DISPLAY_NAMES.get(opt.label, opt.label)
            is_selected = i == selected_idx

            details = opt.description
            if opt.relaxes_constraint:
                details += f"\n\n*Relaxes:* {opt.relaxes_constraint}"
            if opt.new_limit is not None:
                details += f"\n\n*New limit:* {opt.new_limit}"
            if opt.expected_deadline_gap_pct > 0:
                details += f"\n\n*Expected deadline gap:* {opt.expected_deadline_gap_pct:.1f} %"

            if opt.preserves_human_constraints:
                st.success(f"**{display}**\n\n{details}\n\n*Recommended — Keeps Workers Safe*")
            elif is_selected:
                st.warning(f"**{display}**\n\n{details}")
            else:
                st.info(f"**{display}**\n\n{details}")

    if st.button("Confirm Choice", key="operator_escalation_confirm", type="primary"):
        selected_option = options[selected_idx]
        decision = ManagerDecision(
            selected_option=selected_option.label,
            rationale="Operator decision via shift dashboard",
        )
        svc.resolve_escalation(decision)
        st.rerun()


# ── Auditor tab (Mode 3) ────────────────────────────────────────────

def _auditor_tab() -> None:
    """Mode 3: Human as Auditor — post-shift review and governance."""
    st.subheader("Shift Review")

    result = st.session_state.get("experiment_result")
    if result is None:
        st.info("No shift data to review. Run a shift from the **Run & Monitor** tab first.")
        return

    _auditor_shift_summary(result)
    st.markdown("---")
    _auditor_scorecard(result)
    st.markdown("---")
    _auditor_decisions(result)
    st.markdown("---")
    _auditor_recommendations(result)
    st.markdown("---")
    _auditor_contract_adjustment()


def _auditor_shift_summary(result) -> None:
    """Final deployed schedule name + 4 key metrics."""
    st.markdown("#### Shift Summary")

    # Find the deployed schedule (last feasible)
    deployed = None
    deployed_metrics = None
    for name in ["S3", "S1"]:
        f = result.schedule_feasibility.get(name)
        if f and f.is_feasible:
            deployed = name
            deployed_metrics = result.schedule_metrics.get(name)
            break

    if deployed:
        display_name = SCHEDULE_DISPLAY_NAMES.get(deployed, deployed)
        st.success(f"Deployed schedule: **{display_name}**")
    else:
        st.warning("No fully feasible schedule was deployed.")
        deployed_metrics = result.schedule_metrics.get("S3") or result.schedule_metrics.get("S1")

    if deployed_metrics:
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("Production Rate", f"{deployed_metrics.throughput_uph:.1f} u/h")
        with sc2:
            st.metric("Worker Fatigue", f"{deployed_metrics.fatigue_index:.2f}")
        with sc3:
            st.metric("Noise Level", f"{deployed_metrics.noise_db:.1f} dB")
        with sc4:
            st.metric("Deadline Gap", f"{deployed_metrics.deadline_gap_pct:.1f} %")

    # Monitoring summary if available
    mon = get_monitoring_service()
    shift_summary = mon.get_shift_summary()
    if shift_summary:
        st.caption(
            f"Total snapshots: {shift_summary['total_snapshots']} | "
            f"Alerts: {shift_summary['total_alerts']} "
            f"({shift_summary['violations']} violations, {shift_summary['warnings']} warnings)"
        )


def _auditor_scorecard(result) -> None:
    """Safety scorecard: pass/fail with constraint_margins traced to K."""
    st.markdown("#### Safety Scorecard  `K`")
    st.caption("Each constraint below maps to a hard constraint K in the outcome contract.")

    choices = st.session_state.get("legislator_choices", {})
    fatigue_limit = choices.get("fatigue_limit", 0.40)
    noise_limit = choices.get("noise_limit", 80.0)

    for name in ["S1", "S2", "S3"]:
        f = result.schedule_feasibility.get(name)
        if f is None:
            continue
        display_name = SCHEDULE_DISPLAY_NAMES.get(name, name)
        status_icon = "PASS" if f.is_feasible else "FAIL"
        status_color = "green" if f.is_feasible else "red"

        with st.expander(f"{display_name}: :{status_color}[{status_icon}]"):
            if f.constraint_margins:
                for constraint, margin in f.constraint_margins.items():
                    if margin >= 0:
                        st.markdown(f"- `K` **{constraint}**: margin = +{margin:.3f}")
                    else:
                        st.markdown(f"- `K` **{constraint}**: :red[OVER by {abs(margin):.3f}]")
            else:
                if f.is_feasible:
                    st.markdown("All hard constraints `K` satisfied.")
                else:
                    for v in f.violations:
                        st.markdown(f"- `K` :red[{_violation_to_plain(v)}]")
            st.caption(f"Contract limits: Fatigue ≤ {fatigue_limit:.2f}, Noise ≤ {noise_limit:.1f} dB")


def _auditor_decisions(result) -> None:
    """List all human decisions traced to contract fields."""
    st.markdown("#### Decisions Made")

    choices = st.session_state.get("legislator_choices", {})
    priority_display = " > ".join(choices.get("priority_order", []))

    decisions_found = False

    # Legislator contract definition
    if choices:
        decisions_found = True
        st.markdown("**Contract definition** (Legislator):")
        st.markdown(f"- `P` Priority ordering: {priority_display}")
        st.markdown(f"- `K` Fatigue limit: {choices.get('fatigue_limit', 0.40):.2f}")
        st.markdown(f"- `K` Noise limit: {choices.get('noise_limit', 80.0):.1f} dB")
        st.markdown(f"- `K` Cyber risk limit: {choices.get('cyber_risk_limit', 2.0):.1f}")
        st.markdown(f"- `KPI` Defect rate max: {choices.get('kpi_defect_rate_max', 0.01):.3f}")
        st.markdown(f"- `KPI` Deadline gap max: {choices.get('kpi_deadline_gap_max', 15.0):.1f}%")
        assumptions = choices.get("assumptions", {})
        if assumptions:
            st.markdown(f"- `A` Assumptions: {assumptions}")

    # Escalation decisions from phases
    for pr in result.phases:
        if pr.decision:
            decisions_found = True
            opt_display = OPTION_DISPLAY_NAMES.get(pr.decision.selected_option, pr.decision.selected_option)
            phase_label = display_phase(pr.phase, dev_mode=False)
            # Determine which contract field the escalation relates to
            field_tag = "`K`"  # escalations typically relate to hard constraints
            if pr.decision.selected_option == "Option A":
                field_tag = "`X`"  # extends shift time → context change
            elif pr.decision.selected_option == "Option B":
                field_tag = "`K`"  # relaxes fatigue constraint
            elif pr.decision.selected_option == "Option C":
                field_tag = "`P`"  # preserves constraints per priority order
            st.markdown(
                f"- {field_tag} **Step {pr.phase} ({phase_label}):** "
                f"{opt_display} — *{pr.decision.rationale}*"
            )

    # Operator feedback from session state
    feedback = st.session_state.get("operator_feedback", [])
    if feedback:
        decisions_found = True
        st.markdown("**Operator feedback events** (Partner mode):")
        for fb in feedback:
            # Map feedback to contract field
            if fb["action"] in ("Request Slower Pace", "Flag Discomfort"):
                field_tag = "`K`"
            else:
                field_tag = "`X`"
            st.markdown(f"- {field_tag} {fb['action']}")

    if not decisions_found:
        st.caption("No human decisions were recorded this shift.")


def _auditor_recommendations(result) -> None:
    """System-generated suggestions traced to contract fields."""
    st.markdown("#### Recommendations")
    st.caption("Each recommendation is linked to the contract field it would affect.")

    choices = st.session_state.get("legislator_choices", {})
    fatigue_limit = choices.get("fatigue_limit", 0.40)
    noise_limit = choices.get("noise_limit", 80.0)
    defect_limit = choices.get("kpi_defect_rate_max", 0.01)
    deadline_limit = choices.get("kpi_deadline_gap_max", 15.0)

    recs: list[tuple[str, str]] = []  # (contract_field_tag, message)

    # Check final deployed metrics
    for name in ["S3", "S1"]:
        m = result.schedule_metrics.get(name)
        f = result.schedule_feasibility.get(name)
        if m and f and f.is_feasible:
            if fatigue_limit - m.fatigue_index < 0.05:
                recs.append((
                    "`K`",
                    f"Fatigue ({m.fatigue_index:.2f}) was close to your limit ({fatigue_limit:.2f}). "
                    "Consider lowering the threshold or reducing robot speeds.",
                ))
            if noise_limit - m.noise_db < 2.0:
                recs.append((
                    "`K`",
                    f"Noise ({m.noise_db:.1f} dB) was close to your limit ({noise_limit:.1f} dB). "
                    "Consider additional noise dampening or slower operation.",
                ))
            if m.deadline_gap_pct > deadline_limit:
                recs.append((
                    "`KPI`",
                    f"Deadline gap ({m.deadline_gap_pct:.1f}%) exceeded your target ({deadline_limit:.1f}%). "
                    "Adjust demand expectations or extend shift hours.",
                ))
            if m.defect_rate > defect_limit:
                recs.append((
                    "`KPI`",
                    f"Defect rate ({m.defect_rate:.4f}) exceeded your target ({defect_limit:.3f}). "
                    "Consider reducing robot speeds or improving quality controls.",
                ))
            break

    # Check if S2 was rejected
    s2_feas = result.schedule_feasibility.get("S2")
    if s2_feas and not s2_feas.is_feasible:
        recs.append((
            "`K`",
            "The aggressive plan (Plan B) was rejected for safety violations. "
            "The system correctly protected worker well-being per your hard constraints.",
        ))

    # Feedback-based recommendations
    feedback = st.session_state.get("operator_feedback", [])
    discomfort_count = sum(1 for fb in feedback if fb["action"] == "Flag Discomfort")
    if discomfort_count > 0:
        recs.append((
            "`K`",
            f"Operator flagged discomfort {discomfort_count} time(s). "
            "Consider lowering fatigue limits for the next shift.",
        ))
    slower_count = sum(1 for fb in feedback if fb["action"] == "Request Slower Pace")
    if slower_count > 0:
        recs.append((
            "`A`",
            f"Operator requested slower pace {slower_count} time(s). "
            "Robot max speed assumption may need revision.",
        ))

    if recs:
        for tag, msg in recs:
            st.info(f"{tag}  {msg}")
    else:
        st.success("Shift completed within all parameters. No adjustments recommended.")


def _auditor_contract_adjustment() -> None:
    """Show current contract in tuple form and allow reset for next shift."""
    st.markdown("#### Adjust for Next Shift")

    choices = st.session_state.get("legislator_choices", {})
    if choices:
        # Display in C^out = <KPI, K, P, X, A> structure
        st.markdown("**Current contract** `C = <KPI, K, P, X, A>`:")
        priority_order = choices.get("priority_order", [])
        assumptions = choices.get("assumptions", {})
        contract_display = {
            "KPI": {
                "defect_rate_max": choices.get("kpi_defect_rate_max", 0.01),
                "deadline_gap_max_pct": choices.get("kpi_deadline_gap_max", 15.0),
            },
            "K": {
                "fatigue_index_max": choices.get("fatigue_limit", 0.40),
                "noise_db_max": choices.get("noise_limit", 80.0),
                "cyber_risk_level_max": choices.get("cyber_risk_limit", 2.0),
            },
            "P": choices.get("priority_order_formal", []),
            "X": {
                "shift_hours": choices.get("shift_hours", 8),
                "demand_level": choices.get("demand_level", "Normal"),
                "product_mix": choices.get("product_mix", []),
            },
            "A": assumptions,
        }
        st.json(contract_display)
    else:
        st.caption("No contract choices recorded.")

    if st.button("Adjust Contract", key="auditor_adjust"):
        st.session_state.shift_contract_signed = False
        st.info("Contract cleared. Go to the **Define Contract** tab to set up the next shift.")
        st.rerun()


def render() -> None:
    """Main render function for the Experiment Dashboard page."""
    st.header("Experiment Dashboard" if is_dev_mode() else "Shift Dashboard")

    if is_dev_mode():
        tab1, tab2, tab3 = st.tabs(["Configuration", "Phase Execution", "Contract Editor"])
        with tab1:
            _config_tab()
        with tab2:
            _execution_tab()
        with tab3:
            _contract_tab()
    else:
        tab1, tab2, tab3 = st.tabs(["Define Contract", "Run & Monitor", "Shift Review"])
        with tab1:
            _legislator_tab()
        with tab2:
            _partner_tab()
        with tab3:
            _auditor_tab()
