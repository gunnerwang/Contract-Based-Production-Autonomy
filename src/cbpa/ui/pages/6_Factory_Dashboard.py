"""Page 6: Factory Dashboard — Multi-cell CBPA lifecycle execution.

Follows the 6-phase lifecycle for the factory case study, with three
operator-facing modes (Legislator, Partner, Auditor) and a dev mode
with configuration editing and step-by-step execution.

Supports two data sources:
  - **Interactive**: run phases from the dashboard buttons
  - **CLI follow**: auto-refresh to track a CLI experiment running
    in a separate terminal (reads /tmp/cbpa_factory_live.json)

Factory layout:
  - 2 cells (A: assembly, B: test+pack)
  - 4 robots (R1-R4)
  - 3 operators (H1, H2, H3)
  - AGV cross-cell routing

Six phases:
  1. Contract & Deploy (L1-L2-L3-L4) — LEGISLATOR
  2. Monitor & Detect (L4) — AUDITOR
  3. Replan Attempt (L2-L3) — AUDITOR
  4. Escalation (Meta) — PARTNER (interactive handshake)
  5. Adapt & Stabilize (L5-L2-L3-L4-L5) — AUDITOR
  6. Shift-Close Macro-Adapt (L4-L5-L1-L2-L3-L4) — LEGISLATOR
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st

from cbpa.config.scenario import FactoryScenarioConfig
from cbpa.models.escalation import ManagerDecision
from cbpa.runner.factory_experiment import FACTORY_MANAGER_INTENT
from cbpa.service.experiment_service import ExperimentState
from cbpa.service.factory_experiment_service import FactoryExperimentService
from cbpa.ui.components.escalation_panel import escalation_panel
from cbpa.ui.components.factory_schedule_card import factory_schedule_card
from cbpa.ui.components.phase_timeline import phase_timeline
from cbpa.ui.state import get_factory_experiment_service, is_dev_mode
from cbpa.ui.theme import PHASE_COLORS, PHASE_LABELS

_logger = logging.getLogger(__name__)

FACTORY_TOTAL_PHASES = 6
_LIVE_PROGRESS_PATH = Path("/tmp/cbpa_factory_live.json")

# ── Default configuration values ─────────────────────────────────────────────

_DEFAULT_DEMAND_SPIKE_PCT = 20
_DEFAULT_NOISE_LIMIT_DB = 82.0
_DEFAULT_FATIGUE_LIMITS = {"H1": 0.40, "H2": 0.35, "H3": 0.40}
_OPERATORS = ["H1", "H2", "H3"]
_VARIANTS = ["V_A", "V_B", "V_C"]

_ALL_PRIORITIES = [
    "Worker Well-being", "Safety", "Quality",
    "Throughput", "Cost", "Flexibility",
]

_TRADEOFF_TABLE = {
    "Worker Well-being": {"throughput": "~78 u/h", "comfort": "High", "risk": "Low"},
    "Safety": {"throughput": "~80 u/h", "comfort": "High", "risk": "Low"},
    "Quality": {"throughput": "~82 u/h", "comfort": "Medium", "risk": "Low"},
    "Throughput": {"throughput": "~92 u/h", "comfort": "Low", "risk": "High"},
    "Cost": {"throughput": "~86 u/h", "comfort": "Medium", "risk": "Medium"},
    "Flexibility": {"throughput": "~84 u/h", "comfort": "Medium", "risk": "Low"},
}


# ══════════════════════════════════════════════════════════════════════════════
# CLI Live-Follow
# ══════════════════════════════════════════════════════════════════════════════


def _load_cli_progress() -> dict | None:
    """Load live progress from the CLI experiment, or None if unavailable."""
    if not _LIVE_PROGRESS_PATH.exists():
        return None
    try:
        data = json.loads(_LIVE_PROGRESS_PATH.read_text())
        if data.get("phases"):
            return data
    except Exception:
        return None
    return None


def _render_cli_live_follow() -> None:
    """Render the CLI live-follow panel at the top of the page."""
    cli_data = _load_cli_progress()
    if cli_data is None or not cli_data.get("phases"):
        return

    cli_active = cli_data.get("state") in ("starting", "running")

    # Auto-refresh while CLI is running
    if cli_active:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=3000, key="fac_cli_refresh")
        except ImportError:
            st.caption("Install `streamlit-autorefresh` for live tracking.")

    cli_state = cli_data.get("state", "unknown")
    n_done = len(cli_data["phases"])
    total = cli_data.get("total_phases", FACTORY_TOTAL_PHASES)

    if cli_state == "completed":
        st.success(f"CLI experiment completed — {n_done}/{total} phases")
    elif cli_state == "running":
        st.info(f"CLI experiment running — {n_done}/{total} phases done")
    else:
        st.info(f"CLI experiment: {cli_state}")

    # Phase timeline from CLI data
    completed_nums_cli = [p["phase"] for p in cli_data["phases"]]
    current_cli = cli_data.get("current_phase", 0)
    phase_timeline(current_phase=current_cli, completed_phases=completed_nums_cli)

    st.markdown("---")
    _render_cli_phases(cli_data)

    if cli_data.get("final_metrics"):
        _render_cli_final_metrics(cli_data["final_metrics"])

    st.markdown("---")
    st.caption(
        "Above results are from the CLI experiment. "
        "Use the tabs below for interactive operation."
    )
    st.markdown("---")


def _render_cli_phases(cli_data: dict) -> None:
    """Render phase cards from CLI live progress data."""
    st.subheader("CBPA Lifecycle Rounds (CLI)")

    for p in cli_data["phases"]:
        phase_num = p["phase"]
        label = PHASE_LABELS.get(phase_num, f"Round {phase_num}")
        title = f"Round {phase_num}: {p.get('name', '')}"

        with st.expander(title, expanded=(phase_num == len(cli_data["phases"]))):
            st.caption(f"Working mode: **{p.get('working_mode', '?')}** | {label}")

            if p.get("notes"):
                for note in p["notes"]:
                    st.markdown(f"- {note}")

            # Feasibility
            feas = p.get("feasibility")
            if feas:
                feas_ok = feas.get("is_feasible", False)
                feas_color = "green" if feas_ok else "red"
                st.markdown(
                    f"**Feasibility:** :{feas_color}["
                    f"{'FEASIBLE' if feas_ok else 'REJECTED'}]"
                )
                for v in feas.get("violations") or []:
                    st.markdown(f"  - :red[{v}]")

            # Stochastic verification
            p_feas = p.get("stochastic_p_feasible")
            if p_feas is not None:
                st.markdown(f"**P(feasible):** {p_feas:.0%} (200 MC samples)")

            # Manager decision
            if p.get("manager_decision"):
                st.markdown(f"**Manager selected:** :violet[{p['manager_decision']}]")

            # Factory metrics
            fm = p.get("factory_metrics")
            if fm:
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric(
                        "Throughput",
                        f"{fm.get('total_throughput_uph', 0):.1f} u/h",
                    )
                with mc2:
                    st.metric(
                        "Noise",
                        f"{fm.get('factory_noise_db', 0):.1f} dB",
                    )
                with mc3:
                    st.metric(
                        "Balance Loss",
                        f"{fm.get('cell_balance_loss_pct', 0):.1f}%",
                    )


def _render_cli_final_metrics(fm: dict) -> None:
    """Render final metrics from CLI progress data."""
    st.markdown("---")
    st.subheader("Final Factory Metrics (CLI)")

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Total Throughput", f"{fm.get('total_throughput_uph', 0):.1f} u/h")
    with mc2:
        st.metric("Factory Noise", f"{fm.get('factory_noise_db', 0):.1f} dB")
    with mc3:
        st.metric("Cell Balance Loss", f"{fm.get('cell_balance_loss_pct', 0):.1f}%")
    with mc4:
        st.metric("Factory Energy", f"{fm.get('factory_energy_kwh', 0):.1f} kWh")

    fat = fm.get("operator_fatigue", {})
    if fat:
        st.markdown("**Per-Operator Fatigue:**")
        fat_cols = st.columns(len(fat))
        for col, (op_id, f_val) in zip(fat_cols, fat.items()):
            with col:
                limit = _DEFAULT_FATIGUE_LIMITS.get(op_id, 0.40)
                status_lbl = "PASS" if f_val <= limit else "FAIL"
                st.metric(op_id, f"{f_val:.3f}", delta=status_lbl)


# ══════════════════════════════════════════════════════════════════════════════
# Dev Mode Tabs
# ══════════════════════════════════════════════════════════════════════════════


def _dev_config_tab() -> None:
    """Dev mode: Configuration tab with editable factory parameters."""
    st.subheader("Factory Configuration")

    config: FactoryScenarioConfig = st.session_state.get(
        "factory_config", FactoryScenarioConfig()
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Disturbance Parameters**")
        demand_spike = st.slider(
            "Demand Spike %",
            min_value=0,
            max_value=50,
            value=int(getattr(config, "demand_spike_pct", _DEFAULT_DEMAND_SPIKE_PCT)),
            step=1,
            key="fac_cfg_demand_spike",
        )

        st.markdown("**Operator Absence**")
        absence_on = st.checkbox(
            "Enable operator absence",
            value=getattr(config, "enable_operator_absence", True),
            key="fac_cfg_absence_on",
        )
        absence_hour = st.slider(
            "Absence hour",
            min_value=1,
            max_value=8,
            value=int(getattr(config, "absence_hour", 3)),
            key="fac_cfg_absence_hour",
        )
        absence_operator = st.selectbox(
            "Absent operator",
            options=_OPERATORS,
            index=0,
            key="fac_cfg_absence_op",
        )

        st.markdown("**Supply Delay**")
        supply_delay_on = st.checkbox(
            "Enable supply delay",
            value=getattr(config, "enable_supply_delay", False),
            key="fac_cfg_supply_delay_on",
        )
        supply_delay_variant = st.selectbox(
            "Delayed variant",
            options=_VARIANTS,
            index=0,
            key="fac_cfg_supply_variant",
        )

    with col_right:
        st.markdown("**Noise & Fatigue Limits**")
        noise_limit = st.slider(
            "Factory noise limit (dB)",
            min_value=75.0,
            max_value=90.0,
            value=float(getattr(config, "factory_noise_limit_db", _DEFAULT_NOISE_LIMIT_DB)),
            step=1.0,
            key="fac_cfg_noise_limit",
        )

        st.markdown("**Per-Operator Fatigue Limits**")
        fatigue_h1 = st.slider(
            "H1 fatigue limit",
            min_value=0.20,
            max_value=0.60,
            value=_DEFAULT_FATIGUE_LIMITS["H1"],
            step=0.01,
            key="fac_cfg_fatigue_h1",
        )
        fatigue_h2 = st.slider(
            "H2 fatigue limit",
            min_value=0.20,
            max_value=0.60,
            value=_DEFAULT_FATIGUE_LIMITS["H2"],
            step=0.01,
            key="fac_cfg_fatigue_h2",
        )
        fatigue_h3 = st.slider(
            "H3 fatigue limit",
            min_value=0.20,
            max_value=0.60,
            value=_DEFAULT_FATIGUE_LIMITS["H3"],
            step=0.01,
            key="fac_cfg_fatigue_h3",
        )

    # Action buttons
    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("Apply Configuration", type="primary", key="fac_cfg_apply"):
            st.session_state.factory_config = config
            st.success("Configuration applied.")
            st.rerun()

    with bc2:
        if st.button("Reset to Defaults", key="fac_cfg_reset"):
            st.session_state.factory_config = FactoryScenarioConfig()
            svc = get_factory_experiment_service()
            svc.reset(config=FactoryScenarioConfig())
            st.success("Reset to defaults.")
            st.rerun()


def _dev_execution_tab() -> None:
    """Dev mode: Phase execution with step-by-step + run-all."""
    svc = get_factory_experiment_service()
    status = svc.status
    completed_nums = [pr.phase for pr in status.completed_phases]

    # Manager intent
    with st.expander("Manager Intent", expanded=False):
        st.markdown(f"*{FACTORY_MANAGER_INTENT}*")

    # Progress timeline
    phase_timeline(
        current_phase=status.current_phase,
        completed_phases=completed_nums,
    )

    st.markdown("---")

    # Action buttons
    bc1, bc2, bc3 = st.columns(3)

    with bc1:
        if st.button(
            "Run All Phases",
            type="primary",
            disabled=status.state not in (
                ExperimentState.IDLE, ExperimentState.COMPLETED
            ),
            key="fac_exec_run_all",
        ):
            svc.reset(config=st.session_state.factory_config)
            result = svc.run_all()
            st.session_state.factory_experiment_result = result
            st.rerun()

    with bc2:
        next_phase = max(completed_nums) + 1 if completed_nums else 1
        can_step = (
            status.state in (
                ExperimentState.IDLE,
                ExperimentState.RUNNING,
                ExperimentState.COMPLETED,
            )
            and next_phase <= FACTORY_TOTAL_PHASES
        )
        if st.button(
            f"Step Phase {next_phase}",
            disabled=not can_step,
            key="fac_exec_step",
        ):
            if next_phase == 1 and status.state in (
                ExperimentState.IDLE, ExperimentState.COMPLETED
            ):
                svc.reset(config=st.session_state.factory_config)
            svc.run_phase(next_phase)
            if svc.status.state == ExperimentState.COMPLETED:
                st.session_state.factory_experiment_result = svc.status.result
            st.rerun()

    with bc3:
        if st.button("Reset Experiment", key="fac_exec_reset"):
            svc.reset(config=st.session_state.factory_config)
            st.session_state.factory_experiment_result = None
            st.rerun()

    # Escalation handshake
    if status.state == ExperimentState.AWAITING_ESCALATION:
        st.markdown("---")
        query = svc.get_escalation_query()
        if query is not None:
            decision = escalation_panel(query, key_prefix="fac_escalation")
            if decision is not None:
                svc.resolve_escalation(
                    ManagerDecision(
                        selected_option=decision.selected_option,
                        rationale=decision.rationale,
                    )
                )
                st.rerun()

    # Phase result cards
    if status.completed_phases:
        st.markdown("---")
        st.subheader("CBPA Lifecycle Rounds")
        _render_phase_cards(status.completed_phases)
        _final_metrics_section(status)


def _dev_contract_tab() -> None:
    """Dev mode: Display the current factory contract."""
    svc = get_factory_experiment_service()
    status = svc.status

    active_contract = None
    for pr in reversed(status.completed_phases):
        if pr.contract is not None:
            active_contract = pr.contract
            break

    if active_contract is not None:
        st.subheader(f"Active Contract: {active_contract.name}")
        st.json(active_contract.model_dump() if hasattr(active_contract, "model_dump") else str(active_contract))
    else:
        st.info("No contract generated yet. Run Phase 1 to create the initial contract.")

    # Show legislator choices if available
    choices = st.session_state.get("factory_legislator_choices")
    if choices:
        st.markdown("---")
        st.subheader("Legislator Choices (Session)")
        st.json(choices)


# ══════════════════════════════════════════════════════════════════════════════
# Operator Mode Tabs
# ══════════════════════════════════════════════════════════════════════════════


def _legislator_tab() -> None:
    """Operator mode: Define Contract tab (Legislator role).

    The operator sets factory goals, per-operator fatigue limits,
    noise limit, variant routing preferences, shift context, and assumptions.
    """
    st.subheader("Define Factory Contract")
    st.caption("Role: LEGISLATOR — Set goals and constraints for this shift.")

    # Priority ordering
    st.markdown("**Priority Ordering**")
    priority_order = st.multiselect(
        "Drag to reorder priorities (first = highest)",
        options=_ALL_PRIORITIES,
        default=["Worker Well-being", "Safety", "Quality"],
        key="fac_leg_priorities",
    )

    if priority_order:
        _tradeoff_preview(priority_order)

    st.markdown("---")

    # KPI targets
    st.markdown("**KPI Targets**")
    col1, col2 = st.columns(2)
    with col1:
        throughput_target = st.number_input(
            "Throughput target (u/h)",
            min_value=50.0,
            max_value=120.0,
            value=84.0,
            step=1.0,
            key="fac_leg_throughput_target",
        )
    with col2:
        quality_target = st.number_input(
            "Quality target (defect rate max)",
            min_value=0.001,
            max_value=0.05,
            value=0.01,
            step=0.001,
            format="%.3f",
            key="fac_leg_quality_target",
        )

    st.markdown("---")

    # Hard constraints
    st.markdown("**Hard Constraints**")
    col_a, col_b = st.columns(2)
    with col_a:
        noise_limit = st.slider(
            "Factory noise limit (dB)",
            min_value=75.0,
            max_value=90.0,
            value=_DEFAULT_NOISE_LIMIT_DB,
            step=1.0,
            key="fac_leg_noise_limit",
        )
        quality_hard = st.number_input(
            "Quality hard limit (defect rate)",
            min_value=0.001,
            max_value=0.05,
            value=0.015,
            step=0.001,
            format="%.3f",
            key="fac_leg_quality_hard",
        )

    with col_b:
        st.markdown("Per-operator fatigue limits:")
        fatigue_h1 = st.slider(
            "H1 max fatigue",
            min_value=0.20,
            max_value=0.60,
            value=_DEFAULT_FATIGUE_LIMITS["H1"],
            step=0.01,
            key="fac_leg_fatigue_h1",
        )
        fatigue_h2 = st.slider(
            "H2 max fatigue",
            min_value=0.20,
            max_value=0.60,
            value=_DEFAULT_FATIGUE_LIMITS["H2"],
            step=0.01,
            key="fac_leg_fatigue_h2",
        )
        fatigue_h3 = st.slider(
            "H3 max fatigue",
            min_value=0.20,
            max_value=0.60,
            value=_DEFAULT_FATIGUE_LIMITS["H3"],
            step=0.01,
            key="fac_leg_fatigue_h3",
        )

    st.markdown("---")

    # Shift context
    st.markdown("**Shift Context**")
    ctx_col1, ctx_col2, ctx_col3 = st.columns(3)
    with ctx_col1:
        shift_duration = st.selectbox(
            "Shift duration",
            options=[6, 8, 10, 12],
            index=1,
            key="fac_leg_shift_duration",
        )
    with ctx_col2:
        demand_level = st.selectbox(
            "Demand level",
            options=["Normal", "High (+20%)", "Surge (+30%)"],
            index=0,
            key="fac_leg_demand_level",
        )
    with ctx_col3:
        active_variants = st.multiselect(
            "Active variants",
            options=_VARIANTS,
            default=_VARIANTS,
            key="fac_leg_variants",
        )

    st.markdown("---")

    # Assumptions
    st.markdown("**Assumptions & Risk Flags**")
    assume_col1, assume_col2 = st.columns(2)
    with assume_col1:
        h2_risk = st.checkbox(
            "H2 availability risk (may leave early)",
            value=True,
            key="fac_leg_h2_risk",
        )
    with assume_col2:
        supply_risk = st.checkbox(
            "Supply chain risk (possible variant delay)",
            value=False,
            key="fac_leg_supply_risk",
        )

    st.markdown("---")

    # Approve contract
    if st.button(
        "Approve Contract & Begin Shift",
        type="primary",
        key="fac_leg_approve",
    ):
        choices = {
            "priority_order": priority_order,
            "throughput_target": throughput_target,
            "quality_target": quality_target,
            "noise_limit": noise_limit,
            "quality_hard_limit": quality_hard,
            "fatigue_limits": {
                "H1": fatigue_h1,
                "H2": fatigue_h2,
                "H3": fatigue_h3,
            },
            "shift_duration": shift_duration,
            "demand_level": demand_level,
            "active_variants": active_variants,
            "h2_availability_risk": h2_risk,
            "supply_chain_risk": supply_risk,
        }
        st.session_state.factory_legislator_choices = choices
        st.session_state.factory_contract_signed = True
        st.success("Contract approved. Switch to 'Run & Monitor' tab to begin.")
        st.rerun()


def _partner_tab() -> None:
    """Operator mode: Run & Monitor tab (Partner role).

    Phase execution with escalation handshake and live KPIs.
    """
    st.subheader("Run & Monitor")
    st.caption("Role: PARTNER — Execute the shift and handle escalations.")

    # Gate check: contract must be signed
    if not st.session_state.get("factory_contract_signed", False):
        st.warning(
            "No contract signed. Please go to 'Define Contract' tab first."
        )
        return

    svc = get_factory_experiment_service()
    status = svc.status
    completed_nums = [pr.phase for pr in status.completed_phases]

    # Phase timeline
    phase_timeline(
        current_phase=status.current_phase,
        completed_phases=completed_nums,
    )

    st.markdown("---")

    # Action buttons
    bc1, bc2, bc3 = st.columns(3)

    with bc1:
        if st.button(
            "Start Shift (Run All)",
            type="primary",
            disabled=status.state not in (
                ExperimentState.IDLE, ExperimentState.COMPLETED
            ),
            key="fac_partner_run_all",
        ):
            svc.reset(config=st.session_state.factory_config)
            result = svc.run_all()
            st.session_state.factory_experiment_result = result
            st.rerun()

    with bc2:
        next_phase = max(completed_nums) + 1 if completed_nums else 1
        can_step = (
            status.state in (
                ExperimentState.IDLE,
                ExperimentState.RUNNING,
                ExperimentState.COMPLETED,
            )
            and next_phase <= FACTORY_TOTAL_PHASES
        )
        if st.button(
            f"Step Phase {next_phase}",
            disabled=not can_step,
            key="fac_partner_step",
        ):
            if next_phase == 1 and status.state in (
                ExperimentState.IDLE, ExperimentState.COMPLETED
            ):
                svc.reset(config=st.session_state.factory_config)
            svc.run_phase(next_phase)
            if svc.status.state == ExperimentState.COMPLETED:
                st.session_state.factory_experiment_result = svc.status.result
            st.rerun()

    with bc3:
        if st.button("Reset", key="fac_partner_reset"):
            svc.reset(config=st.session_state.factory_config)
            st.session_state.factory_experiment_result = None
            st.rerun()

    # Escalation handshake (Phase 4 interactive)
    if status.state == ExperimentState.AWAITING_ESCALATION:
        st.markdown("---")
        st.error("Escalation required — your decision is needed.")
        query = svc.get_escalation_query()
        if query is not None:
            decision = escalation_panel(query, key_prefix="fac_partner_esc")
            if decision is not None:
                svc.resolve_escalation(
                    ManagerDecision(
                        selected_option=decision.selected_option,
                        rationale=decision.rationale,
                    )
                )
                st.rerun()

    # Phase result cards
    if status.completed_phases:
        st.markdown("---")
        _render_phase_cards(status.completed_phases)

    # Live Factory KPIs
    _render_live_kpis(status)


def _auditor_tab() -> None:
    """Operator mode: Shift Review tab (Auditor role).

    Post-shift summary with safety scorecard, per-operator fatigue,
    decisions traced to contract, and recommendations.
    """
    st.subheader("Shift Review")
    st.caption("Role: AUDITOR — Review shift outcomes and compliance.")

    svc = get_factory_experiment_service()
    status = svc.status

    if status.state != ExperimentState.COMPLETED:
        st.info("Shift not yet completed. Complete all 6 phases to see the review.")
        return

    result = status.result
    if result is None or result.final_metrics is None:
        st.warning("No final metrics available.")
        return

    fm = result.final_metrics

    # ── Final factory metrics ──
    st.markdown("### Factory KPI Summary")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Total Throughput", f"{fm.total_throughput_uph:.1f} u/h")
    with mc2:
        st.metric("Factory Noise", f"{fm.factory_noise_db:.1f} dB")
    with mc3:
        st.metric("Cell Balance Loss", f"{fm.cell_balance_loss_pct:.1f}%")
    with mc4:
        st.metric("Factory Energy", f"{fm.factory_energy_kwh:.1f} kWh")

    mc5, mc6 = st.columns(2)
    with mc5:
        st.metric("AGV Utilization", f"{fm.agv_utilization:.1%}")
    with mc6:
        st.metric("Defect Rate", f"{fm.factory_defect_rate:.4f}")

    st.markdown("---")

    # ── Per-operator fatigue scorecard ──
    st.markdown("### Per-Operator Fatigue Scorecard")
    if fm.operator_fatigue:
        choices = st.session_state.get("factory_legislator_choices", {})
        limits = choices.get("fatigue_limits", _DEFAULT_FATIGUE_LIMITS)

        fat_cols = st.columns(len(fm.operator_fatigue))
        all_pass = True
        for col, (op_id, fat_val) in zip(fat_cols, fm.operator_fatigue.items()):
            limit = limits.get(op_id, 0.40)
            passed = fat_val <= limit
            if not passed:
                all_pass = False
            with col:
                status_icon = ":green[PASS]" if passed else ":red[FAIL]"
                st.metric(
                    f"{op_id} (limit: {limit:.2f})",
                    f"{fat_val:.3f}",
                    delta=f"margin: {limit - fat_val:+.3f}",
                )
                st.markdown(status_icon)

        if all_pass:
            st.success("All operators within fatigue limits.")
        else:
            st.error("One or more operators exceeded fatigue limits.")
    else:
        st.info("No operator fatigue data available.")

    st.markdown("---")

    # ── Per-cell breakdown ──
    st.markdown("### Per-Cell Breakdown")
    if fm.cell_metrics:
        cell_cols = st.columns(len(fm.cell_metrics))
        for col, (cell_id, cm) in zip(cell_cols, fm.cell_metrics.items()):
            with col:
                st.markdown(f"**Cell {cell_id}**")
                st.markdown(f"- Throughput: {cm.throughput_uph:.1f} u/h")
                st.markdown(f"- Noise: {cm.noise_db:.1f} dB")
                st.markdown(f"- Energy: {cm.energy_kwh:.1f} kWh")
                st.markdown(f"- Defect rate: {cm.defect_rate:.4f}")
    else:
        st.info("No per-cell data available.")

    st.markdown("---")

    # ── Decisions traced to contract ──
    st.markdown("### Decision Trace")

    # Legislator contract choices
    choices = st.session_state.get("factory_legislator_choices")
    if choices:
        with st.expander("Legislator Contract Choices", expanded=False):
            st.markdown(f"**Priority order:** {' > '.join(choices.get('priority_order', []))}")
            st.markdown(f"**Noise limit:** {choices.get('noise_limit', 82)} dB")
            fat_lims = choices.get("fatigue_limits", {})
            st.markdown(
                f"**Fatigue limits:** "
                f"H1={fat_lims.get('H1', 0.40)}, "
                f"H2={fat_lims.get('H2', 0.35)}, "
                f"H3={fat_lims.get('H3', 0.40)}"
            )
            st.markdown(f"**Demand level:** {choices.get('demand_level', 'Normal')}")
            st.markdown(f"**Active variants:** {', '.join(choices.get('active_variants', _VARIANTS))}")

    # Phase 4 manager decision
    phase4_result = None
    for pr in status.completed_phases:
        if pr.phase == 4 and pr.manager_decision:
            phase4_result = pr
            break

    if phase4_result:
        with st.expander("Phase 4 Manager Decision (Escalation)", expanded=False):
            st.markdown(f"**Selected:** :violet[{phase4_result.manager_decision}]")
            if hasattr(phase4_result, "escalation_options") and phase4_result.escalation_options:
                st.markdown("**Options considered:**")
                for opt in phase4_result.escalation_options:
                    preserves = "Yes" if opt.get("preserves_human_constraints") else "No"
                    st.markdown(
                        f"- {opt.get('label', '?')}: {opt.get('description', '')}"
                        f" (preserves constraints: {preserves})"
                    )

    st.markdown("---")

    # ── Recommendations ──
    st.markdown("### Recommendations for Next Shift")
    recommendations = _generate_recommendations(fm, choices)
    for rec in recommendations:
        st.markdown(f"- {rec}")


# ══════════════════════════════════════════════════════════════════════════════
# Shared Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _render_phase_cards(completed_phases: list) -> None:
    """Render phase result cards from completed phases."""
    for pr in completed_phases:
        color = PHASE_COLORS.get(pr.phase, "#757575")
        label = PHASE_LABELS.get(pr.phase, f"Round {pr.phase}")
        title = f"Round {pr.phase}: {pr.phase_name}"

        with st.expander(title, expanded=(pr.phase == len(completed_phases))):
            st.caption(f"Working mode: **{pr.working_mode}** | {label}")

            # Notes
            if pr.notes:
                for note in pr.notes:
                    st.markdown(f"- {note}")

            # Factory schedule
            if pr.factory_schedule is not None:
                factory_schedule_card(pr.factory_schedule, metrics=pr.factory_metrics)

            # Feasibility
            if pr.feasibility is not None:
                feas_ok = pr.feasibility.is_feasible
                feas_color = "green" if feas_ok else "red"
                st.markdown(
                    f"**Feasibility:** :{feas_color}["
                    f"{'FEASIBLE' if feas_ok else 'REJECTED'}]"
                )
                if pr.feasibility.violations:
                    for v in pr.feasibility.violations:
                        st.markdown(f"  - :red[{v}]")

            # Stochastic verification
            if pr.stochastic_report is not None:
                p_feas = pr.stochastic_report.p_feasible
                st.markdown(f"**P(feasible):** {p_feas:.0%} (200 MC samples)")

            # Escalation options
            if pr.escalation_options:
                st.markdown("**Remediation Options:**")
                for opt in pr.escalation_options:
                    preserves = "Yes" if opt.get("preserves_human_constraints") else "No"
                    gap = opt.get("expected_deadline_gap_pct", "?")
                    st.markdown(
                        f"- **{opt['label']}**: {opt['description']}\n"
                        f"  - Preserves human constraints: {preserves} | Gap: ~{gap}%"
                    )

            # Manager decision
            if pr.manager_decision:
                st.markdown(f"**Manager selected:** :violet[{pr.manager_decision}]")


def _render_live_kpis(status) -> None:
    """Render live factory KPIs with constraint margins after phases complete."""
    if not status.completed_phases:
        return

    # Get the latest metrics from the most recent phase
    latest_metrics = None
    for pr in reversed(status.completed_phases):
        if pr.factory_metrics is not None:
            latest_metrics = pr.factory_metrics
            break

    if latest_metrics is None:
        return

    st.markdown("---")
    st.subheader("Live Factory KPIs")

    choices = st.session_state.get("factory_legislator_choices", {})
    noise_limit = choices.get("noise_limit", _DEFAULT_NOISE_LIMIT_DB)
    fatigue_limits = choices.get("fatigue_limits", _DEFAULT_FATIGUE_LIMITS)

    fm = latest_metrics

    # Main KPIs
    kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
    with kpi_c1:
        st.metric(
            "Total Throughput",
            f"{fm.total_throughput_uph:.1f} u/h",
        )
    with kpi_c2:
        noise_margin = noise_limit - fm.factory_noise_db
        st.metric(
            "Factory Noise",
            f"{fm.factory_noise_db:.1f} dB",
            delta=f"margin: {noise_margin:+.1f} dB",
            delta_color="normal" if noise_margin >= 0 else "inverse",
        )
    with kpi_c3:
        st.metric(
            "Cell Balance Loss",
            f"{fm.cell_balance_loss_pct:.1f}%",
        )

    # Per-cell throughput
    if fm.cell_metrics:
        cell_cols = st.columns(len(fm.cell_metrics))
        for col, (cell_id, cm) in zip(cell_cols, fm.cell_metrics.items()):
            with col:
                st.metric(f"Cell {cell_id} Throughput", f"{cm.throughput_uph:.1f} u/h")

    # Per-operator fatigue with limits
    if fm.operator_fatigue:
        st.markdown("**Per-Operator Fatigue:**")
        op_cols = st.columns(len(fm.operator_fatigue))
        for col, (op_id, fat_val) in zip(op_cols, fm.operator_fatigue.items()):
            limit = fatigue_limits.get(op_id, 0.40)
            margin = limit - fat_val
            with col:
                st.metric(
                    f"{op_id} (limit: {limit:.2f})",
                    f"{fat_val:.3f}",
                    delta=f"margin: {margin:+.3f}",
                    delta_color="normal" if margin >= 0 else "inverse",
                )

    # AGV utilization
    if hasattr(fm, "agv_utilization") and fm.agv_utilization is not None:
        st.metric("AGV Utilization", f"{fm.agv_utilization:.1%}")


def _final_metrics_section(status) -> None:
    """Render final factory metrics summary after all rounds complete."""
    if status.state != ExperimentState.COMPLETED:
        return

    result = status.result
    if result is None or result.final_metrics is None:
        return

    fm = result.final_metrics
    st.markdown("---")
    st.subheader("Final Factory Metrics")

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Total Throughput", f"{fm.total_throughput_uph:.1f} u/h")
    with mc2:
        st.metric("Factory Noise", f"{fm.factory_noise_db:.1f} dB")
    with mc3:
        st.metric("Cell Balance Loss", f"{fm.cell_balance_loss_pct:.1f}%")
    with mc4:
        st.metric("Factory Energy", f"{fm.factory_energy_kwh:.1f} kWh")

    mc5, mc6 = st.columns(2)
    with mc5:
        st.metric("AGV Utilization", f"{fm.agv_utilization:.1%}")
    with mc6:
        st.metric("Defect Rate", f"{fm.factory_defect_rate:.4f}")

    # Per-operator fatigue
    if fm.operator_fatigue:
        st.markdown("**Per-Operator Fatigue:**")
        fat_cols = st.columns(len(fm.operator_fatigue))
        for col, (op_id, fat) in zip(fat_cols, fm.operator_fatigue.items()):
            limit = _DEFAULT_FATIGUE_LIMITS.get(op_id, 0.40)
            with col:
                status_label = "PASS" if fat <= limit else "FAIL"
                st.metric(op_id, f"{fat:.3f}", delta=status_label)

    # Per-cell breakdown
    if fm.cell_metrics:
        st.markdown("**Per-Cell Breakdown:**")
        cell_cols = st.columns(len(fm.cell_metrics))
        for col, (cell_id, cm) in zip(cell_cols, fm.cell_metrics.items()):
            with col:
                st.markdown(f"**Cell {cell_id}**")
                st.markdown(f"- Throughput: {cm.throughput_uph:.1f} u/h")
                st.markdown(f"- Noise: {cm.noise_db:.1f} dB")
                st.markdown(f"- Energy: {cm.energy_kwh:.1f} kWh")
                st.markdown(f"- Defect rate: {cm.defect_rate:.4f}")


def _tradeoff_preview(priority_order: list[str]) -> None:
    """Show estimated trade-offs based on top priority."""
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


def _generate_recommendations(fm, choices: dict | None) -> list[str]:
    """Generate shift-end recommendations based on metrics and contract."""
    recommendations = []

    # Fatigue-based recommendations
    if fm.operator_fatigue:
        limits = (choices or {}).get("fatigue_limits", _DEFAULT_FATIGUE_LIMITS)
        for op_id, fat_val in fm.operator_fatigue.items():
            limit = limits.get(op_id, 0.40)
            if fat_val > limit:
                recommendations.append(
                    f"{op_id} exceeded fatigue limit ({fat_val:.3f} > {limit:.2f}). "
                    f"Consider reducing task load or rotating duties next shift."
                )
            elif fat_val > limit * 0.9:
                recommendations.append(
                    f"{op_id} near fatigue limit ({fat_val:.3f} / {limit:.2f}). "
                    f"Monitor closely next shift."
                )

    # Noise-based recommendations
    noise_limit = (choices or {}).get("noise_limit", _DEFAULT_NOISE_LIMIT_DB)
    if fm.factory_noise_db > noise_limit:
        recommendations.append(
            f"Factory noise exceeded limit ({fm.factory_noise_db:.1f} > {noise_limit} dB). "
            f"Consider reducing robot speeds or adding acoustic dampening."
        )

    # Cell balance
    if hasattr(fm, "cell_balance_loss_pct") and fm.cell_balance_loss_pct > 5.0:
        recommendations.append(
            f"Cell balance loss is high ({fm.cell_balance_loss_pct:.1f}%). "
            f"Review task distribution between cells A and B."
        )

    # AGV utilization
    if hasattr(fm, "agv_utilization") and fm.agv_utilization < 0.5:
        recommendations.append(
            f"AGV utilization is low ({fm.agv_utilization:.1%}). "
            f"Consider consolidating cross-cell transfers."
        )
    elif hasattr(fm, "agv_utilization") and fm.agv_utilization > 0.9:
        recommendations.append(
            f"AGV utilization near capacity ({fm.agv_utilization:.1%}). "
            f"May become a bottleneck if demand increases."
        )

    if not recommendations:
        recommendations.append(
            "All metrics within acceptable ranges. No changes recommended."
        )

    return recommendations


# ══════════════════════════════════════════════════════════════════════════════
# Main render entry point
# ══════════════════════════════════════════════════════════════════════════════


def render() -> None:
    """Main render function for the Factory Dashboard page."""
    st.title("Factory Dashboard")
    st.caption("Multi-cell CBPA Lifecycle — 2 cells, 4 robots, 3 operators")

    # Initialize session state defaults
    if "factory_config" not in st.session_state:
        st.session_state.factory_config = FactoryScenarioConfig()
    if "factory_experiment_result" not in st.session_state:
        st.session_state.factory_experiment_result = None
    if "factory_contract_signed" not in st.session_state:
        st.session_state.factory_contract_signed = False
    if "factory_legislator_choices" not in st.session_state:
        st.session_state.factory_legislator_choices = None

    # ── CLI live-follow section (always at top) ──
    _render_cli_live_follow()

    # ── Three-mode UI ──
    if is_dev_mode():
        tab_config, tab_exec, tab_contract = st.tabs([
            "Configuration",
            "Phase Execution",
            "Contract Editor",
        ])
        with tab_config:
            _dev_config_tab()
        with tab_exec:
            _dev_execution_tab()
        with tab_contract:
            _dev_contract_tab()
    else:
        tab_legislator, tab_partner, tab_auditor = st.tabs([
            "Define Contract",
            "Run & Monitor",
            "Shift Review",
        ])
        with tab_legislator:
            _legislator_tab()
        with tab_partner:
            _partner_tab()
        with tab_auditor:
            _auditor_tab()


# Streamlit page entry point
render()
