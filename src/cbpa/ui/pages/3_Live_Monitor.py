"""Page 3: Live Monitor — Real-time KPI gauges + alerts.

When ``--with-dashboard`` is active, this page reads the live demo JSON
file produced by the batch experiment and updates in real time.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from cbpa.models.contract import make_c1
from cbpa.models.metrics import SimulationMetrics
from cbpa.service.monitoring_service import MonitoringService
from cbpa.ui.components.alert_feed import alert_feed
from cbpa.ui.components.constraint_indicator import constraint_panel
from cbpa.ui.components.kpi_gauges import kpi_gauge_row
from cbpa.ui.state import get_experiment_service, get_monitoring_service, is_dev_mode
from cbpa.ui.theme import display_phase

_LIVE_FILE = Path("data/results/live_demo.json")
_FACTORY_LIVE_FILE = Path("/tmp/cbpa_factory_live.json")


# ── Live demo file reader ─────────────────────────────────────────

def _read_live_demo() -> dict | None:
    """Read the live demo JSON written by the batch experiment."""
    if not _LIVE_FILE.exists():
        return None
    try:
        data = json.loads(_LIVE_FILE.read_text())
        return data if data.get("phases") else None
    except (json.JSONDecodeError, OSError):
        return None


def _render_live_demo(data: dict) -> None:
    """Render the live demo feed from the batch experiment."""
    state = data.get("state", "unknown")
    elapsed = data.get("elapsed_s", 0)
    phases = data.get("phases", [])

    # Status banner
    if state == "running":
        st.info(f"Experiment running — {len(phases)} phase(s) completed ({elapsed:.1f}s elapsed)")
    elif state == "completed":
        st.success(f"Experiment completed — {len(phases)} phases ({elapsed:.1f}s)")
    else:
        st.warning(f"Status: {state}")

    if not phases:
        return

    # ── Phase timeline ─────────────────────────────────────────────
    st.subheader("Phase Timeline")
    for p in phases:
        shift = p.get("shift", 1)
        phase = p.get("phase", "?")
        mode = p.get("working_mode", "")
        desc = p.get("description", "")
        ts = p.get("timestamp", 0)

        mode_colors = {
            "legislator": "blue",
            "auditor": "green",
            "partner": "orange",
        }
        color = mode_colors.get(mode, "gray")

        prefix = f"**Shift {shift} |** " if shift > 1 else ""
        st.markdown(
            f"{prefix}**Phase {phase}** "
            f"[:{color}[{mode}]] "
            f"— {desc}  "
            f"<small style='color:gray'>+{ts:.1f}s</small>",
            unsafe_allow_html=True,
        )

        # Show metrics if available
        m = p.get("metrics")
        if m:
            cols = st.columns(6)
            cols[0].metric("Throughput", f"{m['throughput_uph']} u/h")
            cols[1].metric("Defect", f"{m['defect_rate']}")
            cols[2].metric("Noise", f"{m['noise_db']} dB")
            cols[3].metric("Fatigue", f"{m['fatigue_index']}")
            cols[4].metric("Energy", f"{m['energy_kwh']} kWh")
            cols[5].metric("Gap", f"{m['deadline_gap_pct']}%")

        # Show V/R score
        vr = p.get("vr_score")
        feasible = p.get("feasible")
        if vr is not None:
            feas_label = "Feasible" if feasible else "Infeasible"
            feas_icon = "white_check_mark" if feasible else "x"
            st.caption(f"V/R = {vr:.3f} | :{feas_icon}: {feas_label}")

        # Show violations
        violations = p.get("violations")
        if violations:
            for v in violations:
                st.error(f"Violation: {v}")

        # Show Pareto
        pareto = p.get("pareto")
        if pareto:
            nd = pareto.get("non_dominated", [])
            sel = pareto.get("selected", "")
            st.caption(
                f"Pareto: {pareto['candidates']} candidates, "
                f"{len(nd)} non-dominated [{', '.join(nd)}], selected={sel}"
            )

        # Show decision
        decision = p.get("decision")
        if decision:
            st.info(
                f"Manager decision: **{decision['selected_option']}** — "
                f"{decision.get('rationale', '')}"
            )

        # Show guard events
        guards = p.get("guard_events")
        if guards:
            for g in guards:
                st.warning(
                    f"Guard: {g['guard']} — {g['metric']}={g['value']} "
                    f"(threshold {g['threshold']}) -> {g['action']}"
                )

        st.divider()

    # ── Latest metrics gauges ──────────────────────────────────────
    # Find the last phase with metrics
    latest_metrics = None
    for p in reversed(phases):
        if p.get("metrics"):
            latest_metrics = p["metrics"]
            break

    if latest_metrics:
        st.subheader("Latest Schedule Metrics")
        kpi_gauge_row(latest_metrics)

        # Factory-specific extras
        factory_data = None
        for p in reversed(phases):
            if p.get("factory"):
                factory_data = p["factory"]
                break
        if factory_data:
            st.subheader("Factory Metrics")
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.metric("Cell Balance Loss", f"{factory_data.get('cell_balance_loss_pct', 0):.1f}%")
            with fc2:
                st.metric("AGV Utilization", f"{factory_data.get('agv_utilization', 0):.1%}")
            with fc3:
                op_fat = factory_data.get("operator_fatigue", {})
                if op_fat:
                    for op_id, fat in op_fat.items():
                        st.metric(f"{op_id} Fatigue", f"{fat:.3f}")

        # Notes from factory phases
        for p in reversed(phases):
            notes = p.get("notes")
            if notes:
                with st.expander(f"Phase {p['phase']} Notes"):
                    for n in notes:
                        st.markdown(f"- {n}")
                break

        st.markdown("---")
        st.subheader("Constraint Compliance")
        # Detect factory mode from data (factory has higher noise limit)
        is_factory = any(p.get("factory") for p in phases)
        noise_limit = 82.0 if is_factory else 80.0
        constraint_panel([
            {"name": "Fatigue Index", "current_value": latest_metrics["fatigue_index"], "limit": 0.4, "unit": ""},
            {"name": "Noise Level", "current_value": latest_metrics["noise_db"], "limit": noise_limit, "unit": "dB"},
        ])

    # ── Mode summary ───────────────────────────────────────────────
    mode_summary = data.get("mode_summary")
    if mode_summary:
        st.subheader("Working Mode Summary")
        st.markdown(f"**Current:** {mode_summary.get('current_mode', '?')}")
        st.markdown(f"**Modes used:** {', '.join(mode_summary.get('modes_used', []))}")
        for t in mode_summary.get("transitions", []):
            st.caption(
                f"{t['from_mode']} -> {t['to_mode']} "
                f"(phase {t['phase']}, {t['trigger']})"
            )


# ── Factory live-follow ──────────────────────────────────────────────

def _read_factory_live() -> dict | None:
    """Read the factory CLI live progress file."""
    if not _FACTORY_LIVE_FILE.exists():
        return None
    try:
        data = json.loads(_FACTORY_LIVE_FILE.read_text())
        return data if data.get("phases") else None
    except (json.JSONDecodeError, OSError):
        return None


def _render_factory_live(data: dict) -> None:
    """Render factory live monitoring with per-operator fatigue and factory noise."""
    state = data.get("state", "unknown")
    phases = data.get("phases", [])
    total = data.get("total_phases", 6)

    if state == "running":
        st.info(f"Factory experiment running — {len(phases)}/{total} phases done")
    elif state == "completed":
        st.success(f"Factory experiment completed — {len(phases)}/{total} phases")

    # Find latest metrics
    latest_fm = None
    for p in reversed(phases):
        if p.get("factory_metrics"):
            latest_fm = p["factory_metrics"]
            break

    if latest_fm is None:
        st.info("Waiting for first factory metrics...")
        return

    # ── KPI Gauges ──
    st.subheader("Factory KPI Gauges")
    gc1, gc2, gc3, gc4 = st.columns(4)
    with gc1:
        st.metric("Total Throughput", f"{latest_fm.get('total_throughput_uph', 0):.1f} u/h")
    with gc2:
        st.metric("Factory Noise", f"{latest_fm.get('factory_noise_db', 0):.1f} dB")
    with gc3:
        st.metric("Cell Balance Loss", f"{latest_fm.get('cell_balance_loss_pct', 0):.1f}%")
    with gc4:
        st.metric("Defect Rate", f"{latest_fm.get('factory_defect_rate', 0):.4f}")

    st.markdown("---")

    # ── Constraint Compliance: per-operator fatigue + noise ──
    st.subheader("Constraint Compliance")

    fatigue_limits = {"H1": 0.40, "H2": 0.35, "H3": 0.40}
    noise_limit = 82.0
    op_fatigue = latest_fm.get("operator_fatigue", {})

    # Build constraint items for factory
    factory_constraints = []
    for op_id, limit in fatigue_limits.items():
        current = op_fatigue.get(op_id, 0.0)
        factory_constraints.append({
            "name": f"Fatigue {op_id}",
            "current_value": current,
            "limit": limit,
            "unit": "",
        })
    factory_constraints.append({
        "name": "Factory Noise",
        "current_value": latest_fm.get("factory_noise_db", 0),
        "limit": noise_limit,
        "unit": "dB",
    })

    constraint_panel(factory_constraints)

    # Per-operator fatigue progress bars
    if op_fatigue:
        st.markdown("**Per-Operator Fatigue:**")
        op_cols = st.columns(len(op_fatigue))
        for col, (op_id, fat_val) in zip(op_cols, op_fatigue.items()):
            limit = fatigue_limits.get(op_id, 0.40)
            pct = min(fat_val / limit, 1.0) if limit > 0 else 0
            with col:
                color = "normal" if fat_val <= limit else "inverse"
                st.metric(
                    f"{op_id} (limit: {limit:.2f})",
                    f"{fat_val:.3f}",
                    delta=f"margin: {limit - fat_val:+.3f}",
                    delta_color=color,
                )
                st.progress(pct)

    st.markdown("---")

    # ── Per-cell throughput ──
    cell_metrics = latest_fm.get("cell_metrics", {})
    if cell_metrics:
        st.subheader("Per-Cell Metrics")
        cell_cols = st.columns(len(cell_metrics))
        for col, (cell_id, cm) in zip(cell_cols, cell_metrics.items()):
            with col:
                st.markdown(f"**Cell {cell_id}**")
                st.metric("Throughput", f"{cm.get('throughput_uph', 0):.1f} u/h")
                st.metric("Noise", f"{cm.get('noise_db', 0):.1f} dB")

    st.markdown("---")

    # ── Phase notes from latest ──
    for p in reversed(phases):
        if p.get("notes"):
            with st.expander(f"Phase {p['phase']}: {p.get('name', '')}"):
                for n in p["notes"]:
                    st.markdown(f"- {n}")
            break

    # ── Final metrics if completed ──
    final = data.get("final_metrics")
    if final:
        st.markdown("---")
        st.subheader("Final Factory Metrics")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.metric("Total Throughput", f"{final.get('total_throughput_uph', 0):.1f} u/h")
        with fc2:
            st.metric("Factory Energy", f"{final.get('factory_energy_kwh', 0):.1f} kWh")
        with fc3:
            st.metric("AGV Utilization", f"{final.get('agv_utilization', 0):.1%}")


# ── Existing monitor helpers ───────────────────────────────────────

def _build_simulated_stream() -> list[SimulationMetrics]:
    """Build a sequence of metrics snapshots from experiment results for demo."""
    result = st.session_state.get("experiment_result")
    if result is None:
        return []

    snapshots = []
    for name in ["S1", "S2", "S3"]:
        if name in result.schedule_metrics:
            snapshots.append(result.schedule_metrics[name])
    return snapshots


def _operator_alert(alert) -> str:
    """Translate a technical alert message to plain language."""
    msg = alert.message.lower()
    if "fatigue" in msg:
        if alert.severity == "violation":
            return "STOP — worker fatigue exceeds safe limit. Reduce robot speed or take a break."
        return "Reduce robot speed — operator fatigue approaching safe limit."
    if "noise" in msg:
        if alert.severity == "violation":
            return "STOP — noise level exceeds 80 dB limit. Check robot speeds."
        return "Noise is getting loud — consider slowing robots."
    return alert.message


def _operator_monitor_view(mon, latest) -> None:
    """Big status cards + actionable alerts for operators."""
    st.subheader("Current Status")

    # Six status cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Production Rate", f"{latest.throughput_uph:.1f} u/h")
    with col2:
        fi = latest.fatigue_index
        if fi <= 0.30:
            label = "OK"
        elif fi <= 0.40:
            label = "Caution"
        else:
            label = "HIGH"
        st.metric("Worker Fatigue", label)
    with col3:
        nd = latest.noise_db
        if nd <= 75.0:
            label = "OK"
        elif nd <= 80.0:
            label = "Caution"
        else:
            label = "TOO LOUD"
        st.metric("Noise", label)
    with col4:
        st.metric("Defect Rate", f"{latest.defect_rate:.4f}")
    with col5:
        st.metric("Deadline Gap", f"{latest.deadline_gap_pct:.1f} %")
    with col6:
        phase_num = latest.active_phase or 0
        step_name = display_phase(phase_num, dev_mode=False) if phase_num > 0 else "—"
        st.metric("Current Step", step_name)

    # Actionable alerts
    all_alerts = mon.get_alerts()
    if all_alerts:
        st.markdown("---")
        st.subheader("Alerts")
        for a in reversed(all_alerts[-10:]):
            plain = _operator_alert(a)
            if a.severity == "violation":
                st.error(plain)
            else:
                st.warning(plain)
    else:
        st.markdown("---")
        st.success("All readings are within safe limits.")

    # Safety Limits section
    st.markdown("---")
    st.subheader("Safety Limits")
    sl1, sl2 = st.columns(2)
    with sl1:
        fatigue_pct = min(latest.fatigue_index / 0.4, 1.0)
        st.markdown(f"**Fatigue:** {latest.fatigue_index:.2f} / 0.4")
        st.progress(fatigue_pct)
    with sl2:
        noise_pct = min(latest.noise_db / 80.0, 1.0)
        st.markdown(f"**Noise:** {latest.noise_db:.1f} / 80.0 dB")
        st.progress(noise_pct)


# ── Main render ────────────────────────────────────────────────────

def render() -> None:
    """Main render function for the Live Monitor page."""
    st.header("Live Monitor" if is_dev_mode() else "Live Status")

    # Auto-refresh option
    refresh_enabled = st.toggle("Auto-refresh (2s)", value=True, key="monitor_autorefresh")
    if refresh_enabled:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=2000, key="monitor_refresh")
        except ImportError:
            st.caption("Install `streamlit-autorefresh` for auto-refresh support.")

    # ── Check for factory live feed first ────────────────────────────
    factory_live = _read_factory_live()
    if factory_live is not None:
        st.caption(f"Reading factory live feed from `{_FACTORY_LIVE_FILE}`")
        _render_factory_live(factory_live)
        return

    # ── Check for single-cell live demo feed ─────────────────────────
    live_data = _read_live_demo()
    if live_data is not None:
        st.caption(f"Reading live feed from `{_LIVE_FILE}`")
        _render_live_demo(live_data)
        return

    # ── Fallback: existing monitor-based view ──────────────────────
    mon = get_monitoring_service()

    # Initialize monitoring with experiment data if available
    result = st.session_state.get("experiment_result")
    if result is not None and not mon.get_snapshots():
        # Set up contract
        contract = make_c1()
        for pr in result.phases:
            if pr.contract is not None:
                contract = pr.contract
        mon.set_contract(contract)

        # Push all schedule metrics as snapshots
        phase_map = {1: "S1", 3: "S2", 5: "S3"}
        for pr in result.phases:
            if pr.metrics is not None:
                mon.push_snapshot(pr.metrics, phase=pr.phase, layer="L4")

    latest = mon.get_latest_snapshot()

    if latest is None:
        st.info("No monitoring data. Run an experiment first, or push simulated data."
                if is_dev_mode()
                else "No data yet. Run an experiment from the Dashboard first.")

        if st.button("Load Demo Data"):
            contract = make_c1()
            mon.set_contract(contract)
            demo_metrics = [
                SimulationMetrics(throughput_uph=44.2, defect_rate=0.008, noise_db=76.5,
                                  fatigue_index=0.33, energy_kwh=398.0, deadline_gap_pct=15.8,
                                  units_produced=353, shift_hours=8.0),
                SimulationMetrics(throughput_uph=53.6, defect_rate=0.010, noise_db=85.0,
                                  fatigue_index=0.60, energy_kwh=462.0, deadline_gap_pct=0.0,
                                  units_produced=428, shift_hours=8.0),
                SimulationMetrics(throughput_uph=46.7, defect_rate=0.009, noise_db=79.1,
                                  fatigue_index=0.38, energy_kwh=417.0, deadline_gap_pct=14.9,
                                  units_produced=373, shift_hours=8.0),
            ]
            for i, m in enumerate(demo_metrics):
                mon.push_snapshot(m, phase=i+1)
            st.rerun()
        return

    # Operator mode: simplified view
    if not is_dev_mode():
        _operator_monitor_view(mon, latest)
        return

    # Top row: KPI gauges
    st.subheader("KPI Gauges")
    kpi_gauge_row({
        "throughput_uph": latest.throughput_uph,
        "defect_rate": latest.defect_rate,
        "noise_db": latest.noise_db,
        "fatigue_index": latest.fatigue_index,
        "energy_kwh": latest.energy_kwh,
        "deadline_gap_pct": latest.deadline_gap_pct,
    })

    st.markdown("---")

    # Middle row: Constraint compliance
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Constraint Compliance")
        constraint_panel([
            {"name": "Fatigue Index", "current_value": latest.fatigue_index, "limit": 0.4, "unit": ""},
            {"name": "Noise Level", "current_value": latest.noise_db, "limit": 80.0, "unit": "dB"},
        ])

    with col_right:
        st.subheader("Alerts")
        all_alerts = mon.get_alerts()
        alert_feed(all_alerts)

    st.markdown("---")

    # Bottom: Phase info + snapshot history
    bc1, bc2 = st.columns(2)
    with bc1:
        st.subheader("Current Status")
        st.markdown(f"**Active Phase:** {latest.active_phase}")
        st.markdown(f"**Active Layer:** {latest.active_layer or 'N/A'}")

    with bc2:
        st.subheader("Snapshot History")
        snapshots = mon.get_snapshots()
        if snapshots:
            import pandas as pd
            snap_data = [
                {
                    "Time": s.timestamp,
                    "Throughput": s.throughput_uph,
                    "Fatigue": s.fatigue_index,
                    "Noise": s.noise_db,
                    "Phase": s.active_phase,
                }
                for s in snapshots
            ]
            st.dataframe(pd.DataFrame(snap_data), use_container_width=True, hide_index=True)
