"""CBPA Case Study — Streamlit application entrypoint.

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path so ``cbpa`` package is importable
_src = str(Path(__file__).resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import streamlit as st
from cbpa.ui.state import init_session_state, is_dev_mode, is_factory_mode
from cbpa.ui.theme import PHASE_LABELS, PHASE_DISPLAY_NAMES

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="CBPA Case Study",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────
init_session_state()

# Auto-activate factory mode from query param (?factory=1)
_qp = st.query_params
if _qp.get("factory") == "1" and not st.session_state.factory_mode:
    st.session_state.factory_mode = True

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("CBPA Case Study")
    st.caption("Contract-Based Production Autonomy")

    st.divider()

    # Mode toggles
    factory_mode = st.toggle("Factory Mode (2-cell)", value=st.session_state.factory_mode)
    if factory_mode != st.session_state.factory_mode:
        st.session_state.factory_mode = factory_mode
        st.rerun()

    dev_mode = st.toggle("Developer Mode", value=st.session_state.dev_mode)
    if dev_mode != st.session_state.dev_mode:
        st.session_state.dev_mode = dev_mode
        st.rerun()

    # LLM toggle only visible in dev mode
    if is_dev_mode():
        use_llm = st.toggle("LLM Mode", value=st.session_state.use_llm)
        if use_llm != st.session_state.use_llm:
            st.session_state.use_llm = use_llm

    st.divider()

    # System status — use the right service based on mode
    if is_factory_mode():
        exp_svc = st.session_state.factory_experiment_service
    else:
        exp_svc = st.session_state.experiment_service
    status = exp_svc.status

    if is_dev_mode():
        st.subheader("System Status")
        st.markdown(f"**Experiment:** {status.state.value}")
        if status.current_phase > 0:
            label = PHASE_LABELS.get(status.current_phase, "")
            st.markdown(f"**Phase:** {status.current_phase} — {label}")
        # Integration status (compact)
        ros_status = st.session_state.ros_bridge.status
        mes_status = st.session_state.mes_bridge.status
        st.markdown(f"**ROS 2:** {ros_status}")
        st.markdown(f"**MES:** {mes_status}")
    else:
        # Operator-friendly status
        state_val = status.state.value
        _status_map = {
            "idle": ("🟢", "Ready"),
            "running": ("🔵", "Running"),
            "awaiting_escalation": ("🟠", "Needs Your Input"),
            "completed": ("✅", "Done"),
            "error": ("🔴", "Error"),
        }
        icon, label = _status_map.get(state_val, ("⚪", state_val))
        st.markdown(f"### {icon} {label}")
        if status.current_phase > 0:
            phase_label = PHASE_DISPLAY_NAMES.get(status.current_phase, f"Step {status.current_phase}")
            st.caption(f"Current step: {phase_label}")
        if st.session_state.get("shift_contract_signed"):
            st.caption("Contract: Approved")
        else:
            st.caption("Contract: Not yet defined")

# ── Check for live demo feed ─────────────────────────────────────────
_live_demo_active = Path("data/results/live_demo.json").exists()

# Update sidebar status when live demo is active
if _live_demo_active:
    import json as _json
    try:
        _live = _json.loads(Path("data/results/live_demo.json").read_text())
        _n_phases = len(_live.get("phases", []))
        _live_state = _live.get("state", "unknown")
        with st.sidebar:
            if _live_state == "running":
                st.info(f"Live demo: {_n_phases} phases completed")
            elif _live_state == "completed":
                st.success(f"Live demo: {_n_phases} phases completed")
    except Exception:
        pass

# ── Navigation ────────────────────────────────────────────────────────
if is_factory_mode():
    pages = {
        "Factory Dashboard": "cbpa.ui.pages.6_Factory_Dashboard",
        "Live Monitor": "cbpa.ui.pages.3_Live_Monitor",
    }
    if is_dev_mode():
        pages["Audit Trail"] = "cbpa.ui.pages.4_Audit_Trail"
        pages["Integration Status"] = "cbpa.ui.pages.5_Integration_Status"
elif is_dev_mode():
    pages = {
        "Experiment Dashboard": "cbpa.ui.pages.1_Experiment_Dashboard",
        "Results Explorer": "cbpa.ui.pages.2_Results_Explorer",
        "Live Monitor": "cbpa.ui.pages.3_Live_Monitor",
        "Audit Trail": "cbpa.ui.pages.4_Audit_Trail",
        "Integration Status": "cbpa.ui.pages.5_Integration_Status",
    }
else:
    pages = {
        "Shift Dashboard": "cbpa.ui.pages.1_Experiment_Dashboard",
        "Detailed Results": "cbpa.ui.pages.2_Results_Explorer",
    }
    # Always show Live Monitor when a live demo feed is available
    if _live_demo_active:
        pages["Live Monitor"] = "cbpa.ui.pages.3_Live_Monitor"

# Default to Live Monitor when live demo is active
_default_page = list(pages.keys()).index("Live Monitor") if _live_demo_active and "Live Monitor" in pages else 0

selection = st.sidebar.radio("Navigate", list(pages.keys()), index=_default_page, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption("Built with Streamlit + CBPA Simulation Engine")

# ── Render selected page ──────────────────────────────────────────────
import importlib

module_name = pages[selection]  # type: ignore[index]
page_module = importlib.import_module(module_name)
page_module.render()
