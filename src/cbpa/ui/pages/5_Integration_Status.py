"""Page 5: Integration Status — all bridge connections + config."""

from __future__ import annotations

import streamlit as st

from cbpa.service.integration.event_bus import (
    TOPIC_AAS_CONTRACT_SYNC,
    TOPIC_AAS_KPI_SUBMODEL,
    TOPIC_ALERTS,
    TOPIC_CONSTRAINT_STATUS,
    TOPIC_ISAAC_SCENE_COMMAND,
    TOPIC_ISAAC_SENSOR_DATA,
    TOPIC_KPI_REPORT,
    TOPIC_KPI_STREAM,
    TOPIC_OPCUA_CONSTRAINT_UPDATE,
    TOPIC_OPCUA_KPI_UPDATE,
    TOPIC_PHASE_TRANSITION,
    TOPIC_PRODUCTION_ORDER,
    TOPIC_SCHEDULE_DEPLOY,
    TOPIC_SCHEDULE_UPLOAD,
)
from cbpa.ui.state import get_event_bus, get_factory_experiment_service, is_dev_mode, is_factory_mode


def _bridge_status_card(name: str, bridge: object, help_text: str) -> None:
    """Render a compact bridge status card."""
    connected = bridge.is_connected  # type: ignore[attr-defined]
    icon = "🟢" if connected else "🔴"
    st.markdown(f"**{name}**")
    st.markdown(f"{icon} {bridge.status}")  # type: ignore[attr-defined]
    if not connected:
        st.caption(help_text)


def _readiness_card(name: str, ready: bool, help_text: str) -> None:
    """Simple readiness indicator for factory orchestrator bridges."""
    icon = "🟢" if ready else "🔴"
    label = "Connected" if ready else "Stub / Offline"
    st.markdown(f"**{name}**")
    st.markdown(f"{icon} {label}")
    if not ready:
        st.caption(help_text)


def _render_factory_status() -> None:
    """Render factory-specific integration status."""
    st.subheader("Factory Orchestrator")

    fsvc = get_factory_experiment_service()
    exp = fsvc._experiment
    if exp is None or exp._orchestrator is None:
        st.info(
            "Factory orchestrator not initialized. "
            "Run the factory experiment with `--integrated` to activate."
        )
        # Still show the session-state bridges as fallback
        _render_single_cell_bridges()
        return

    orch = exp._orchestrator
    readiness = orch.check_readiness()

    st.subheader("Bridge Status (Factory)")

    col1, col2, col3 = st.columns(3)
    with col1:
        _readiness_card(
            "Isaac Sim (Factory)",
            readiness.get("isaac_sim", False),
            "Run `isaac_scene_setup.py --live-ui --multi-cell`",
        )
        st.markdown(f"Mode: `{orch.isaac.mode}`")
    with col2:
        _readiness_card(
            "Eclipse BaSyx (AAS)",
            readiness.get("basyx_aas", False),
            "Run `docker compose up` in docker/",
        )
        st.markdown(f"AAS ID: `{orch.basyx.aas_id}`")
    with col3:
        _readiness_card(
            "OPC-UA Server",
            readiness.get("opcua", False),
            "Install asyncua and provide an endpoint URL.",
        )

    # Factory-specific: show per-cell endpoints
    st.markdown("---")
    st.subheader("Factory Endpoints")
    st.markdown("""
| Endpoint | Purpose |
|---|---|
| `/scene/cellA/deploy` | Deploy Cell A schedule |
| `/scene/cellB/deploy` | Deploy Cell B schedule |
| `/scene/cellA/sensors` | Cell A sensor readings |
| `/scene/cellB/sensors` | Cell B sensor readings |
| `/scene/factory/sensors` | Aggregated factory sensors |
| `/scene/agv/status` | AGV corridor status |
    """)


def _render_single_cell_bridges() -> None:
    """Render the standard single-cell bridge status grid."""
    ros_bridge = st.session_state.ros_bridge
    mes_bridge = st.session_state.mes_bridge
    basyx_bridge = st.session_state.basyx_bridge
    opcua_bridge = st.session_state.opcua_bridge
    isaac_bridge = st.session_state.isaac_bridge

    st.subheader("Bridge Status")

    col1, col2, col3 = st.columns(3)
    with col1:
        _bridge_status_card(
            "ROS 2",
            ros_bridge,
            "Install ROS 2 + rclpy and source the workspace.",
        )
        st.markdown(f"Node: `{ros_bridge.node_name}`")

    with col2:
        _bridge_status_card(
            "MES (REST)",
            mes_bridge,
            "Provide a base URL and install httpx.",
        )

    with col3:
        _bridge_status_card(
            "Eclipse BaSyx (AAS)",
            basyx_bridge,
            "Run `docker compose up` in docker/ and wait for services to start.",
        )
        st.markdown(f"AAS ID: `{basyx_bridge.aas_id}`")

    col4, col5 = st.columns(2)
    with col4:
        _bridge_status_card(
            "OPC-UA Server",
            opcua_bridge,
            "Install asyncua and provide an endpoint URL.",
        )
        st.markdown(f"Endpoint: `{opcua_bridge.endpoint}`")

    with col5:
        _bridge_status_card(
            "NVIDIA Isaac Sim",
            isaac_bridge,
            "Run Isaac Sim (native or remote at a configured URL).",
        )
        st.markdown(f"Mode: `{isaac_bridge.mode}`")


def render() -> None:
    """Main render function for the Integration Status page."""
    if not is_dev_mode():
        st.info("Integration Status is available in **Developer Mode**. "
                "Enable it from the sidebar toggle.")
        return

    st.header("Integration Status")

    event_bus = get_event_bus()

    if is_factory_mode():
        _render_factory_status()
    else:
        _render_single_cell_bridges()

    st.markdown("---")

    # ── Connection parameters ─────────────────────────────────────
    mes_bridge = st.session_state.mes_bridge
    basyx_bridge = st.session_state.basyx_bridge
    opcua_bridge = st.session_state.opcua_bridge
    isaac_bridge = st.session_state.isaac_bridge

    st.subheader("Connection Parameters")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        ros_uri = st.text_input(
            "ROS Master URI",
            value="localhost:11311",
            key="ros_uri",
            disabled=True,
        )
        st.caption("Set via ROS_MASTER_URI environment variable")

    with col_b:
        mes_url = st.text_input(
            "MES Base URL",
            value=mes_bridge.base_url or "(not configured)",
            key="mes_url",
        )
        if st.button("Update MES URL"):
            from cbpa.service.integration.mes_bridge import MESBridge
            st.session_state.mes_bridge = MESBridge(event_bus, base_url=mes_url)
            st.rerun()

    with col_c:
        basyx_url = st.text_input(
            "BaSyx Registry URL",
            value=basyx_bridge.registry_url or "(not configured)",
            key="basyx_url",
        )
        if st.button("Update BaSyx URL"):
            from cbpa.service.integration.basyx_bridge import BaSyxBridge
            st.session_state.basyx_bridge = BaSyxBridge(event_bus, registry_url=basyx_url)
            st.rerun()

    col_d, col_e = st.columns(2)
    with col_d:
        opcua_ep = st.text_input(
            "OPC-UA Endpoint",
            value=opcua_bridge.endpoint,
            key="opcua_endpoint",
        )
        if st.button("Update OPC-UA Endpoint"):
            from cbpa.service.integration.opcua_bridge import OPCUABridge
            st.session_state.opcua_bridge = OPCUABridge(event_bus, endpoint=opcua_ep)
            st.rerun()

    with col_e:
        isaac_url = st.text_input(
            "Isaac Sim Remote URL",
            value=isaac_bridge.remote_url or "(not configured)",
            key="isaac_url",
        )
        if st.button("Update Isaac URL"):
            from cbpa.service.integration.isaac_bridge import IsaacBridge
            st.session_state.isaac_bridge = IsaacBridge(event_bus, remote_url=isaac_url)
            st.rerun()

    st.markdown("---")

    # ── Architecture diagram ──────────────────────────────────────
    with st.expander("Integration Architecture"):
        if is_factory_mode():
            st.markdown("""
**Factory layer mapping** (CBPA lifecycle → integration):

| CBPA Layer | Single-Cell Target | Factory Extension |
|---|---|---|
| L1: Contract Elicitation | Eclipse BaSyx AAS | Factory contract with per-operator constraints |
| L2: Planning | Pareto candidate generation | Per-cell schedule variants + variant routing |
| L3: Verification | Isaac Sim physics (1 cell) | Isaac Sim multi-cell (`/scene/cellA`, `/scene/cellB`) |
| L4: Execution | SimPy + OPC-UA + Guards | `FactoryOrchestrator` deploys to both cells + AGV |
| L5: Adaptation | Learning store + micro/meso | Per-cell adaptation + cross-cell redistribution |
| Meta: Governance | 3-option escalation | 5-option factory escalation (operator reassignment) |
            """)
        else:
            st.markdown("""
**Layer mapping** (paper Section 3 → integration):

| CBPA Layer | Integration Target | Bridge |
|---|---|---|
| L1: Contract Definition | Eclipse BaSyx AAS | `basyx_bridge` |
| L2: Planning | Python optimizer + SimPy | (built-in) |
| L3: Verification | NVIDIA Isaac Sim physics | `isaac_bridge` |
| L4: Execution | ROS 2 + OPC-UA to PLC | `ros_bridge` + `opcua_bridge` |
| L5: Adaptation | Monitoring → EventBus | `event_bus` |
| Meta: Governance | Streamlit dashboard | (built-in) |
| MES Integration | REST API | `mes_bridge` |
            """)

    st.markdown("---")

    # ── Topic/endpoint mapping ────────────────────────────────────
    st.subheader("Topic / Endpoint Mapping")

    import pandas as pd

    topics = [
        {"Topic": TOPIC_KPI_STREAM, "Direction": "Publish", "Target": "ROS + MES", "Description": "Real-time KPI readings"},
        {"Topic": TOPIC_SCHEDULE_DEPLOY, "Direction": "Publish", "Target": "ROS + Isaac", "Description": "Deploy schedule to cell controller"},
        {"Topic": TOPIC_ALERTS, "Direction": "Publish", "Target": "ROS + MES", "Description": "Constraint violation alerts"},
        {"Topic": TOPIC_PHASE_TRANSITION, "Direction": "Publish", "Target": "ROS", "Description": "Phase change notifications"},
        {"Topic": TOPIC_CONSTRAINT_STATUS, "Direction": "Publish", "Target": "ROS + OPC-UA", "Description": "Constraint compliance snapshots"},
        {"Topic": TOPIC_PRODUCTION_ORDER, "Direction": "Subscribe", "Target": "MES", "Description": "Incoming production orders"},
        {"Topic": TOPIC_SCHEDULE_UPLOAD, "Direction": "Subscribe", "Target": "MES", "Description": "Schedule upload acknowledgements"},
        {"Topic": TOPIC_KPI_REPORT, "Direction": "Publish", "Target": "MES", "Description": "Aggregated shift KPI reports"},
        {"Topic": TOPIC_AAS_CONTRACT_SYNC, "Direction": "Publish", "Target": "BaSyx", "Description": "Contract → AAS submodel sync"},
        {"Topic": TOPIC_AAS_KPI_SUBMODEL, "Direction": "Publish", "Target": "BaSyx", "Description": "Live KPI → AAS submodel update"},
        {"Topic": TOPIC_OPCUA_KPI_UPDATE, "Direction": "Publish", "Target": "OPC-UA", "Description": "KPI values to OPC-UA nodes"},
        {"Topic": TOPIC_OPCUA_CONSTRAINT_UPDATE, "Direction": "Publish", "Target": "OPC-UA", "Description": "Constraint limits to OPC-UA nodes"},
        {"Topic": TOPIC_ISAAC_SCENE_COMMAND, "Direction": "Publish", "Target": "Isaac Sim", "Description": "Scene deploy/step/verify commands"},
        {"Topic": TOPIC_ISAAC_SENSOR_DATA, "Direction": "Subscribe", "Target": "Isaac Sim", "Description": "Sensor readings from simulation"},
    ]

    st.dataframe(pd.DataFrame(topics), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Message log ───────────────────────────────────────────────
    st.subheader("Event Bus Message Log")
    limit = st.slider("Messages to show", 10, 100, 50, key="bus_log_limit")
    history = event_bus.get_history(limit=limit)

    if not history:
        st.info("No messages on the event bus yet.")
    else:
        log_data = [
            {
                "Timestamp": msg.timestamp[:19],
                "Topic": msg.topic,
                "Source": msg.source,
                "Payload": str(msg.payload)[:100],
            }
            for msg in reversed(history)
        ]
        st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
