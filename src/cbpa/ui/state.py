"""Streamlit session state initialization and helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from cbpa.service.experiment_service import ExperimentService, ExperimentStatus
    from cbpa.service.monitoring_service import MonitoringService
    from cbpa.service.integration.event_bus import EventBus


def init_session_state() -> None:
    """Initialize all session state keys on first run."""
    from cbpa.config.scenario import ScenarioConfig
    from cbpa.service.experiment_service import ExperimentService
    from cbpa.service.monitoring_service import MonitoringService
    from cbpa.service.integration.event_bus import EventBus
    from cbpa.service.integration.ros_bridge import ROSBridge
    from cbpa.service.integration.mes_bridge import MESBridge
    from cbpa.service.integration.basyx_bridge import BaSyxBridge
    from cbpa.service.integration.opcua_bridge import OPCUABridge
    from cbpa.service.integration.isaac_bridge import IsaacBridge

    if "initialized" in st.session_state:
        return

    # Configuration
    st.session_state.config = ScenarioConfig()

    # Experiment
    st.session_state.experiment_service = ExperimentService(
        config=st.session_state.config
    )
    st.session_state.experiment_result = None

    # Monitoring
    st.session_state.monitoring_service = MonitoringService()

    # Integration
    st.session_state.event_bus = EventBus()
    st.session_state.ros_bridge = ROSBridge(st.session_state.event_bus)
    st.session_state.mes_bridge = MESBridge(st.session_state.event_bus)
    st.session_state.basyx_bridge = BaSyxBridge(
        st.session_state.event_bus,
        registry_url="http://localhost:9082",
        aas_server_url="http://localhost:9081",
    )
    st.session_state.opcua_bridge = OPCUABridge(
        st.session_state.event_bus, endpoint="opc.tcp://localhost:4840/cbpa/"
    )
    st.session_state.isaac_bridge = IsaacBridge(
        st.session_state.event_bus, remote_url="http://localhost:8211"
    )

    # Factory experiment service
    from cbpa.config.scenario import FactoryScenarioConfig
    from cbpa.service.factory_experiment_service import FactoryExperimentService

    st.session_state.factory_config = FactoryScenarioConfig()
    st.session_state.factory_experiment_service = FactoryExperimentService(
        config=st.session_state.factory_config
    )
    st.session_state.factory_experiment_result = None

    # UI state
    st.session_state.use_llm = False
    st.session_state.dev_mode = False  # Operator mode by default
    st.session_state.factory_mode = False  # Single-cell by default

    # Operator three-mode dashboard state
    st.session_state.shift_contract_signed = False
    st.session_state.legislator_choices = {}
    st.session_state.operator_feedback = []

    st.session_state.initialized = True


def get_experiment_service() -> ExperimentService:
    from cbpa.service.experiment_service import ExperimentService
    return st.session_state.experiment_service  # type: ignore[return-value]


def get_factory_experiment_service():
    from cbpa.service.factory_experiment_service import FactoryExperimentService
    return st.session_state.factory_experiment_service


def is_factory_mode() -> bool:
    """Return True if the UI is in factory (multi-cell) mode."""
    return bool(st.session_state.get("factory_mode", False))


def get_monitoring_service() -> MonitoringService:
    from cbpa.service.monitoring_service import MonitoringService
    return st.session_state.monitoring_service  # type: ignore[return-value]


def get_event_bus() -> EventBus:
    from cbpa.service.integration.event_bus import EventBus
    return st.session_state.event_bus  # type: ignore[return-value]


def is_dev_mode() -> bool:
    """Return True if the UI is in developer mode."""
    return bool(st.session_state.get("dev_mode", False))
