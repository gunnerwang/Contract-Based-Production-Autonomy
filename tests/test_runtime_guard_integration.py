"""Polled runtime guards require actual observations at every checkpoint."""
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from cbpa.config.scenario import ScenarioConfig
from cbpa.layer3_verification.guard_synthesizer import (
    GuardEnforcer,
    RuntimeGuard,
    RuntimeGuardViolation,
)
from cbpa.models.schedule import Schedule
from cbpa.service.integration.orchestrator import IntegrationOrchestrator


def mock_orchestrator(samples):
    """Construct no bridges and make no connections."""
    orchestrator = IntegrationOrchestrator.__new__(IntegrationOrchestrator)
    orchestrator.config = ScenarioConfig()
    orchestrator.isaac = MagicMock()
    orchestrator.isaac.is_connected = True
    orchestrator.isaac.step_simulation.return_value = {"stub": False, "sim_time_s": 1}
    orchestrator.isaac.get_sensor_data.side_effect = samples
    orchestrator.opcua = MagicMock()
    orchestrator.basyx = MagicMock()
    for field, value in [
        ("_noise_model", 70),
        ("_fatigue_model", .2),
        ("_energy_model", 200),
        ("_defect_model", .01),
    ]:
        model = MagicMock()
        model.compute.return_value = value
        setattr(orchestrator, field, model)
    return orchestrator


def schedule():
    return Schedule(
        r1_speed_fraction=.7,
        r2_speed_fraction=.7,
        human_cycle_rate_multiplier=1,
        buffer_time_s=1,
        demand_target_uph=40,
    )


def observation(noise=70, fatigue=.2):
    return {"acoustic": {"noise_db": noise}, "human_operator": {"fatigue_estimate": fatigue}}


def test_missing_live_noise_cannot_be_replaced_with_analytical_estimate():
    samples = [{"human_operator": {"fatigue_estimate": .2}} for _ in range(20)]
    original = deepcopy(samples)
    orchestrator = mock_orchestrator(samples)
    enforcer = GuardEnforcer([RuntimeGuard("Noise", "noise_db", "<=", 80)])
    with pytest.raises(RuntimeGuardViolation) as caught:
        orchestrator.deploy_and_run_schedule(schedule(), guard_enforcer=enforcer)
    assert orchestrator.isaac.get_sensor_data.call_count == 1
    assert orchestrator.isaac.step_simulation.call_count == 1
    assert caught.value.events[0].metric_value is None
    assert not caught.value.events[0].enforced
    assert samples == original


def test_second_poll_violation_cannot_be_hidden_by_final_average():
    samples = [observation(v) for v in [70, 100] + [70] * 18]
    original = deepcopy(samples)
    orchestrator = mock_orchestrator(samples)
    enforcer = GuardEnforcer([RuntimeGuard("Noise", "noise_db", "<=", 80)])
    with pytest.raises(RuntimeGuardViolation) as caught:
        orchestrator.deploy_and_run_schedule(schedule(), guard_enforcer=enforcer)
    assert orchestrator.isaac.get_sensor_data.call_count == 2
    assert orchestrator.isaac.step_simulation.call_count == 2
    assert caught.value.events[0].metric_value == 100
    assert samples == original


def test_safe_observations_allow_all_polls_to_complete():
    orchestrator = mock_orchestrator([observation() for _ in range(20)])
    enforcer = GuardEnforcer([
        RuntimeGuard("Noise", "noise_db", "<=", 80),
        RuntimeGuard("FatigueIndex", "fatigue_index", "<=", .4),
    ])
    result = orchestrator.deploy_and_run_schedule(schedule(), guard_enforcer=enforcer)
    assert orchestrator.isaac.get_sensor_data.call_count == 20
    assert orchestrator.isaac.step_simulation.call_count == 20
    assert result.guard_events == []
    assert enforcer.enforcement_log == []


def test_missing_live_fatigue_cannot_be_replaced_with_analytical_estimate():
    orchestrator = mock_orchestrator([{"acoustic": {"noise_db": 70}}] * 20)
    enforcer = GuardEnforcer([RuntimeGuard("FatigueIndex", "fatigue_index", "<=", .4)])
    with pytest.raises(RuntimeGuardViolation) as caught:
        orchestrator.deploy_and_run_schedule(schedule(), guard_enforcer=enforcer)
    assert orchestrator.isaac.get_sensor_data.call_count == 1
    assert caught.value.events[0].metric_value is None


def test_rounding_cannot_turn_unequal_sensor_reading_into_equality():
    orchestrator = mock_orchestrator([observation(70.04)] * 20)
    enforcer = GuardEnforcer([RuntimeGuard("Noise", "noise_db", "==", 70)])
    with pytest.raises(RuntimeGuardViolation) as caught:
        orchestrator.deploy_and_run_schedule(schedule(), guard_enforcer=enforcer)
    assert orchestrator.isaac.get_sensor_data.call_count == 1
    assert caught.value.events[0].metric_value == 70.04


@pytest.mark.parametrize("name,field,limit", [
    ("Energy", "energy_kwh", 500),
    ("DefectRate", "defect_rate", .1),
    ("DeadlineGap", "deadline_gap_pct", 100),
    ("Throughput", "throughput_uph", 1000),
])
def test_unobserved_required_metric_cannot_pass_using_defaults_or_estimates(name, field, limit):
    orchestrator = mock_orchestrator([observation()] * 20)
    enforcer = GuardEnforcer([RuntimeGuard(name, field, "<=", limit)])
    with pytest.raises(RuntimeGuardViolation) as caught:
        orchestrator.deploy_and_run_schedule(schedule(), guard_enforcer=enforcer)
    assert orchestrator.isaac.get_sensor_data.call_count == 1
    assert caught.value.events[0].metric_value is None


@pytest.mark.parametrize("flag,value", [("stub", True), ("error", "sensor unavailable")])
def test_stub_or_error_payload_is_not_a_sensor_observation(flag, value):
    payload = observation()
    payload[flag] = value
    orchestrator = mock_orchestrator([payload] * 20)
    enforcer = GuardEnforcer([RuntimeGuard("Noise", "noise_db", "<=", 80)])
    with pytest.raises(RuntimeGuardViolation) as caught:
        orchestrator.deploy_and_run_schedule(schedule(), guard_enforcer=enforcer)
    assert orchestrator.isaac.get_sensor_data.call_count == 1
    assert caught.value.events[0].metric_value is None


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 'unavailable', True])
def test_invalid_observation_aborts_before_display_formatting(bad):
    orchestrator = mock_orchestrator([observation(bad)] * 20)
    enforcer = GuardEnforcer([RuntimeGuard('Noise', 'noise_db', '<=', 80)])
    with pytest.raises(RuntimeGuardViolation) as caught:
        orchestrator.deploy_and_run_schedule(schedule(), guard_enforcer=enforcer)
    assert orchestrator.isaac.get_sensor_data.call_count == 1
    assert caught.value.events[0].metric_value is None
    orchestrator.opcua.update_kpi.assert_not_called()
