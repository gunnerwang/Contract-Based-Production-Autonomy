"""Shared test fixtures."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cbpa.config.scenario import ScenarioConfig
from cbpa.models.contract import OutcomeContract, make_c1, make_c2
from cbpa.models.metrics import SimulationMetrics
from cbpa.models.schedule import Schedule


@pytest.fixture
def config():
    return ScenarioConfig()


@pytest.fixture
def c1():
    return make_c1()


@pytest.fixture
def c2():
    return make_c2()


@pytest.fixture
def s1_schedule(config):
    p = config.schedules
    return Schedule(
        name="S1",
        r1_speed_fraction=p.s1_r1_speed,
        r2_speed_fraction=p.s1_r2_speed,
        human_cycle_rate_multiplier=p.s1_human_rate,
        buffer_time_s=p.s1_buffer_s,
        demand_target_uph=config.cell.demand_base_uph,
    )


@pytest.fixture
def s2_schedule(config):
    p = config.schedules
    return Schedule(
        name="S2",
        r1_speed_fraction=p.s2_r1_speed,
        r2_speed_fraction=p.s2_r2_speed,
        human_cycle_rate_multiplier=p.s2_human_rate,
        buffer_time_s=p.s2_buffer_s,
        demand_target_uph=config.cell.demand_base_uph * 1.2,
    )


@pytest.fixture
def s3_schedule(config):
    p = config.schedules
    return Schedule(
        name="S3",
        r1_speed_fraction=p.s3_r1_speed,
        r2_speed_fraction=p.s3_r2_speed,
        human_cycle_rate_multiplier=p.s3_human_rate,
        buffer_time_s=p.s3_buffer_s,
        demand_target_uph=config.cell.demand_base_uph * 1.2,
    )


@pytest.fixture
def s1_metrics():
    return SimulationMetrics(
        throughput_uph=44.2, defect_rate=0.008, noise_db=76.5,
        fatigue_index=0.33, energy_kwh=398.0, deadline_gap_pct=15.8,
        units_produced=353, shift_hours=8.0,
    )


@pytest.fixture
def s2_metrics():
    return SimulationMetrics(
        throughput_uph=53.6, defect_rate=0.010, noise_db=85.0,
        fatigue_index=0.60, energy_kwh=462.0, deadline_gap_pct=0.0,
        units_produced=428, shift_hours=8.0,
    )


@pytest.fixture
def s3_metrics():
    return SimulationMetrics(
        throughput_uph=46.7, defect_rate=0.009, noise_db=79.1,
        fatigue_index=0.38, energy_kwh=417.0, deadline_gap_pct=14.9,
        units_produced=373, shift_hours=8.0,
    )
