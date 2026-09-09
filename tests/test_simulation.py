"""Tests for SimPy simulation."""

import pytest
from cbpa.layer4_execution.simulation import ManufacturingCell
from cbpa.layer4_execution.monitor import ContractMonitor


class TestManufacturingCell:
    def test_simulation_produces_units(self, s1_schedule):
        cell = ManufacturingCell(seed=42)
        result = cell.run(s1_schedule)
        assert result.metrics.units_produced > 0
        assert result.metrics.throughput_uph > 0

    def test_simulation_noise_matches_model(self, s1_schedule):
        cell = ManufacturingCell(seed=42)
        result = cell.run(s1_schedule)
        # Noise is deterministic for given speeds
        assert result.metrics.noise_db == pytest.approx(76.6, abs=1.0)

    def test_simulation_energy_matches_model(self, s1_schedule):
        cell = ManufacturingCell(seed=42)
        result = cell.run(s1_schedule)
        assert result.metrics.energy_kwh == 398.0  # Exact from energy model

    def test_s2_higher_throughput_than_s1(self, s1_schedule, s2_schedule):
        cell = ManufacturingCell(seed=42)
        r1 = cell.run(s1_schedule)
        r2 = cell.run(s2_schedule)
        assert r2.metrics.throughput_uph > r1.metrics.throughput_uph

    def test_defect_rate_reasonable(self, s1_schedule):
        cell = ManufacturingCell(seed=42)
        result = cell.run(s1_schedule)
        assert 0.0 <= result.metrics.defect_rate <= 0.05


class TestContractMonitor:
    def test_no_alerts_for_s1(self, c1, s1_metrics):
        monitor = ContractMonitor(c1)
        alerts = monitor.check(s1_metrics)
        violations = [a for a in alerts if a.severity == "violation"]
        assert len(violations) == 0

    def test_violation_alerts_for_s2(self, c1, s2_metrics):
        monitor = ContractMonitor(c1)
        alerts = monitor.check(s2_metrics)
        violations = [a for a in alerts if a.severity == "violation"]
        assert len(violations) >= 2

    def test_deadline_alert(self, c1, s1_metrics):
        monitor = ContractMonitor(c1)
        alert = monitor.check_deadline(s1_metrics.throughput_uph, 62.4)
        assert alert is not None
        assert "miss" in alert.message.lower()
