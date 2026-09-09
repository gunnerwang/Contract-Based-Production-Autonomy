"""Tests for physics models and cell evaluator."""

import pytest
from cbpa.config.defaults import TABLE3_TARGETS
from cbpa.models.schedule import Schedule
from cbpa.physics.cell_evaluator import CellEvaluator
from cbpa.physics.defect_model import DefectModel
from cbpa.physics.energy_model import EnergyModel
from cbpa.physics.fatigue_model import FatigueModel
from cbpa.physics.noise_model import NoiseModel


class TestFatigueModel:
    def test_higher_rate_increases_fatigue(self):
        model = FatigueModel()
        f_low = model.compute(0.65, 0.60, 1.0)
        f_high = model.compute(0.65, 0.60, 1.15)
        assert f_high > f_low

    def test_fatigue_bounded(self):
        model = FatigueModel()
        f = model.compute(1.0, 1.0, 2.0)
        assert f <= 1.0

    def test_zero_shift_zero_fatigue(self):
        model = FatigueModel()
        f = model.compute(0.5, 0.5, 1.0, shift_fraction=0.0)
        assert f == 0.0


class TestNoiseModel:
    def test_higher_speed_more_noise(self):
        model = NoiseModel()
        n_low = model.compute(0.5, 0.5)
        n_high = model.compute(0.9, 0.9)
        assert n_high > n_low

    def test_noise_positive(self):
        model = NoiseModel()
        n = model.compute(0.1, 0.1)
        assert n > 0


class TestEnergyModel:
    def test_energy_increases_with_speed(self):
        model = EnergyModel()
        e_low = model.compute(0.5, 0.5, 1.0, 8.0)
        e_high = model.compute(0.9, 0.9, 1.0, 8.0)
        assert e_high > e_low

    def test_energy_exact_s1(self):
        model = EnergyModel()
        e = model.compute(0.65, 0.60, 1.0, 8.0)
        assert e == 398.0

    def test_energy_exact_s2(self):
        model = EnergyModel()
        e = model.compute(0.90, 1.0, 1.15, 8.0)
        assert e == 462.0

    def test_energy_exact_s3(self):
        model = EnergyModel()
        e = model.compute(0.72, 0.70, 1.05, 8.0)
        assert e == 417.0


class TestDefectModel:
    def test_higher_speed_more_defects(self):
        model = DefectModel()
        d_low = model.compute(0.4, 0.2)
        d_high = model.compute(0.9, 0.2)
        assert d_high > d_low

    def test_higher_fatigue_more_defects(self):
        model = DefectModel()
        d_low = model.compute(0.5, 0.1)
        d_high = model.compute(0.5, 0.8)
        assert d_high > d_low


class TestCellEvaluator:
    @pytest.mark.parametrize("name", ["S1", "S2", "S3"])
    def test_deterministic_matches_table3(self, name, config):
        evaluator = CellEvaluator(config.cell, mode="deterministic")
        p = config.schedules
        params = {
            "S1": (p.s1_r1_speed, p.s1_r2_speed, p.s1_human_rate, p.s1_buffer_s, config.cell.demand_base_uph),
            "S2": (p.s2_r1_speed, p.s2_r2_speed, p.s2_human_rate, p.s2_buffer_s, config.cell.demand_base_uph * 1.2),
            "S3": (p.s3_r1_speed, p.s3_r2_speed, p.s3_human_rate, p.s3_buffer_s, config.cell.demand_base_uph * 1.2),
        }
        r1, r2, hr, buf, demand = params[name]
        schedule = Schedule(
            name=name, r1_speed_fraction=r1, r2_speed_fraction=r2,
            human_cycle_rate_multiplier=hr, buffer_time_s=buf,
            demand_target_uph=demand,
        )
        metrics = evaluator.evaluate(schedule)
        targets = TABLE3_TARGETS[name]

        assert metrics.throughput_uph == targets["throughput_uph"]
        assert metrics.defect_rate == targets["defect_rate"]
        assert metrics.noise_db == targets["noise_db"]
        assert metrics.fatigue_index == targets["fatigue_index"]
        assert metrics.energy_kwh == targets["energy_kwh"]
        assert metrics.deadline_gap_pct == targets["deadline_gap_pct"]
