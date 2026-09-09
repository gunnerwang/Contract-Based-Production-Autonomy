"""Tests for Pydantic data models."""

import pytest
from cbpa.models.contract import (
    HardConstraint,
    KPIDirection,
    KPITarget,
    OutcomeContract,
    PriorityLevel,
    make_c1,
    make_c2,
)
from cbpa.models.escalation import AuditEntry, EscalationQuery, RemediationOption
from cbpa.models.metrics import FeasibilityReport, SimulationMetrics, VRScoreResult
from cbpa.models.schedule import Schedule


class TestOutcomeContract:
    def test_make_c1_has_required_fields(self):
        c1 = make_c1()
        assert c1.name == "C1"
        assert len(c1.kpi_targets) >= 1
        assert len(c1.hard_constraints) >= 2
        assert len(c1.priority_order) >= 1

    def test_c1_has_fatigue_constraint(self):
        c1 = make_c1()
        fc = c1.get_constraint("FatigueIndex")
        assert fc is not None
        assert fc.limit == 0.4

    def test_c1_has_noise_constraint(self):
        c1 = make_c1()
        nc = c1.get_constraint("Noise")
        assert nc is not None
        assert nc.limit == 80.0

    def test_c1_priority_ordering(self):
        c1 = make_c1()
        assert c1.priority_order[0] == PriorityLevel.HUMAN_WELLBEING
        assert c1.priority_order[-1] == PriorityLevel.THROUGHPUT

    def test_make_c2_preserves_constraints(self):
        c2 = make_c2()
        assert c2.name == "C2"
        fc = c2.get_constraint("FatigueIndex")
        assert fc is not None
        assert fc.limit == 0.4  # Same as C1

    def test_c2_acknowledges_demand(self):
        c2 = make_c2()
        assert c2.context.get("demand_increase_pct") == 20
        assert c2.context.get("deadline_gap_acknowledged") is True


class TestSchedule:
    def test_schedule_creation(self):
        s = Schedule(
            name="test", r1_speed_fraction=0.5, r2_speed_fraction=0.5,
            human_cycle_rate_multiplier=1.0, buffer_time_s=2.0,
            demand_target_uph=50.0,
        )
        assert s.name == "test"
        assert not s.is_aggressive

    def test_aggressive_detection(self):
        s = Schedule(
            name="fast", r1_speed_fraction=0.9, r2_speed_fraction=0.95,
            human_cycle_rate_multiplier=1.2, buffer_time_s=1.0,
            demand_target_uph=60.0,
        )
        assert s.is_aggressive

    def test_speed_bounds(self):
        with pytest.raises(Exception):
            Schedule(
                name="bad", r1_speed_fraction=1.5, r2_speed_fraction=0.5,
                human_cycle_rate_multiplier=1.0, buffer_time_s=2.0,
                demand_target_uph=50.0,
            )


class TestMetrics:
    def test_simulation_metrics(self, s1_metrics):
        assert s1_metrics.throughput_uph == 44.2
        assert s1_metrics.defect_rate == 0.008

    def test_feasibility_report(self):
        report = FeasibilityReport(is_feasible=True, violations=[], constraint_margins={"FatigueIndex": 0.07})
        assert report.is_feasible

    def test_vr_score_result(self):
        vr = VRScoreResult(value_numerator=0.8, resource_denominator=1.0, vr_score=0.8)
        assert vr.vr_score == 0.8


class TestEscalation:
    def test_remediation_option(self):
        opt = RemediationOption(
            label="Option C",
            description="Keep constraints",
            preserves_human_constraints=True,
        )
        assert opt.preserves_human_constraints

    def test_audit_entry(self):
        entry = AuditEntry(phase=1, layer="L1", action="test", contract_state="C1")
        assert entry.phase == 1
