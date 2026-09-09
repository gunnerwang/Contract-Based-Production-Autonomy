"""Lifecycle regression: preserve diagnostic metrics and refuse uncertified S3."""

import pytest
from cbpa.config.defaults import TABLE3_TARGETS
from cbpa.runner.experiment import CBPAExperiment
from cbpa.runner.lifecycle import DeploymentBlocked


class TestExperimentE2E:
    @pytest.fixture
    def result(self):
        experiment = CBPAExperiment(use_llm=False)
        completed = []
        with pytest.raises(DeploymentBlocked, match="Phase 5"):
            experiment.run_all_phases(on_phase_complete=completed.append)
        assert experiment.learning.total_records == 0
        assert not any(e.action == "schedule_deployed" and e.phase == 5
                       for e in experiment.audit.entries)
        return experiment.build_result(completed)

    def test_four_phases_complete_before_rejection(self, result):
        assert len(result.phases) == 4

    def test_phase1_s1_deployed(self, result):
        p1 = result.phases[0]
        assert p1.phase == 1
        assert p1.schedule is not None
        assert p1.schedule.name == "S1"
        assert p1.feasibility is not None
        assert p1.feasibility.is_feasible

    def test_phase2_shortfall_detected(self, result):
        p2 = result.phases[1]
        assert p2.phase == 2
        assert "Shortfall" in p2.description

    def test_phase3_s2_rejected(self, result):
        p3 = result.phases[2]
        assert p3.phase == 3
        assert p3.feasibility is not None
        assert not p3.feasibility.is_feasible
        assert len(p3.feasibility.violations) == 2

    def test_phase4_option_c_selected(self, result):
        p4 = result.phases[3]
        assert p4.phase == 4
        assert p4.decision is not None
        assert p4.decision.selected_option == "Option C"

    def test_phase5_has_no_accepted_schedule(self, result):
        assert "S3" not in result.schedule_metrics
        assert result.audit.entries[-1].action == "deployment_blocked"

    @pytest.mark.parametrize("name", ["S1", "S2"])
    def test_table3_throughput(self, result, name):
        assert result.schedule_metrics[name].throughput_uph == TABLE3_TARGETS[name]["throughput_uph"]

    @pytest.mark.parametrize("name", ["S1", "S2"])
    def test_table3_defect_rate(self, result, name):
        assert result.schedule_metrics[name].defect_rate == TABLE3_TARGETS[name]["defect_rate"]

    @pytest.mark.parametrize("name", ["S1", "S2"])
    def test_table3_noise(self, result, name):
        assert result.schedule_metrics[name].noise_db == TABLE3_TARGETS[name]["noise_db"]

    @pytest.mark.parametrize("name", ["S1", "S2"])
    def test_table3_fatigue(self, result, name):
        assert result.schedule_metrics[name].fatigue_index == TABLE3_TARGETS[name]["fatigue_index"]

    @pytest.mark.parametrize("name", ["S1", "S2"])
    def test_table3_energy(self, result, name):
        assert result.schedule_metrics[name].energy_kwh == TABLE3_TARGETS[name]["energy_kwh"]

    @pytest.mark.parametrize("name", ["S1", "S2"])
    def test_table3_deadline_gap(self, result, name):
        assert result.schedule_metrics[name].deadline_gap_pct == TABLE3_TARGETS[name]["deadline_gap_pct"]

    def test_vr_ordering(self, result):
        """S2 should have highest V/R despite being infeasible."""
        vr1 = result.schedule_vr["S1"].vr_score
        vr2 = result.schedule_vr["S2"].vr_score
        assert vr2 > vr1

    def test_s1_feasible_s2_not(self, result):
        assert result.schedule_feasibility["S1"].is_feasible
        assert not result.schedule_feasibility["S2"].is_feasible

    def test_audit_trail_has_entries(self, result):
        assert len(result.audit.entries) > 0
        phases = {e.phase for e in result.audit.entries}
        assert phases == {1, 2, 3, 4, 5}
