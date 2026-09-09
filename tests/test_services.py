"""Tests for the service layer: ExperimentService, ConfigService, MonitoringService, ExportService."""

import json
import zipfile
import io

import pytest

from cbpa.config.scenario import ScenarioConfig
from cbpa.models.contract import make_c1
from cbpa.models.escalation import ManagerDecision
from cbpa.models.metrics import SimulationMetrics
from cbpa.service.config_service import ConfigService
from cbpa.service.experiment_service import ExperimentService, ExperimentState
from cbpa.service.export_service import ExportService
from cbpa.service.monitoring_service import MonitoringService
from cbpa.service.integration.event_bus import EventBus, TOPIC_KPI_STREAM


# ═══════════════════════════════════════════════════════════════════════
#  ConfigService
# ═══════════════════════════════════════════════════════════════════════

class TestConfigService:
    def test_create_default(self):
        config = ConfigService.create_default()
        assert isinstance(config, ScenarioConfig)
        assert config.cell.demand_base_uph == 52.0

    def test_to_flat_dict_roundtrip(self):
        config = ConfigService.create_default()
        flat = ConfigService.to_flat_dict(config)
        assert flat["cell.shift_hours"] == 8.0
        assert flat["demand_spike_pct"] == 20.0
        assert flat["schedules.s1_r1_speed"] == 0.65

    def test_from_dict(self):
        config = ConfigService.from_dict({
            "cell.shift_hours": 10.0,
            "schedules.s1_r1_speed": 0.70,
            "demand_spike_pct": 25.0,
        })
        assert config.cell.shift_hours == 10.0
        assert config.schedules.s1_r1_speed == 0.70
        assert config.demand_spike_pct == 25.0

    def test_validate_weights_ok(self):
        errors = ConfigService.validate_weights([0.5, 0.3, 0.2])
        assert errors == []

    def test_validate_weights_bad_sum(self):
        errors = ConfigService.validate_weights([0.5, 0.5, 0.5])
        assert len(errors) == 1
        assert "1.0" in errors[0]

    def test_validate_weights_negative(self):
        errors = ConfigService.validate_weights([-0.1, 0.6, 0.5])
        assert any("negative" in e or ">= 0" in e for e in errors)


# ═══════════════════════════════════════════════════════════════════════
#  ExperimentService
# ═══════════════════════════════════════════════════════════════════════

class TestExperimentService:
    def test_initial_state(self):
        svc = ExperimentService()
        assert svc.status.state == ExperimentState.IDLE
        assert svc.status.current_phase == 0
        assert svc.status.completed_phases == []

    def test_run_phase1(self):
        svc = ExperimentService()
        pr = svc.run_phase(1)
        assert pr is not None
        assert pr.phase == 1
        assert pr.schedule is not None
        assert pr.schedule.name == "S1"
        assert pr.metrics is not None
        assert pr.feasibility is not None
        assert pr.feasibility.is_feasible

    def test_run_phases_1_through_3(self):
        svc = ExperimentService()
        svc.run_phase(1)
        svc.run_phase(2)
        pr3 = svc.run_phase(3)
        assert pr3 is not None
        assert pr3.phase == 3
        assert pr3.feasibility is not None
        assert not pr3.feasibility.is_feasible  # S2 rejected

    def test_phase4_escalation_handshake(self):
        svc = ExperimentService()
        svc.run_phase(1)
        svc.run_phase(2)
        svc.run_phase(3)

        # Phase 4 without decision -> awaiting escalation
        result = svc.run_phase(4)
        assert result is None
        assert svc.status.state == ExperimentState.AWAITING_ESCALATION

        # Get escalation query
        query = svc.get_escalation_query()
        assert query is not None
        assert len(query.options) == 3

        # Resolve with Option C
        decision = ManagerDecision(
            selected_option="Option C",
            rationale="Keep human constraints",
        )
        pr4 = svc.resolve_escalation(decision)
        assert pr4 is not None
        assert pr4.phase == 4
        assert pr4.decision.selected_option == "Option C"

    def test_run_all_phases(self):
        svc = ExperimentService()
        result = svc.run_all()
        assert result is not None
        assert len(result.phases) == 4
        assert svc.status.state == ExperimentState.ERROR
        assert "S1" in result.schedule_metrics
        assert "S2" in result.schedule_metrics
        assert "S3" not in result.schedule_metrics
        assert "Phase 5" in svc.status.error
        assert result.audit.entries[-1].action == "deployment_blocked"
        assert svc.run_phase(5) is None

    def test_reset(self):
        svc = ExperimentService()
        svc.run_all()
        assert svc.status.state == ExperimentState.ERROR

        svc.reset()
        assert svc.status.state == ExperimentState.IDLE
        assert svc.status.completed_phases == []

    def test_run_phase4_with_inline_decision(self):
        svc = ExperimentService()
        svc.run_phase(1)
        svc.run_phase(2)
        svc.run_phase(3)

        decision = ManagerDecision(
            selected_option="Option A",
            rationale="Relax fatigue",
        )
        pr4 = svc.run_phase(4, manager_decision=decision)
        assert pr4 is not None
        assert pr4.decision.selected_option == "Option A"


# ═══════════════════════════════════════════════════════════════════════
#  MonitoringService
# ═══════════════════════════════════════════════════════════════════════

class TestMonitoringService:
    def test_push_snapshot(self):
        c1 = make_c1()
        mon = MonitoringService(contract=c1)
        metrics = SimulationMetrics(
            throughput_uph=44.2, defect_rate=0.008, noise_db=76.5,
            fatigue_index=0.33, energy_kwh=398.0, deadline_gap_pct=15.8,
        )
        alerts = mon.push_snapshot(metrics, phase=1)
        assert isinstance(alerts, list)
        snap = mon.get_latest_snapshot()
        assert snap is not None
        assert snap.throughput_uph == 44.2

    def test_violation_alert(self):
        c1 = make_c1()
        mon = MonitoringService(contract=c1)
        bad_metrics = SimulationMetrics(
            throughput_uph=53.6, defect_rate=0.010, noise_db=85.0,
            fatigue_index=0.60, energy_kwh=462.0, deadline_gap_pct=0.0,
        )
        alerts = mon.push_snapshot(bad_metrics, phase=3)
        violations = [a for a in alerts if a.severity == "violation"]
        assert len(violations) >= 2  # Fatigue AND noise violated

    def test_stream_simulation(self):
        c1 = make_c1()
        mon = MonitoringService(contract=c1)
        metrics_seq = [
            SimulationMetrics(
                throughput_uph=44.2, defect_rate=0.008, noise_db=76.5,
                fatigue_index=0.33, energy_kwh=398.0, deadline_gap_pct=15.8,
            ),
            SimulationMetrics(
                throughput_uph=46.7, defect_rate=0.009, noise_db=79.1,
                fatigue_index=0.38, energy_kwh=417.0, deadline_gap_pct=14.9,
            ),
        ]
        snapshots = list(mon.stream_simulation(metrics_seq, phase_sequence=[1, 5]))
        assert len(snapshots) == 2
        assert snapshots[0].active_phase == 1
        assert snapshots[1].active_phase == 5

    def test_clear(self):
        c1 = make_c1()
        mon = MonitoringService(contract=c1)
        metrics = SimulationMetrics(
            throughput_uph=44.2, defect_rate=0.008, noise_db=76.5,
            fatigue_index=0.33, energy_kwh=398.0, deadline_gap_pct=15.8,
        )
        mon.push_snapshot(metrics)
        mon.clear()
        assert mon.get_snapshots() == []
        assert mon.get_alerts() == []


# ═══════════════════════════════════════════════════════════════════════
#  ExportService
# ═══════════════════════════════════════════════════════════════════════

class TestExportService:
    @pytest.fixture
    def experiment_result(self):
        svc = ExperimentService()
        return svc.run_all()

    def test_table3_csv(self, experiment_result):
        csv = ExportService.table3_csv(experiment_result)
        assert "S1" in csv
        assert "S2" in csv
        assert "S3" not in csv
        lines = csv.strip().split("\n")
        assert len(lines) == 3  # header + two evaluated schedules

    def test_table3_latex(self, experiment_result):
        latex = ExportService.table3_latex(experiment_result)
        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert "$S_{1}$" in latex

    def test_table3_json(self, experiment_result):
        j = ExportService.table3_json(experiment_result)
        data = json.loads(j)
        assert "S1" in data
        assert "throughput_uph" in data["S1"]

    def test_full_results_json(self, experiment_result):
        j = ExportService.full_results_json(experiment_result)
        data = json.loads(j)
        assert "phases" in data
        assert "audit" in data
        assert len(data["phases"]) == 4

    def test_results_zip(self, experiment_result):
        zdata = ExportService.results_zip(experiment_result)
        assert isinstance(zdata, bytes)
        with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
            names = zf.namelist()
            assert "table3.csv" in names
            assert "table3.tex" in names
            assert "table3.json" in names
            assert "full_results.json" in names
            assert "audit_trail.json" in names


# ═══════════════════════════════════════════════════════════════════════
#  EventBus
# ═══════════════════════════════════════════════════════════════════════

class TestEventBus:
    def test_publish_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe(TOPIC_KPI_STREAM, lambda msg: received.append(msg))
        bus.publish(TOPIC_KPI_STREAM, {"test": 1})
        assert len(received) == 1
        assert received[0].payload == {"test": 1}

    def test_history(self):
        bus = EventBus()
        bus.publish(TOPIC_KPI_STREAM, {"a": 1})
        bus.publish(TOPIC_KPI_STREAM, {"b": 2})
        history = bus.get_history(TOPIC_KPI_STREAM)
        assert len(history) == 2

    def test_clear(self):
        bus = EventBus()
        bus.publish(TOPIC_KPI_STREAM, {"a": 1})
        bus.clear()
        assert bus.get_history() == []
