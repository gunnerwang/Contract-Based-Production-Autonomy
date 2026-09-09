"""Cross-path regression checks from the comprehensive revision audit."""
from unittest.mock import MagicMock
import pytest
from cbpa.config.scenario import ScenarioConfig
from cbpa.layer4_execution.assumption_tracker import AssumptionTracker
from cbpa.models.contract import Assumption, HardConstraint, make_c1, make_c1_factory, make_c3_factory, AmbiguityFlag
from cbpa.runner.experiment import CBPAExperiment
from cbpa.runner.factory_experiment import FactoryExperiment
from cbpa.runner.lifecycle import DeploymentBlocked
from cbpa.service.experiment_service import ExperimentService, ExperimentState
from cbpa.service.factory_experiment_service import FactoryExperimentService


def test_factory_service_retains_error_and_stops_before_later_phases():
    service = FactoryExperimentService(use_llm=False)
    exp = service._ensure_experiment()
    exp._round_initial_deploy = MagicMock(side_effect=DeploymentBlocked('rejected'))
    exp._round_monitor = MagicMock()
    callbacks = []
    result = service.run_all(on_phase_complete=callbacks.append)
    assert service.status.state == ExperimentState.ERROR
    assert service.status.error == 'rejected'
    assert result.phases == [] and callbacks == []
    assert service.run_phase(2) is None
    exp._round_monitor.assert_not_called()
    service.reset()
    assert service.status.state == ExperimentState.IDLE


def test_cell_service_uses_canonical_runtime_failure_audit():
    service = ExperimentService(use_llm=False)
    p1 = service.run_phase(1)
    assert p1 is not None
    p1.metrics.noise_db = 100
    assert service.run_phase(2) is None
    assert service.status.state == ExperimentState.ERROR
    assert service._experiment.audit.entries[-1].action == 'runtime_execution_blocked'
    assert service._experiment._guard_enforcer is None
    assert service.run_phase(3) is None


@pytest.mark.parametrize('phase',[5,6])
def test_runtime_rejection_never_commits_accepted_learning(phase):
    exp = CBPAExperiment(config=ScenarioConfig(optimiser_anchor=True, enable_macro_phase=True, use_simulation=True),use_llm=False)
    prior = []
    for n in range(1,phase):
        prior.append(exp.run_single_phase(n,prior))
    count = exp.learning.total_records
    def violate(*args,**kwargs):
        bad = prior[0].metrics.model_copy(update={'noise_db':100})
        exp._guard_enforcer.check_and_enforce(bad)
    exp._execute_simulation = violate
    with pytest.raises(DeploymentBlocked,match='Runtime guard stopped'):
        exp.run_single_phase(phase,prior)
    assert exp.learning.total_records == count
    assert exp.audit.entries[-1].action == 'runtime_execution_blocked'


@pytest.mark.parametrize('factory',[False,True])
@pytest.mark.parametrize('phase',[1,5,6])
@pytest.mark.parametrize('operator',['>=','>'])
def test_mandatory_upper_bound_cannot_be_inverted(factory,phase,operator):
    exp = FactoryExperiment(use_llm=False) if factory else CBPAExperiment(use_llm=False)
    contract = make_c1_factory() if factory else make_c1()
    contract.hard_constraints[0].operator = operator
    with pytest.raises(DeploymentBlocked,match='mandatory upper bound'):
        exp._prepare_contract(contract,phase,factory=factory)


@pytest.mark.parametrize('factory',[False,True])
@pytest.mark.parametrize('phase',[4,5,6])
def test_context_revision_preserves_stricter_prior_obligations(factory,phase):
    exp = FactoryExperiment(use_llm=False) if factory else CBPAExperiment(use_llm=False)
    old = make_c1_factory() if factory else make_c1()
    new = old.model_copy(deep=True)
    noise = 'FactoryNoise' if factory else 'Noise'
    old.get_constraint(noise).limit = 75
    old.hard_constraints.append(HardConstraint(name='CellBalanceLoss' if factory else 'Energy',limit=10))
    before = old.model_dump()
    result = exp._prepare_contract(new,phase,factory=factory,previous=old)
    assert result.get_constraint(noise).limit == 75
    assert result.get_constraint('CellBalanceLoss' if factory else 'Energy').limit == 10
    assert old.model_dump() == before
    assert new.get_constraint(noise).limit > 75


def test_stricter_new_bound_retained_and_incompatible_change_halts():
    exp = CBPAExperiment(use_llm=False)
    old = make_c1();new = make_c1()
    new.get_constraint('Noise').limit = 70
    assert exp._prepare_contract(new,5,previous=old).get_constraint('Noise').limit == 70
    new.get_constraint('Noise').operator = '=='
    with pytest.raises(DeploymentBlocked,match='Incompatible revision'):
        exp._prepare_contract(new,5,previous=old)


@pytest.mark.parametrize('phase',[5,6])
def test_revised_contract_must_also_resolve_ambiguities(phase):
    exp = CBPAExperiment(use_llm=False)
    contract = make_c1()
    contract.ambiguities = [AmbiguityFlag(field='test',question='Which?',default_resolution='unknown')]
    with pytest.raises(DeploymentBlocked,match='unresolved'):
        exp._prepare_contract(contract,phase)


def test_next_shift_assumption_represents_absence_and_zero_change_is_detected():
    contract = make_c3_factory()
    absence = next(a for a in contract.typed_assumptions if a.name == 'h2_availability')
    assert absence.expected_value == 0 and absence.tolerance_pct == 0
    tracker = AssumptionTracker([absence])
    assert tracker.update(absence.name,0) is None
    drift = tracker.update(absence.name,1)
    assert drift is not None and drift.assumption.violated
    assert drift.drift_pct is None and 'undefined' in drift.attribution
    assert tracker.update(absence.name,0) is None
    assert tracker.get_violations() == []


@pytest.mark.parametrize('bad',[None,float('nan'),float('inf'),True])
def test_invalid_assumption_observation_cannot_be_reported_as_stable(bad):
    assumption = Assumption(name='demand',description='test',expected_value=52,tolerance_pct=10)
    with pytest.raises(ValueError):
        assumption.check(bad)
    assert assumption.violated
    with pytest.raises(ValueError):
        AssumptionTracker([assumption]).update('demand',bad)


def test_assumption_threshold_uses_unrounded_change():
    assumption = Assumption(name='demand',description='test',expected_value=100,tolerance_pct=10)
    assert not assumption.check(110.01)
    assert assumption.drift_pct == 10.0 and assumption.violated
    assert AssumptionTracker([assumption]).update('demand',110.01) is not None


@pytest.mark.parametrize('error',[ValueError('malformed sensor data'), TypeError('invalid sensor payload'), RuntimeError('bridge failure')])
def test_integrated_execution_errors_cannot_fall_back_to_another_run(error):
    exp = CBPAExperiment(use_llm=False)
    exp.config.use_integrated = True
    exp._orchestrator = MagicMock()
    exp._orchestrator.deploy_and_run_schedule.side_effect = error
    from cbpa.models.schedule import Schedule
    schedule = Schedule(r1_speed_fraction=.7,r2_speed_fraction=.7,human_cycle_rate_multiplier=1,buffer_time_s=1,demand_target_uph=40)
    with pytest.raises(DeploymentBlocked,match='Integrated execution failed'):
        exp._execute_integrated(schedule,1)
    assert exp.audit.entries[-1].action == 'runtime_execution_blocked'
    assert exp._guard_enforcer is None


@pytest.mark.parametrize('service_type',[ExperimentService,FactoryExperimentService])
def test_escalation_failure_persists_error_and_returns_partial_result(service_type):
    service = service_type(use_llm=False)
    service._complete_phase4 = MagicMock(side_effect=DeploymentBlocked('invalid revised contract'))
    callbacks = []
    result = service.run_all(on_phase_complete=callbacks.append)
    assert service.status.state == ExperimentState.ERROR
    assert service.status.error == 'invalid revised contract'
    assert [p.phase for p in result.phases] == [1,2,3]
    assert [p.phase for p in callbacks] == [1,2,3]
    assert service.run_phase(5) is None


@pytest.mark.parametrize('observed',[None,20.0])
def test_attribution_retains_observed_drift_instead_of_invented_llm_value(observed):
    from cbpa.layer4_execution.attribution_agent import L4AttributionAgent
    from cbpa.llm.prompts import L4_ATTRIBUTION_TOOL_SCHEMA
    client = MagicMock()
    client.is_available = True
    client.query_structured.return_value = {'explanations':[{'drifted_assumption':'demand','drift_pct':0}]}
    result = L4AttributionAgent(client,use_llm=True).explain_breach(
        2,[{'name':'demand','expected':0,'actual':1,'drift_pct':observed}],{}, {'Noise':-1})
    assert result.llm_used
    assert result.explanations[0].drift_pct == observed
    schema = L4_ATTRIBUTION_TOOL_SCHEMA['properties']['explanations']['items']['properties']['drift_pct']
    assert set(schema['type']) == {'number','null'}
