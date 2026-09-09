"""Runtime obligations fail closed without falsifying observations."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from cbpa.layer3_verification.constraint_checker import ConstraintChecker, _METRIC_MAP
from cbpa.layer3_verification.feasibility_cert import CertificateIssuer
from cbpa.layer3_verification.guard_synthesizer import GuardEnforcer, GuardSynthesizer, RuntimeGuard, RuntimeGuardViolation
from cbpa.models.contract import HardConstraint, make_c1, make_c1_factory
from cbpa.models.metrics import FactoryMetrics, SimulationMetrics
from cbpa.runner.experiment import CBPAExperiment
from cbpa.runner.factory_experiment import FactoryExperiment
from cbpa.runner.lifecycle import DeploymentBlocked


def readings(factory=False):
    if factory:
        return FactoryMetrics(factory_noise_db=70, operator_fatigue={'H1':.2,'H2':.2,'H3':0}, agv_utilization=.5, cell_balance_loss_pct=0)
    return SimulationMetrics(throughput_uph=40, defect_rate=.01, noise_db=70, fatigue_index=.2, energy_kwh=200, deadline_gap_pct=20)


@pytest.mark.parametrize('factory', [False, True])
def test_certificate_guards_cover_every_encoded_constraint(factory):
    contract = make_c1_factory() if factory else make_c1()
    checker = ConstraintChecker()
    report = checker.check_factory(contract, readings(True)) if factory else checker.check(contract, readings())
    cert = CertificateIssuer().issue('test', report, contract)
    assert cert.is_certified
    assert {g.name for g in cert.guards if g.action_on_violation != 'warn'} == {c.name for c in contract.hard_constraints}
    assert GuardEnforcer(cert.guards, cyber_risk_level=1).check_and_enforce(readings(factory)) == []


@pytest.mark.parametrize('factory', [False, True])
@pytest.mark.parametrize('op,limit,passes', [('<=',70,True),('<',70,False),('>=',70,True),('>',70,False),('==',70,True),('==',69,False),('>=',71,False)])
def test_runtime_and_nominal_operator_semantics_agree(factory, op, limit, passes):
    contract = make_c1_factory() if factory else make_c1()
    contract.hard_constraints = [HardConstraint(name='FactoryNoise' if factory else 'Noise', operator=op, limit=limit)]
    guards = GuardSynthesizer().synthesize(contract, factory=factory)
    enforcer = GuardEnforcer(guards)
    metrics = readings(factory)
    original = metrics.model_dump()
    if passes:
        assert all(e.action == 'warn' for e in enforcer.check_and_enforce(metrics))
    else:
        with pytest.raises(RuntimeGuardViolation):
            enforcer.check_and_enforce(metrics)
    assert metrics.model_dump() == original


@pytest.mark.parametrize('factory', [False, True])
@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), float('-inf')])
def test_missing_or_nonfinite_observation_aborts_and_serializes_as_null(factory, bad):
    contract = make_c1_factory() if factory else make_c1()
    enforcer = GuardEnforcer(GuardSynthesizer().synthesize(contract, factory=factory), cyber_risk_level=1)
    metrics = readings(factory).model_copy(update={'factory_noise_db' if factory else 'noise_db':bad})
    with pytest.raises(RuntimeGuardViolation) as caught:
        enforcer.check_and_enforce(metrics)
    assert caught.value.events == enforcer.enforcement_log
    assert all(e.action == 'escalate' and e.metric_value is None and not e.enforced for e in caught.value.events)
    assert all('"metric_value":null' in e.model_dump_json() for e in caught.value.events)


def test_factory_missing_operator_and_defaulted_noise_abort():
    enforcer = GuardEnforcer(GuardSynthesizer().synthesize(make_c1_factory(), factory=True), cyber_risk_level=1)
    for metrics in [readings(True).model_copy(update={'operator_fatigue': {'H1':.2,'H3':0}}), FactoryMetrics(operator_fatigue={'H1':.2,'H2':.2,'H3':0},agv_utilization=.5)]:
        with pytest.raises(RuntimeGuardViolation):
            enforcer.check_and_enforce(metrics)


@pytest.mark.parametrize('name', list(_METRIC_MAP))
def test_every_supported_cell_alias_is_monitored(name):
    contract = make_c1()
    checker = ConstraintChecker()
    actual = checker._actual(name, readings(), False)
    contract.hard_constraints = [HardConstraint(name=name, operator='==', limit=actual)]
    enforcer = GuardEnforcer(GuardSynthesizer().synthesize(contract), cyber_risk_level=1)
    assert enforcer.check_and_enforce(readings()) == []
    contract.hard_constraints[0].limit = actual + 1
    with pytest.raises(RuntimeGuardViolation):
        GuardEnforcer(GuardSynthesizer().synthesize(contract), cyber_risk_level=1).check_and_enforce(readings())


@pytest.mark.parametrize('name', ['FactoryNoise','factory_noise','CellBalanceLoss','AGVTransferTime','FatigueIndex_H1','FatigueIndex_H2','FatigueIndex_H3'])
def test_every_factory_mapping_and_agv_conversion_is_monitored(name):
    contract = make_c1_factory()
    actual = ConstraintChecker()._actual(name, readings(True), True)
    contract.hard_constraints = [HardConstraint(name=name, operator='==', limit=actual)]
    assert GuardEnforcer(GuardSynthesizer().synthesize(contract, factory=True)).check_and_enforce(readings(True)) == []
    contract.hard_constraints[0].limit = actual + 1
    with pytest.raises(RuntimeGuardViolation):
        GuardEnforcer(GuardSynthesizer().synthesize(contract, factory=True)).check_and_enforce(readings(True))


def test_cyber_proxy_requires_explicit_model_input():
    contract = make_c1()
    contract.hard_constraints = [HardConstraint(name='CyberRiskLevel',limit=2)]
    guards = GuardSynthesizer().synthesize(contract)
    with pytest.raises(RuntimeGuardViolation):
        GuardEnforcer(guards).check_and_enforce(readings())
    assert GuardEnforcer(guards, cyber_risk_level=1).check_and_enforce(readings()) == []


@pytest.mark.parametrize('name,operator', [('Unknown','<='), ('Noise','~=')])
def test_unmonitorable_contract_is_rejected_at_synthesis(name, operator):
    contract = make_c1()
    contract.hard_constraints = [HardConstraint(name=name, operator='<=' ,limit=1).model_copy(update={'operator':operator})]
    with pytest.raises(ValueError):
        GuardSynthesizer().synthesize(contract)
    assert not RuntimeGuard('bad','noise_db','~=',70).check(70)


def test_clamp_request_does_not_rewrite_a_measurement():
    metrics = readings()
    enforcer = GuardEnforcer([RuntimeGuard('energy','energy_kwh','<=',100,'clamp')])
    with pytest.raises(RuntimeGuardViolation) as caught:
        enforcer.check_and_enforce(metrics)
    assert metrics.energy_kwh == 200
    assert caught.value.events[0].action == 'escalate'
    assert not caught.value.events[0].enforced


@pytest.mark.parametrize('factory', [False,True])
@pytest.mark.parametrize('missing', [False,True])
def test_runtime_failure_aborts_lifecycle_and_is_audited(factory, missing):
    exp = FactoryExperiment(mode='analytical',use_llm=False) if factory else CBPAExperiment(use_llm=False)
    contract = make_c1_factory() if factory else make_c1()
    exp._guard_enforcer = GuardEnforcer(GuardSynthesizer().synthesize(contract, factory=factory),cyber_risk_level=1)
    metrics = readings(factory).model_copy(update={'factory_noise_db' if factory else 'noise_db': None if missing else 100})
    p1 = SimpleNamespace(contract=contract, factory_schedule=SimpleNamespace(name='test'), factory_metrics=metrics) if factory else SimpleNamespace(contract=contract, schedule=SimpleNamespace(name='test'), metrics=metrics)
    # Stop at the first monitoring operation, before disturbance/replanning.
    with pytest.raises(DeploymentBlocked, match='Runtime guard stopped phase 2'):
        exp.run_single_phase(2,[p1])
    assert exp._guard_enforcer is None
    assert exp.audit.entries[-1].action == 'runtime_execution_blocked'
    assert not any(e.action == 'schedule_deployed' for e in exp.audit.entries)


def test_integrated_guard_abort_cannot_fall_back_to_simulation():
    exp = CBPAExperiment(use_llm=False)
    exp.config.use_integrated = True
    exp._orchestrator = MagicMock()
    enforcer = GuardEnforcer([RuntimeGuard('noise','noise_db','<=',60)])
    with pytest.raises(RuntimeGuardViolation) as caught:
        enforcer.check_and_enforce(readings())
    exp._orchestrator.deploy_and_run_schedule.side_effect = caught.value
    with pytest.raises(RuntimeGuardViolation):
        exp._execute_integrated(SimpleNamespace(name='test'),phase=1)
