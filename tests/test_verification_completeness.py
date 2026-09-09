"""Every encoded obligation must be evaluable before certification."""
from unittest.mock import MagicMock
import pytest
from cbpa.layer1_outcome.contract_validator import ContractValidator
from cbpa.layer3_verification.constraint_checker import ConstraintChecker
from cbpa.layer3_verification.feasibility_cert import CertificateIssuer
from cbpa.models.contract import HardConstraint, make_c1, make_c1_factory
from cbpa.models.metrics import FactoryMetrics, SimulationMetrics
from cbpa.runner.experiment import CBPAExperiment
from cbpa.runner.factory_experiment import FactoryExperiment
from cbpa.runner.lifecycle import DeploymentBlocked


def metrics(factory):
    if factory:
        return FactoryMetrics(factory_noise_db=70, operator_fatigue={'H1':.2,'H2':.2,'H3':0},agv_utilization=.5,cell_balance_loss_pct=0)
    return SimulationMetrics(throughput_uph=40,defect_rate=.01,noise_db=70,fatigue_index=.2,energy_kwh=200,deadline_gap_pct=20)


def check(factory):
    c=ConstraintChecker()
    return (c.check_factory,c.check_factory_stochastic) if factory else (c.check,c.check_stochastic)


@pytest.mark.parametrize('factory',[False,True])
@pytest.mark.parametrize('defect',['unknown','operator','nan_limit','infinite_limit','empty','duplicate'])
def test_unevaluable_contract_never_certifies(factory,defect):
    contract=make_c1_factory() if factory else make_c1()
    first=contract.hard_constraints[0]
    if defect=='unknown': contract.hard_constraints.append(HardConstraint(name='UnmodelledHazard',limit=0))
    elif defect=='operator': contract.hard_constraints[0]=first.model_copy(update={'operator':'~='})
    elif defect in {'nan_limit','infinite_limit'}: contract.hard_constraints[0]=first.model_copy(update={'limit':float('nan') if defect=='nan_limit' else float('inf')})
    elif defect=='empty': contract.hard_constraints=[]
    else: contract.hard_constraints.append(first.model_copy())
    assert not ContractValidator().validate(contract,factory=factory).is_valid
    nominal,stochastic=check(factory)
    report=nominal(contract,metrics(factory))
    sampled=stochastic(contract,[metrics(factory)]*200)
    assert report.evaluation_errors and not report.is_feasible
    assert sampled.evaluation_errors and not sampled.certified
    cert=CertificateIssuer().issue('test',report,contract,stochastic_report=sampled)
    assert not cert.is_certified and not cert.guards
    with pytest.raises(DeploymentBlocked): CBPAExperiment(use_llm=False)._require_deployment_certificate(cert,1)


@pytest.mark.parametrize('factory',[False,True])
@pytest.mark.parametrize('bad_value',[None,float('nan'),float('inf'),float('-inf')])
def test_one_missing_or_nonfinite_sample_blocks_despite_high_fraction(factory,bad_value):
    contract=make_c1_factory() if factory else make_c1()
    good=metrics(factory)
    bad=good.model_copy(update={'factory_noise_db' if factory else 'noise_db':bad_value})
    nominal,stochastic=check(factory)
    sampled=stochastic(contract,[good]*199+[bad])
    assert sampled.p_feasible==.995
    assert not sampled.certified and sampled.evaluation_errors
    name='FactoryNoise' if factory else 'Noise'
    assert sampled.constraint_violation_probabilities[name]==.005
    assert name not in sampled.worst_case_margins
    cert=CertificateIssuer().issue('test',nominal(contract,good),contract,stochastic_report=sampled)
    assert not cert.is_certified
    with pytest.raises(DeploymentBlocked): CBPAExperiment(use_llm=False)._require_deployment_certificate(cert,1)


def test_missing_operator_and_defaulted_factory_measurement_are_not_zero_readings():
    nominal,stochastic=check(True)
    contract=make_c1_factory()
    for bad in [metrics(True).model_copy(update={'operator_fatigue':{'H1':.2,'H3':0}}),FactoryMetrics(operator_fatigue={'H1':.2,'H2':.2,'H3':0},agv_utilization=.5)]:
        assert not nominal(contract,bad).is_feasible
        assert not stochastic(contract,[bad]*200).certified


@pytest.mark.parametrize('factory',[False,True])
@pytest.mark.parametrize('op,limit,expected',[('<=',70,True),('<',70,False),('>=',70,True),('>',70,False),('==',70,True),('==',69,False),('>=',69,True),('>=',71,False)])
def test_probability_obeys_operator_including_equality(factory,op,limit,expected):
    contract=make_c1_factory() if factory else make_c1()
    contract.hard_constraints=[HardConstraint(name='FactoryNoise' if factory else 'Noise',operator=op,limit=limit)]
    nominal,stochastic=check(factory)
    report=nominal(contract,metrics(factory));sampled=stochastic(contract,[metrics(factory)]*200)
    assert report.is_feasible==expected
    assert sampled.p_feasible==int(expected)
    assert list(sampled.constraint_violation_probabilities.values())==[int(not expected)]


@pytest.mark.parametrize('factory',[False,True])
def test_unknown_constraint_halts_actual_lifecycle_before_planning(factory):
    exp=FactoryExperiment(use_llm=False) if factory else CBPAExperiment(use_llm=False)
    contract=make_c1_factory() if factory else make_c1()
    contract.hard_constraints.append(HardConstraint(name='UnmodelledHazard',limit=0))
    exp.elicitation.elicit_with_refinement=MagicMock(return_value=(contract,[]))
    planner=exp.factory_planner if factory else exp.planner
    planner.generate_initial=MagicMock()
    with pytest.raises(DeploymentBlocked,match='UnmodelledHazard'): exp.run_all_phases()
    planner.generate_initial.assert_not_called()
    assert exp.audit.entries[-1].action=='deployment_blocked'
    assert not any(e.action=='schedule_deployed' for e in exp.audit.entries)


def test_sensitivity_tuple_preserves_documented_metric_constraint_order():
    good=metrics(False)
    varied=[good.model_copy(update={'noise_db':70+i/100}) for i in range(200)]
    report=ConstraintChecker().check_stochastic(make_c1(),varied)
    assert ('noise_db','Noise',1.0) in report.sensitivity_ranking


def test_certified_optimizer_refuses_incomplete_sampling():
    from cbpa.baselines.optimizer_planner import OptimiserPlanner
    from cbpa.physics.cell_evaluator import CellEvaluator
    from cbpa.config.scenario import CellConfig
    from cbpa.layer2_planning.vr_scorer import VRScorer
    ev=CellEvaluator(CellConfig(),mode='analytical')
    good=metrics(False);bad=good.model_copy(update={'noise_db':None})
    ev.evaluate_monte_carlo=MagicMock(return_value=[good]*199+[bad])
    opt=OptimiserPlanner(ev,VRScorer(mode='analytical'),ConstraintChecker(),certify=True)
    result=opt.plan(make_c1(),52,grid=2,refine=0)
    assert ev.evaluate_monte_carlo.called
    assert not result.feasible_found and result.schedule is None


@pytest.mark.parametrize('factory',[False,True])
@pytest.mark.parametrize('change',['added','changed_limit','sample_report'])
def test_certificate_reports_must_cover_exact_contract(factory,change):
    c=make_c1_factory() if factory else make_c1()
    nominal,stochastic=check(factory)
    report=nominal(c,metrics(factory));sampled=stochastic(c,[metrics(factory)]*200)
    if change=='added': c.hard_constraints.append(HardConstraint(name='UnmodelledHazard',limit=0))
    elif change=='changed_limit': c.hard_constraints[0]=c.hard_constraints[0].model_copy(update={'limit':.1})
    else: sampled.checked_constraints=[]
    cert=CertificateIssuer().issue('test',report,c,stochastic_report=sampled)
    assert not cert.is_certified and 'cover' in cert.notes


@pytest.mark.parametrize('defect',['missing_evaluator','missing_cell','unknown_cell','unknown_operator'])
@pytest.mark.parametrize('mc',[False,True])
def test_partial_factory_never_produces_certifiable_metrics(defect,mc):
    exp=FactoryExperiment(use_llm=False)
    schedules=exp.factory_planner.generate_initial(contract=make_c1_factory(),routing=exp.config.default_variant_routing,assignments=exp.config.default_operator_assignments)
    schedule=schedules[0].model_copy(deep=True)
    if defect=='missing_evaluator': del exp.evaluator.cell_evaluators['B']
    elif defect=='missing_cell': del schedule.cell_schedules['B']
    elif defect=='unknown_cell': schedule.cell_schedules['X']=schedule.cell_schedules.pop('B')
    else: schedule.operator_assignments['Unknown']='A'
    with pytest.raises(ValueError,match='factory|operator'):
        (exp.evaluator.evaluate_monte_carlo(schedule) if mc else exp.evaluator.evaluate(schedule))
    exp.factory_planner.generate_initial=MagicMock(return_value=[schedule])
    with pytest.raises(ValueError): exp.run_all_phases()
    assert not any(e.action=='schedule_deployed' for e in exp.audit.entries)


@pytest.mark.parametrize('factory',[False,True])
def test_candidate_gate_skips_high_probability_incomplete_report(factory):
    from types import SimpleNamespace
    exp=FactoryExperiment(use_llm=False) if factory else CBPAExperiment(use_llm=False)
    c=make_c1_factory() if factory else make_c1()
    if factory:
        candidates=exp.factory_planner.generate_initial(contract=c,routing=exp.config.default_variant_routing,assignments=exp.config.default_operator_assignments)[:2]
    else: candidates=exp.planner.generate_initial(c)[:2]
    names=[s.name for s in candidates]
    rank=SimpleNamespace(selected=names[0],non_dominated=names,dominated=[])
    scores={n:SimpleNamespace(total_throughput_uph=40,vr_score=.5) for n in names}
    nominal,_=check(factory);reports={n:nominal(c,metrics(factory)) for n in names}
    exp.evaluator.evaluate_monte_carlo=MagicMock(side_effect=lambda s,**kw:s.name)
    fake=MagicMock(side_effect=lambda contract,name,**kw:SimpleNamespace(p_feasible=.995,evaluation_errors=['missing sample'] if name==names[0] else []))
    if factory: exp.checker.check_factory_stochastic=fake
    else: exp.checker.check_stochastic=fake
    selected,info=exp._certification_gate(candidates,rank,scores,reports,c,phase=1)
    assert selected.name==names[1] and info['certified']


def test_factory_optimizer_refuses_incomplete_sampling():
    from cbpa.baselines.optimizer_planner import FactoryOptimiserPlanner
    exp=FactoryExperiment(use_llm=False)
    good=metrics(True);bad=good.model_copy(update={'factory_noise_db':None})
    exp.evaluator.evaluate_monte_carlo=MagicMock(return_value=[good]*199+[bad])
    opt=FactoryOptimiserPlanner(exp.evaluator,ConstraintChecker(),certify=True)
    result=opt.plan(make_c1_factory(),exp.config.default_variant_routing,exp.config.default_operator_assignments,
                    ['V_A','V_B','V_C'],['V_A','V_B','V_C'],52,48,samples=32,refine=0)
    assert exp.evaluator.evaluate_monte_carlo.called
    assert not result.feasible_found and result.schedule is None
