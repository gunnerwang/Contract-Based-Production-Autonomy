"""Behavior regressions for contract-specific phase-6 adaptation and review gating."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from cbpa.models.contract import make_c1,HardConstraint
from cbpa.models.schedule import Schedule
from cbpa.config.scenario import ScenarioConfig
from cbpa.runner.experiment import CBPAExperiment
from cbpa.runner.lifecycle import DeploymentBlocked
from cbpa.physics.cell_evaluator import CellEvaluator
from cbpa.layer2_planning.vr_scorer import VRScorer
from cbpa.layer3_verification.constraint_checker import ConstraintChecker
from cbpa.baselines.optimizer_planner import OptimiserPlanner,BOUNDS
from cbpa.layer1_outcome.reconciliation import reconcile_for_review
from cbpa.layer1_outcome.contract_validator import ContractValidator
from cbpa.layer1_outcome.clarification import ClarificationAnswer,apply_answer


def schedule(**kw):
    params=dict(name='test',r1_speed_fraction=.3,r2_speed_fraction=.3,
                human_cycle_rate_multiplier=.8,buffer_time_s=6,demand_target_uph=52)
    params.update(kw);return Schedule(**params)


def test_degraded_envelope_applies_to_nominal_and_sampling():
    ev=CellEvaluator(mode='analytical',r1_speed_cap=.45)
    with pytest.raises(ValueError):ev.evaluate(schedule(r1_speed_fraction=.46))
    with pytest.raises(ValueError):ev.evaluate_monte_carlo(schedule(r1_speed_fraction=.46))
    # Capture the physical model inputs to confirm perturbations cannot exceed cap.
    original=ev.energy_model.compute;seen=[]
    def record(r1,*args): seen.append(r1);return original(r1,*args)
    ev.energy_model.compute=record
    ev.evaluate_monte_carlo(schedule(r1_speed_fraction=.45),n_samples=200,seed=42)
    assert max(seen)<=.45 and len(seen)==200


def test_optimizer_searches_retained_contract_within_degraded_bounds():
    c=make_c1();c.get_constraint('FatigueIndex').limit=.30;c.get_constraint('Noise').limit=75
    ev=CellEvaluator(mode='analytical',r1_speed_cap=.45);checker=ConstraintChecker()
    bounds=dict(BOUNDS,r1=(.3,.45))
    result=OptimiserPlanner(ev,VRScorer(mode='analytical'),checker,certify=True).plan(c,52,bounds=bounds)
    assert result.feasible_found and result.schedule.r1_speed_fraction<=.45
    assert result.p_feasible>=.95 and checker.check(c,ev.evaluate(result.schedule)).is_feasible
    assert BOUNDS['r1']==(.3,1.)


def test_certified_lower_tier_is_executed_and_retains_obligations():
    c=make_c1();exp=CBPAExperiment(config=ScenarioConfig(use_simulation=True,enable_macro_phase=True),use_llm=False)
    p=exp._phase6_macro(SimpleNamespace(contract=c,schedule=schedule()))
    assert p.phase==6 and p.feasibility.is_feasible and p.schedule.r1_speed_fraction<=.5
    assert ConstraintChecker.constraint_spec(p.contract)==ConstraintChecker.constraint_spec(c)
    assert exp.learning.total_records==1
    assert any(e.action=='schedule_deployed' and e.phase==6 for e in exp.audit.entries)


def test_impossible_contract_still_refuses_after_macro_search():
    c=make_c1();c.hard_constraints.append(HardConstraint(name='Energy',limit=3,unit='kWh'))
    exp=CBPAExperiment(config=ScenarioConfig(optimiser_anchor=True),use_llm=False)
    exp.elicitation.elicit=MagicMock(return_value=c)
    with pytest.raises(DeploymentBlocked):exp._phase6_macro(SimpleNamespace(contract=c,schedule=schedule()))
    assert not any(e.action=='schedule_deployed' for e in exp.audit.entries)
    assert exp.learning.total_records==0


def test_disagreement_is_not_silently_added_or_discarded():
    llm=make_c1();rule=make_c1();rule.hard_constraints.append(HardConstraint(name='Energy',limit=350,unit='kWh'))
    result=reconcile_for_review(llm,rule)
    assert result.get_constraint('Energy') is None
    assert not ContractValidator().validate(result).is_valid
    assert result.ambiguities[0].field=='crosscheck:Energy'
    answered=apply_answer(result,'crosscheck:Energy',ClarificationAnswer(text='The quoted cap is not a requirement.',patch={'disposition':'retain'}),confirmed=True)
    assert ContractValidator().validate(answered).is_valid
    assert not result.ambiguities[0].resolved


def test_review_gate_does_not_relax_a_mandatory_bound():
    llm=make_c1();llm.get_constraint('Noise').limit=90
    with pytest.raises(ValueError):reconcile_for_review(llm,llm)


def test_review_gate_flags_missing_llm_constraint_and_strictness():
    llm=make_c1();rule=make_c1();rule.get_constraint('Noise').operator='<'
    result=reconcile_for_review(llm,rule)
    assert not ContractValidator().validate(result).is_valid
    assert result.get_constraint('Noise').operator=='<='
