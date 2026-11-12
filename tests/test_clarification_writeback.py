"""Regression tests for intent writeback and refusal before scheduling."""
import pytest
from unittest.mock import MagicMock
from cbpa.models.contract import make_c1, AmbiguityFlag
from cbpa.layer1_outcome.clarification import ClarificationAnswer, apply_answer
from cbpa.layer1_outcome.elicitation_agent import ElicitationAgent
from cbpa.layer1_outcome.contract_validator import ContractValidator
from cbpa.runner.experiment import CBPAExperiment
from cbpa.runner.factory_experiment import FactoryExperiment
from cbpa.runner.lifecycle import DeploymentBlocked

def draft():
    c = make_c1()
    c.ambiguities = [AmbiguityFlag(field="manager", question="Confirm limits?", default_resolution="No default approval")]
    return c

def answer(patch):
    return ClarificationAnswer(text="Reviewed manager answer", patch=patch)

def test_confirmed_energy_cap_is_written_before_candidate_evaluation():
    exp = CBPAExperiment(use_llm=False)
    with pytest.raises(DeploymentBlocked):
        exp._phase1()
    assert any(e.action == "deployment_blocked" and e.phase == 1 for e in exp.audit.entries)
    assert not any(e.action == "schedule_deployed" for e in exp.audit.entries)
    c, flags = ElicitationAgent().elicit_with_refinement("Increase output")
    assert c.get_constraint("Energy").limit == 3.0
    assert all(f.resolved and f.applied_patch for f in flags)

def test_atomic_writeback_keeps_strict_comparison_kpi_and_priority():
    c = draft()
    patch = {"constraints": [{"name":"Noise","operator":"<","limit":77,"unit":"dB"}],
             "kpis":[{"name":"Throughput","direction":"lower_bound","threshold":31,"unit":"u/h"}],
             "priorities":["Safety","HumanWellbeing","Throughput","Quality"]}
    updated = apply_answer(c,"manager",answer(patch),confirmed=True)
    assert updated.get_constraint("Noise").operator == "<"
    assert updated.get_constraint("Noise").limit == 77
    assert next(k for k in updated.kpi_targets if k.name=="Throughput").threshold == 31
    assert updated.priority_order[0].value == "Safety"
    assert updated.ambiguities[0].resolved
    assert c.get_constraint("Noise").limit == 80 and not c.ambiguities[0].resolved

@pytest.mark.parametrize("patch",[
 {"constraints":[{"name":"Noise","operator":"<=","limit":81}]},
 {"constraints":[{"name":"Noise","operator":">=","limit":70}]},
 {"constraints":[{"name":"Noise","operator":"<=","limit":float('nan')}]},
 {"constraints":[{"name":"Energy","limit":3,"unit":"MWh"}]},
 {"constraints":[{"name":"UnobservedMetric","limit":3}]},
 {"context":{"shift_hours":9}},
 {"context":{"overtime_allowed":True}},
 {"kpis":[{"name":"Throughput","direction":"lower_bound"}]},
 {"priorities":["Safety","Safety"]},
 {"assumptions":[{"name":"untracked","description":"x","expected_value":1}]},
 {}])
def test_bad_updates_refuse_without_partial_mutation(patch):
    c=draft(); before=c.model_dump()
    with pytest.raises(ValueError): apply_answer(c,"manager",answer(patch),confirmed=True)
    assert c.model_dump()==before

def test_unconfirmed_patch_cannot_resolve():
    with pytest.raises(ValueError): apply_answer(draft(),"manager",answer({"disposition":"retain"}),confirmed=False)

def test_unknown_question_never_uses_model_default_as_manager_approval():
    agent=ElicitationAgent()
    agent._detect_ambiguities=MagicMock(return_value=draft().ambiguities)
    c,flags=agent.elicit_with_refinement("Increase output")
    assert not flags[0].resolved and flags[0].resolution==""
    assert not ContractValidator().validate(c).is_valid

def test_explicit_retain_does_not_add_energy():
    agent=ElicitationAgent()
    c,flags=agent.elicit_with_refinement("Increase output",confirmed_answers={
        "energy_kwh_cap":{"text":"Keep energy as a soft guideline; no hard cap.","patch":{"disposition":"retain"}},
        "shift_hours_overflow":{"text":"Keep the existing shift.","patch":{"context":{"overtime_allowed":False}}}})
    assert c.get_constraint("Energy") is None and all(f.resolved for f in flags)

@pytest.mark.parametrize("cls",[CBPAExperiment,FactoryExperiment])
def test_refinement_exception_cannot_fall_back_to_unclarified_contract(cls):
    exp=cls(use_llm=False)
    exp.elicitation.elicit_with_refinement=MagicMock(side_effect=ValueError("invalid confirmed patch"))
    with pytest.raises(DeploymentBlocked): exp.run_all_phases()
    assert not any(e.action=="schedule_deployed" for e in exp.audit.entries)


def test_model_question_cannot_authorize_a_scripted_answer_by_field_name():
    agent = ElicitationAgent()
    agent.use_llm = True
    agent.elicit = MagicMock(return_value=make_c1())
    agent._detect_ambiguities = MagicMock(return_value=[AmbiguityFlag(
        field="energy_kwh_cap",question="Approve a different cap?",default_resolution="Approve")])
    c, flags = agent.elicit_with_refinement("Increase output")
    assert not flags[0].resolved and c.get_constraint("Energy") is None
    assert not ContractValidator().validate(c).is_valid


def test_legacy_resolved_boolean_is_not_writeback_evidence():
    c = draft(); c.ambiguities[0].resolved=True; c.ambiguities[0].resolution="yes"
    assert not ContractValidator().validate(c).is_valid
