"""Review equivalent units automatically, retain unresolved meaning differences."""
from unittest.mock import MagicMock
import pytest
from cbpa.models.contract import make_c1, HardConstraint
from cbpa.layer1_outcome.reconciliation import reconcile_for_review
from cbpa.layer1_outcome.clarification import ClarificationAnswer, apply_answer
from cbpa.layer1_outcome.contract_validator import ContractValidator
from cbpa.runner.experiment import CBPAExperiment, _canonicalise_contract
from cbpa.runner.lifecycle import DeploymentBlocked


def test_equivalent_percent_and_fraction_do_not_request_review():
    left=make_c1();right=make_c1()
    left.hard_constraints.append(HardConstraint(name='DefectRate',limit=.9,unit='percent'))
    right.hard_constraints.append(HardConstraint(name='DefectRate',limit=.009,unit='fraction'))
    result=reconcile_for_review(left,right)
    assert not result.ambiguities and result.get_constraint('DefectRate').limit==.009
    assert left.get_constraint('DefectRate').unit=='percent'


def test_equivalent_energy_units_do_not_request_review():
    left=make_c1();right=make_c1()
    left.hard_constraints.append(HardConstraint(name='Energy',limit=375000,unit='Wh'))
    right.hard_constraints.append(HardConstraint(name='Energy',limit=375,unit='kWh per shift'))
    assert not reconcile_for_review(left,right).ambiguities


def test_unknown_rule_unit_cannot_be_silently_ignored():
    right=make_c1();right.hard_constraints.append(HardConstraint(name='Energy',limit=375,unit='kW'))
    with pytest.raises(ValueError):reconcile_for_review(make_c1(),right)


def test_confirmation_is_bound_to_exact_parser_pair():
    left=make_c1();right=make_c1()
    right.hard_constraints.append(HardConstraint(name='Energy',limit=350,unit='kWh'))
    pending=reconcile_for_review(left,right)
    confirmed=apply_answer(pending,'crosscheck:Energy',ClarificationAnswer(text='The cited cap is not an active requirement.',patch={'disposition':'retain'}),confirmed=True)
    assert ContractValidator().validate(reconcile_for_review(confirmed,right)).is_valid
    right.get_constraint('Energy').limit=300
    changed=reconcile_for_review(confirmed,right)
    assert not ContractValidator().validate(changed).is_valid
    assert not changed.ambiguities[0].resolved and changed.ambiguities[0].applied_patch is None


def test_live_llm_crosscheck_blocks_before_planning_without_tightening(monkeypatch):
    import cbpa.runner.experiment as module
    exp=CBPAExperiment(use_llm=False);exp.elicitation.use_llm=True
    draft=make_c1()
    monkeypatch.setattr(module,'MANAGER_INTENT','Keep noise under 77 dB and fatigue under 0.35.')
    exp.elicitation.elicit_with_refinement=MagicMock(return_value=(draft,[]))
    exp.planner.generate_initial=MagicMock()
    with pytest.raises(DeploymentBlocked):exp._phase1()
    assert draft.get_constraint('Noise').limit==80
    exp.planner.generate_initial.assert_not_called()
    entry=next(e for e in exp.audit.entries if e.action=='parser_crosscheck')
    assert entry.details['policy']=='review' and entry.details['pending_fields']
    assert not any(e.action=='schedule_deployed' for e in exp.audit.entries)


def test_default_binding_does_not_restore_negated_energy_constraint():
    c=_canonicalise_contract(make_c1(),'Energy cap 350 kWh was rejected; keep noise under 80 dB.')
    assert c.get_constraint('Energy') is None
    assert any(f.field=='crosscheck:Energy' and not f.resolved for f in c.ambiguities)
