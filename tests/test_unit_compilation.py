"""Unit representation must preserve physical requirements at compilation/dispatch."""
from unittest.mock import MagicMock
import pytest
from cbpa.models.contract import make_c1, make_c1_factory, HardConstraint, KPITarget
from cbpa.models.metrics import SimulationMetrics
from cbpa.models.units import normalize_contract_units, UnitCompilationError
from cbpa.layer1_outcome.elicitation_agent import ElicitationAgent
from cbpa.layer1_outcome.contract_validator import ContractValidator
from cbpa.layer3_verification.constraint_checker import ConstraintChecker


@pytest.mark.parametrize('name,value,unit,expected,canonical', [
    ('Energy',375000,'Wh',375,'kWh'), ('Energy',.38,'MWh',380,'kWh'),
    ('Energy',375,'kWh per shift',375,'kWh'), ('DefectRate',.9,'percent',.009,'fraction'),
    ('FatigueIndex',33,'%',.33,'dimensionless'), ('DeadlineGap',.2,'fraction',20,'%'),
    ('AGVTransferTime',.5,'min',30,'s'), ('Throughput',.5,'units/min',30,'u/h'),
])
def test_positive_conversion_retains_comparison_and_input(name,value,unit,expected,canonical):
    c=make_c1();c.hard_constraints=[HardConstraint(name=name,operator='<',limit=value,unit=unit)]
    before=c.model_dump(mode='json');normalized,audit=normalize_contract_units(c)
    bound=normalized.hard_constraints[0]
    assert bound.limit==expected and bound.unit==canonical and bound.operator=='<'
    assert c.model_dump(mode='json')==before
    assert any(v['name']==name and v['old_value']==value for v in audit)
    repeated,changes=normalize_contract_units(normalized)
    assert repeated==normalized and changes==[]


def test_raw_percentage_cannot_pass_nominal_or_sampled_checker():
    c=make_c1();c.hard_constraints=[HardConstraint(name='DefectRate',limit=.9,unit='percent')]
    metrics=SimulationMetrics(throughput_uph=30, defect_rate=.02, noise_db=75, fatigue_index=.3, energy_kwh=350, deadline_gap_pct=40)
    checker=ConstraintChecker()
    raw=checker.check(c,metrics);sampled=checker.check_stochastic(c,[metrics])
    assert not raw.is_feasible and raw.evaluation_errors
    assert not sampled.certified and sampled.evaluation_errors
    normalized,_=normalize_contract_units(c)
    checked=checker.check(normalized,metrics)
    assert not checked.is_feasible and not checked.evaluation_errors
    assert checked.violations


@pytest.mark.parametrize('unit',['kW','mWh','bananas'])
def test_unrecognised_unit_never_falls_back_to_default_contract(unit):
    client=MagicMock();client.is_available=True
    result=make_c1().model_dump(mode='json')
    result['hard_constraints'].append(dict(name='Energy',operator='<=',limit=3,unit=unit))
    client.query_structured.return_value=result
    agent=ElicitationAgent(client=client,use_llm=True)
    agent._elicit_deterministic=MagicMock(side_effect=AssertionError('unsafe fallback'))
    with pytest.raises(UnitCompilationError):agent.elicit('Keep this explicit energy budget.')
    agent._elicit_deterministic.assert_not_called()


def test_actual_llm_entry_compiles_percent_and_keeps_trace():
    client=MagicMock();client.is_available=True
    result=make_c1().model_dump(mode='json')
    result['hard_constraints'].append(dict(name='DefectRate',operator='<=',limit=.9,unit='percent'))
    client.query_structured.return_value=result
    c=ElicitationAgent(client=client,use_llm=True).elicit('Defects at most 0.9 percent.')
    assert c.get_constraint('DefectRate').limit==.009
    assert ContractValidator().validate(c).is_valid
    assert any(v['old_unit']=='percent' for v in c.context['_unit_conversions'])


def test_kpi_target_conversion_and_raw_rejection():
    c=make_c1();c.kpi_targets=[KPITarget(name='Throughput',direction='lower_bound',threshold=.5,unit='units/min')]
    assert not ContractValidator().validate(c).is_valid
    c,_=normalize_contract_units(c)
    assert c.kpi_targets[0].threshold==30 and c.kpi_targets[0].direction.value=='lower_bound'
    assert ContractValidator().validate(c).is_valid


def test_factory_defaults_remain_valid_after_compilation():
    c,_=normalize_contract_units(make_c1_factory())
    assert ContractValidator().validate(c,factory=True).is_valid


def test_unknown_obligation_is_retained_and_rejected():
    c=make_c1();c.hard_constraints.append(HardConstraint(name='UnknownMetric',limit=1))
    normalized,_=normalize_contract_units(c)
    assert normalized.get_constraint('UnknownMetric') is not None
    assert not ContractValidator().validate(normalized).is_valid
