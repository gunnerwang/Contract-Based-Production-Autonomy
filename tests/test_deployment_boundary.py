"""Regression tests for rejecting candidates before any runner dispatch."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import os
import subprocess
import sys

import pytest
from cbpa.models.contract import make_c1
from cbpa.layer3_verification.constraint_checker import ConstraintChecker
from cbpa.models.metrics import FeasibilityReport, StochasticFeasibilityReport
from cbpa.models.schedule import Schedule
from cbpa.runner.experiment import CBPAExperiment
from cbpa.runner.factory_experiment import FactoryExperiment
from cbpa.runner.lifecycle import DeploymentBlocked


def certificate(exp, feasible=True, probability=1.0, samples=200):
    contract = make_c1()
    spec = ConstraintChecker.constraint_spec(contract)
    return exp.cert_issuer.issue(
        "candidate", FeasibilityReport(is_feasible=feasible, checked_constraints=spec,
            constraint_results={c.name: feasible for c in contract.hard_constraints}), contract,
        StochasticFeasibilityReport(p_feasible=probability, n_samples=samples, seed=42, checked_constraints=spec),
    )


@pytest.mark.parametrize("feasible,p,samples", [(False,1,200),(True,.585,200),(True,.945,200),(True,1,0),(True,1,1),(True,float('nan'),200)])
def test_final_boundary_rejects_invalid_report(feasible,p,samples):
    exp=CBPAExperiment(use_llm=False)
    with pytest.raises(DeploymentBlocked):
        exp._require_deployment_certificate(certificate(exp,feasible,p,samples),phase=1)
    assert exp.audit.entries[-1].action == "deployment_blocked"
    assert exp._guard_enforcer is None


def test_missing_or_inconsistent_stochastic_report_is_not_authorization():
    exp=CBPAExperiment(use_llm=False)
    cert=certificate(exp)
    cert.stochastic_report=None
    with pytest.raises(DeploymentBlocked): exp._require_deployment_certificate(cert,1)
    cert=certificate(exp)
    cert.stochastic_report.p_feasible=.1
    with pytest.raises(DeploymentBlocked): exp._require_deployment_certificate(cert,1)


def test_exact_threshold_authorizes_and_refreshes_guards():
    exp=CBPAExperiment(use_llm=False)
    stale=object()
    exp._guard_enforcer=stale
    exp._require_deployment_certificate(certificate(exp,probability=.95),1)
    assert exp._guard_enforcer is not None and exp._guard_enforcer is not stale


@pytest.mark.parametrize("kind", ["cell","factory"])
@pytest.mark.parametrize("phase", [1,5,6])
@pytest.mark.parametrize("nominal,p", [(False,1),(True,.585)])
def test_empty_acceptance_set_halts_deployment_phases(kind,phase,nominal,p):
    exp=CBPAExperiment(use_llm=False) if kind=='cell' else FactoryExperiment(use_llm=False)
    candidate=Schedule(name='bad',r1_speed_fraction=.5,r2_speed_fraction=.5,human_cycle_rate_multiplier=.8,buffer_time_s=3,demand_target_uph=52)
    rank=SimpleNamespace(selected='bad',non_dominated=['bad'],dominated=[])
    scores={'bad':SimpleNamespace(vr_score=.5,total_throughput_uph=40)}
    feas={'bad':FeasibilityReport(is_feasible=nominal)}
    exp.evaluator.evaluate_monte_carlo=MagicMock(return_value=[])
    check=MagicMock(return_value=SimpleNamespace(p_feasible=p,evaluation_errors=[]))
    exp.checker.check_stochastic=check
    exp.checker.check_factory_stochastic=check
    with pytest.raises(DeploymentBlocked): exp._certification_gate([candidate],rank,scores,feas,make_c1(),phase)
    if not nominal: exp.evaluator.evaluate_monte_carlo.assert_not_called()


@pytest.mark.parametrize('kind',['cell','factory'])
def test_certified_alternative_selected_and_rejected_replan_retained(kind):
    exp=CBPAExperiment(use_llm=False) if kind=='cell' else FactoryExperiment(use_llm=False)
    first=Schedule(name='first',r1_speed_fraction=.5,r2_speed_fraction=.5,human_cycle_rate_multiplier=.8,buffer_time_s=3,demand_target_uph=52)
    second=first.model_copy(update={'name':'second'})
    rank=SimpleNamespace(selected='first',non_dominated=['first'],dominated=['second'])
    scores={n:SimpleNamespace(vr_score=.5,total_throughput_uph=40) for n in ['first','second']}
    feas={n:FeasibilityReport(is_feasible=True) for n in scores}
    exp.evaluator.evaluate_monte_carlo=MagicMock(side_effect=lambda s,**kw:s.name)
    check=MagicMock(side_effect=lambda c,s,**kw:SimpleNamespace(p_feasible=.585 if s=='first' else 1.0,evaluation_errors=[]))
    exp.checker.check_stochastic=check
    exp.checker.check_factory_stochastic=check
    chosen,info=exp._certification_gate([first,second],rank,scores,feas,make_c1(),5)
    assert chosen.name=='second' and info['certified']
    expected_seed=44 if kind=='factory' else 42
    assert all(c.kwargs['seed']==expected_seed for c in exp.evaluator.evaluate_monte_carlo.call_args_list)
    feas={n:FeasibilityReport(is_feasible=False) for n in scores}
    chosen,info=exp._certification_gate([first,second],rank,scores,feas,make_c1(),3)
    assert chosen.name=='first' and not info['certified']
    assert not any(e.action=='schedule_deployed' for e in exp.audit.entries)


@pytest.mark.parametrize('target',['FS1','FS3','FS4'])
def test_final_factory_rejection_prevents_dispatch_and_prestage(target):
    exp=FactoryExperiment(use_llm=False,mode='analytical')
    exp._orchestrator=MagicMock()
    original=exp.cert_issuer.issue
    def issue(name,*args,**kwargs):
        cert=original(name,*args,**kwargs)
        if name==target:
            cert.is_certified=False
            exp._orchestrator.reset_mock()
        return cert
    exp.cert_issuer.issue=issue
    with pytest.raises(DeploymentBlocked,match=target): exp.run_all_phases()
    exp._orchestrator.deploy_factory_schedule.assert_not_called()
    exp._orchestrator.sync_factory_schedule.assert_not_called()
    phase={'FS1':1,'FS3':5,'FS4':6}[target]
    assert not any(e.phase==phase and e.action in {'schedule_deployed','next_shift_ready'} for e in exp.audit.entries)


def test_final_cell_rejection_prevents_execution():
    exp=CBPAExperiment(use_llm=False)
    exp._execute_integrated=MagicMock()
    exp._execute_simulation=MagicMock()
    original=exp.cert_issuer.issue
    def issue(*args,**kwargs):
        cert=original(*args,**kwargs)
        cert.is_certified=False
        return cert
    exp.cert_issuer.issue=issue
    with pytest.raises(DeploymentBlocked): exp.run_all_phases()
    exp._execute_integrated.assert_not_called()
    exp._execute_simulation.assert_not_called()
    assert not any(e.action=='schedule_deployed' for e in exp.audit.entries)


def test_factory_sampling_is_identical_across_python_hash_seeds():
    code='''import contextlib,io,json
from cbpa.runner.factory_experiment import FactoryExperiment
with contextlib.redirect_stdout(io.StringIO()):
 e=FactoryExperiment(use_llm=False)
 p=e.run_single_phase(1)
 samples=e.evaluator.evaluate_monte_carlo(p.factory_schedule,n_samples=20,seed=42)
print(json.dumps([s.model_dump() for s in samples],sort_keys=True))
'''
    results=[subprocess.check_output([sys.executable,'-c',code],env={**os.environ,'PYTHONHASHSEED':str(seed)},text=True) for seed in [1,2]]
    assert results[0]==results[1]
