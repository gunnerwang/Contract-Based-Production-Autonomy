#!/usr/bin/env python3
"""Full six-phase continuations from frozen parser outputs, no new LLM calls.

Only Layer-1 inputs differ; all downstream agents are deterministic with the
same optimiser pool. This is not a rerun of the historical all-agent LLM batch.
"""
import contextlib,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cbpa.models.contract import OutcomeContract,HardConstraint,make_c1
from cbpa.config.scenario import ScenarioConfig
from cbpa.runner.experiment import CBPAExperiment
from cbpa.runner.lifecycle import DeploymentBlocked
from cbpa.layer3_verification.constraint_checker import ConstraintChecker

class RecordedElicitor:
    def __init__(self,contract): self.contract=contract
    def elicit(self,intent,contract_name='C1'):
        c=self.contract.model_copy(deep=True);c.name=contract_name;return c
    def elicit_with_refinement(self,intent,contract_name='C1'):
        return self.elicit(intent,contract_name),[]

def main():
    old=ROOT/'data/validation/semantic_followup_2026-09-11'
    model=ROOT/'data/validation/semantic_model_followup_2026-09-13'
    out=ROOT/'data/validation/semantic_lifecycle_followup_2026-09-13';out.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((old/'results.json').read_text());new=json.loads((model/'results.json').read_text())
    cases={c['id']:c for c in json.loads((old/'cases.json').read_text())}
    rows=[r for r in baseline['rows'] if r['method'] in ('rule','structured_reference')]+new['rows']
    protocol={'scope':__doc__,'n_attempts':len(rows),'conditions':['rule','structured_reference','llm','llm_rule_union'],
      'source_inputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [old/'cases.json',old/'results.json',model/'results.json']},
      'simulation':'cell SimPy, optimiser anchors, original scripted disturbances and escalation, fixed per-run seed; generated contract retained through context-only transitions',
      'failure_policy':'No replacement of invalid parser output; blocked and runtime-stop attempts retain full denominator and audits',
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    runs=[]
    for row in rows:
        name=f"{row['method']}_{row['case']}_{row['seed']}";folder=out/name;folder.mkdir(exist_ok=True)
        result={'case':row['case'],'method':row['method'],'seed':row['seed'],'status':'invalid_input',
          'semantic_exact':row['constraints_exact'] and row['kpi_exact'],'reference_compliant_deployments':0,'deployments_evaluated':0,
          'completed_phases':[],'refused_phases':[]}
        if row.get('predicted_contract') is None or row.get('validation_errors'):
            result['reason']=row.get('validation_errors') or row.get('error');runs.append(result);continue
        contract=OutcomeContract.model_validate(row['predicted_contract'])
        config=ScenarioConfig(optimiser_anchor=True,enable_macro_phase=True,use_simulation=True)
        exp=CBPAExperiment(config=config,use_llm=False,mc_seed_base=row['seed']);exp.elicitation=RecordedElicitor(contract)
        phases=[];result['status']='completed'
        try:
            with (folder/'execution.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                exp.run_all_phases(on_phase_complete=phases.append)
        except DeploymentBlocked as exc:
            result['status']='blocked';result['reason']=str(exc)
        except Exception as exc:
            result['status']='error';result['reason']=type(exc).__name__+': '+str(exc)
        finally: exp.shutdown()
        exp.audit.export_json(folder/'audit_trail.json')
        result['completed_phases']=[p.phase for p in phases]
        refusals=[e.phase for e in exp.audit.entries if e.action in ('deployment_blocked','runtime_execution_blocked')]
        dispatched=[e.phase for e in exp.audit.entries if e.action in ('schedule_deployed','next_shift_prepared')]
        assert not refusals or not any(p>min(refusals) for p in dispatched)
        assert not set(e.phase for e in exp.audit.entries if e.action=='deployment_blocked')&set(dispatched)
        result['refused_phases']=refusals;result['dispatch_phases']=dispatched
        ref=make_c1();ref.hard_constraints=[HardConstraint(**v) for v in cases[row['case']]['reference']['constraints']]
        evaluated=[]
        for p in phases:
            if p.phase not in (1,5,6) or p.metrics is None:continue
            check=ConstraintChecker().check(ref,p.metrics)
            evaluated.append({'phase':p.phase,'reference_nominal_feasible':check.is_feasible,'metrics':p.metrics.model_dump(mode='json')})
        result['deployment_observations']=evaluated
        result['deployments_evaluated']=len(evaluated)
        result['reference_compliant_deployments']=sum(v['reference_nominal_feasible'] for v in evaluated)
        result['complete_reference_compliant']=result['status']=='completed' and bool(evaluated) and all(v['reference_nominal_feasible'] for v in evaluated)
        result['complete_semantic_and_reference_compliant']=result['complete_reference_compliant'] and result['semantic_exact']
        runs.append(result)
    summary={}
    for method in protocol['conditions']:
        rr=[r for r in runs if r['method']==method]
        summary[method]={'attempts':len(rr),'completed':sum(r['status']=='completed' for r in rr),
          'blocked':sum(r['status']=='blocked' for r in rr),'invalid_or_error':sum(r['status'] in ('invalid_input','error') for r in rr),
          'complete_reference_compliant':sum(r.get('complete_reference_compliant',False) for r in rr),
          'complete_semantic_and_reference_compliant':sum(r.get('complete_semantic_and_reference_compliant',False) for r in rr)}
    (out/'results.json').write_text(json.dumps({'protocol':protocol,'summary':summary,'runs':runs},indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
