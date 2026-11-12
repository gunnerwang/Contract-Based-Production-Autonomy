#!/usr/bin/env python3
"""New offline lifecycles; historical validations are never overwritten."""
import contextlib,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cbpa.config.scenario import ScenarioConfig,FactoryScenarioConfig
from cbpa.runner.experiment import CBPAExperiment
from cbpa.runner.factory_experiment import FactoryExperiment
from cbpa.runner.lifecycle import DeploymentBlocked

CONFIRMED_NO_CAP={
    'energy_kwh_cap':{'text':'Keep energy as a soft guideline; do not add a hard energy cap.', 'patch':{'disposition':'retain'}},
    'shift_hours_overflow':{'text':'Keep the existing shift; no overtime.', 'patch':{'context':{'overtime_allowed':False}}}}

def main():
    out=ROOT/'data/validation/clarification_lifecycles_2026-09-11';out.mkdir(parents=True,exist_ok=True)
    runs=[]
    for seed in [42,43,44]:
        for case in ['cell_scripted_cap','cell_confirmed_no_cap','factory']:
            for optimizer in [False,True]:
                name=f'{case}_{"optimizer" if optimizer else "anchors"}_{seed}'
                folder=out/name;folder.mkdir(exist_ok=True)
                factory=case=='factory'
                config=(FactoryScenarioConfig if factory else ScenarioConfig)(optimiser_anchor=optimizer,enable_macro_phase=True,use_simulation=not factory)
                exp=FactoryExperiment(config=config,use_llm=False,mode='analytical') if factory else CBPAExperiment(config=config,use_llm=False)
                exp.mc_seed_base=seed
                if case=='cell_confirmed_no_cap':
                    original=exp.elicitation.elicit_with_refinement
                    exp.elicitation.elicit_with_refinement=lambda intent,contract_name,fn=original: fn(intent,contract_name,confirmed_answers=CONFIRMED_NO_CAP)
                phases=[];status='completed';reason=None
                try:
                    with (folder/'execution.log').open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
                        exp.run_all_phases(on_phase_complete=phases.append)
                except DeploymentBlocked as exc:
                    status='blocked';reason=str(exc)
                finally: exp.shutdown()
                exp.audit.export_json(folder/'audit_trail.json')
                entries=exp.audit.entries
                refusals=[e.phase for e in entries if e.action in {'deployment_blocked','runtime_execution_blocked'}]
                dispatched=[e.phase for e in entries if e.action in {'schedule_deployed','next_shift_prepared'}]
                # No later phase may execute after refusal. A runtime refusal can
                # follow deployment within its phase; a predispatch refusal cannot.
                assert not refusals or not any(p>min(refusals) for p in dispatched)
                pre=[e.phase for e in entries if e.action=='deployment_blocked']
                assert not set(pre)&set(dispatched)
                if case=='cell_scripted_cap': assert status=='blocked' and pre==[1] and not dispatched
                runs.append({'name':name,'case':case,'optimizer':optimizer,'seed':seed,'status':status,'reason':reason,
                    'completed_phases':[p.phase for p in phases],'refused_phases':refusals,'dispatch_phases':dispatched,
                    'manager_answers':CONFIRMED_NO_CAP if case=='cell_confirmed_no_cap' else 'existing cell presets or factory contract',
                    'experience_records':exp.learning.total_records,
                    'phases':[{'phase':p.phase,'metrics':(p.factory_metrics if factory else p.metrics).model_dump(mode='json') if (p.factory_metrics if factory else p.metrics) else None} for p in phases]})
                print(name,status,refusals,flush=True)
    summary={'scope':'18 new no-LLM lifecycles, three MC seeds, cell SimPy and factory analytical. Explicitly different manager answers are separate conditions; not historical LLM reruns or field validation.',
        'runs':runs,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*.py'))}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
