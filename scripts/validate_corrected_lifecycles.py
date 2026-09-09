#!/usr/bin/env python3
"""Controlled no-LLM lifecycle validation, independent of historical batches.

Two candidate pools (anchors; anchors plus certified optimizer) in each case.
Cell runs use SimPy; factory runs use the analytical evaluator, no services.
These four executions test corrected control flow, not LLM or field performance.
"""
import contextlib
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cbpa.config.scenario import ScenarioConfig, FactoryScenarioConfig
from cbpa.runner.experiment import CBPAExperiment
from cbpa.runner.factory_experiment import FactoryExperiment
from cbpa.runner.lifecycle import DeploymentBlocked


def main():
    out=ROOT/'data/validation/verification_completeness'
    out.mkdir(parents=True,exist_ok=True)
    summary={'scope':'Four controlled no-LLM lifecycles: one per case/pool. Cell SimPy; factory analytical; no external services. Not historical LLM reruns or a comparative performance estimate.','seed':42,'runs':{}}
    for factory in [False,True]:
        for optimizer in [False,True]:
            name=('factory' if factory else 'cell')+('_optimizer' if optimizer else '_anchors')
            folder=out/name;folder.mkdir(exist_ok=True)
            config=(FactoryScenarioConfig if factory else ScenarioConfig)(optimiser_anchor=optimizer,enable_macro_phase=True,use_simulation=not factory)
            exp=FactoryExperiment(config=config,mode='analytical',use_llm=False) if factory else CBPAExperiment(config=config,use_llm=False)
            completed=[]
            status='completed';reason=None
            try:
                with (folder/'execution.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    exp.run_all_phases(on_phase_complete=completed.append)
            except DeploymentBlocked as exc:
                status='blocked';reason=str(exc)
            finally:exp.shutdown()
            exp.audit.export_json(folder/'audit_trail.json')
            phases=[]
            for phase in completed:
                metrics=phase.factory_metrics if factory else phase.metrics
                phases.append({'phase':phase.phase,'metrics':metrics.model_dump() if metrics else None})
            blocked=[e.phase for e in exp.audit.entries if e.action=='deployment_blocked']
            dispatched=[e.phase for e in exp.audit.entries if e.action in {'schedule_deployed','next_shift_prepared'}]
            assert not set(blocked)&set(dispatched),(name,blocked,dispatched)
            if name=='cell_anchors':
                assert status=='blocked' and blocked==[5]
                assert exp.learning.total_records==0
            else:assert status=='completed',(name,reason)
            summary['runs'][name]={'status':status,'completed_phases':[p.phase for p in completed],'blocked_phases':blocked,'dispatch_or_preparation_phases':dispatched,'reason':reason,'phases':phases,'experience_records':exp.learning.total_records}
    summary['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*.py'))}
    (out/'lifecycle_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:{'status':v['status'],'completed_phases':v['completed_phases'],'blocked_phases':v['blocked_phases']} for k,v in summary['runs'].items()},indent=2))

if __name__=='__main__':main()
