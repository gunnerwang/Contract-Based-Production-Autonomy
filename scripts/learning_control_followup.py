#!/usr/bin/env python3
"""Matched offline L5-feedback ablation: fixed contracts, no live LLM calls.

Ten paired seed sequences, ten shifts each. All conditions preserve writes to
memory; only read access for planning/reasoning is disabled in the control.
"""
import contextlib,copy,hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cbpa.models.contract import make_c1
from cbpa.config.scenario import ScenarioConfig
from cbpa.layer5_adaptation.learning_store import LearningStore
from cbpa.runner.experiment import CBPAExperiment
from cbpa.runner.lifecycle import DeploymentBlocked
from cbpa.layer3_verification.constraint_checker import ConstraintChecker


class FeedbackStore(LearningStore):
    def __init__(self,enabled):
        super().__init__();self.enabled=enabled
    def get_similar(self,disturbance_type):
        return super().get_similar(disturbance_type) if self.enabled else []


class FixedElicitor:
    use_llm=False
    def elicit(self,intent,contract_name='C1'):
        c=make_c1();c.name=contract_name;return c
    def elicit_with_refinement(self,intent,contract_name='C1'):
        return self.elicit(intent,contract_name),[]


def sequences():
    result=[]
    for replicate in range(10):
        demands=[10,15,20,25,30]*2
        random.Random(20260914+replicate).shuffle(demands)
        result.append({'replicate':replicate,'shifts':[{'shift':i+1,'demand_spike_pct':d,'mc_seed':4200+replicate*100+i*7} for i,d in enumerate(demands)]})
    return result


def main():
    out=ROOT/'data/validation/learning_control_2026-09-14';out.mkdir(exist_ok=True)
    protocol={'scope':__doc__,'sequences':sequences(),'arms':['feedback_on','feedback_off'],
      'attempts':200,'new_model_calls':0,'contract':make_c1().model_dump(mode='json'),
      'configuration':{'optimiser_anchor':True,'enable_macro_phase':True,'use_simulation':True},
      'search':'Identical five anchor variants plus certified optimiser anchor where enabled; optimiser uses existing fixed seed 42, grid 7, refinement 300, MC 200 and threshold .95. Feedback may change variants, not resource allowance.',
      'seed_policy':'Identical per-shift mc_seed across arms; execution uses existing mc_seed+phase. Ten seed sequences are paired analysis units; shifts within a sequence are dependent.',
      'failure_policy':'Blocked/error shifts retain denominators and histories; no replacement, relaxation or new manager answers. The next scheduled shift is a separate attempt under its frozen configuration.',
      'outcomes':'Completion; nominal observations against retained hard constraints; phase-5 throughput/V-R/energy/deadline-gap in pairs where both phase-5 observations exist; candidate changes and feedback exposure. No population-level superiority or equivalence claim.',
      'limitations':'Deterministic planning and reasoning with fixed structured contracts, not live LLM learning, model fine-tuning, human decisions, independent plant validation or a representative disturbance distribution.',
      'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*.py'))},
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Create a new experiment version'
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[]
    for seq in protocol['sequences']:
        for arm in protocol['arms']:
            memory=FeedbackStore(arm=='feedback_on')
            for spec in seq['shifts']:
                folder=out/f"r{seq['replicate']:02}_{arm}_s{spec['shift']:02}";folder.mkdir(exist_ok=True)
                exp=CBPAExperiment(config=ScenarioConfig(**protocol['configuration'],demand_spike_pct=spec['demand_spike_pct']),use_llm=False,mc_seed_base=spec['mc_seed'])
                exp.elicitation=FixedElicitor();exp.learning=memory
                phases=[];captured={};before=memory.total_records
                original=exp.planner.generate_balanced
                def capture(*args,**kwargs):
                    captured['learning_bias']=copy.deepcopy(kwargs.get('learning_bias'))
                    candidates=original(*args,**kwargs)
                    captured['candidates']=[v.model_dump(mode='json') for v in candidates]
                    return candidates
                exp.planner.generate_balanced=capture
                row={'replicate':seq['replicate'],'arm':arm,**spec,'status':'completed','records_before':before}
                try:
                    with (folder/'run.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                        exp.run_all_phases(on_phase_complete=phases.append)
                except DeploymentBlocked as exc:row.update(status='blocked',error=str(exc))
                except Exception as exc:row.update(status='error',error=f'{type(exc).__name__}: {exc}')
                finally:exp.shutdown()
                exp.audit.export_json(folder/'audit.json')
                row['completed_phases']=[v.phase for v in phases];row['records_after']=memory.total_records
                row['feedback_nonempty']=bool(captured.get('learning_bias'));row['planning_trace']=captured
                observations=[]
                for p in phases:
                    if p.phase in (1,5,6) and p.metrics is not None:
                        observations.append({'phase':p.phase,'metrics':p.metrics.model_dump(mode='json'),
                            'schedule':p.schedule.model_dump(mode='json'),'vr':p.vr_score.vr_score,
                            'reference_hard_constraints_met':ConstraintChecker().check(make_c1(),p.metrics).is_feasible,
                            'contract_hard_constraints':ConstraintChecker.constraint_spec(p.contract)})
                row['observations']=observations
                row['complete_reference_compliant']=row['status']=='completed' and len(observations)==3 and all(v['reference_hard_constraints_met'] for v in observations)
                row['phase5']=next((v for v in observations if v['phase']==5),None)
                (folder/'result.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
            print('completed sequence',seq['replicate'],arm,flush=True)
    pairs=[]
    for seq in protocol['sequences']:
        for spec in seq['shifts']:
            on=next(r for r in rows if r['replicate']==seq['replicate'] and r['shift']==spec['shift'] and r['arm']=='feedback_on')
            off=next(r for r in rows if r['replicate']==seq['replicate'] and r['shift']==spec['shift'] and r['arm']=='feedback_off')
            assert on['mc_seed']==off['mc_seed'] and on['demand_spike_pct']==off['demand_spike_pct']
            p={'replicate':seq['replicate'],'shift':spec['shift'],'on_status':on['status'],'off_status':off['status'],
                'candidates_differ':on['planning_trace'].get('candidates')!=off['planning_trace'].get('candidates'),
                'candidate_count_on':len(on['planning_trace'].get('candidates',[])),'candidate_count_off':len(off['planning_trace'].get('candidates',[])),
                'phase5_pair_observed':bool(on['phase5'] and off['phase5'])}
            if p['phase5_pair_observed']:
                a,b=on['phase5'],off['phase5'];p['schedule_different']=a['schedule']!=b['schedule']
                p['deltas']={k:a['metrics'][k]-b['metrics'][k] for k in ['throughput_uph','energy_kwh','deadline_gap_pct']}
                p['deltas']['vr']=a['vr']-b['vr']
            pairs.append(p)
    summaries={}
    for arm in protocol['arms']:
        rr=[r for r in rows if r['arm']==arm]
        summaries[arm]={'attempts':len(rr),'completed':sum(r['status']=='completed' for r in rr),
            'blocked':sum(r['status']=='blocked' for r in rr),'errors':sum(r['status']=='error' for r in rr),
            'complete_reference_compliant':sum(r['complete_reference_compliant'] for r in rr),
            'nonempty_feedback':sum(r['feedback_nonempty'] for r in rr)}
    paired=[p for p in pairs if p['phase5_pair_observed']]
    summaries['paired']={'pairs':len(pairs),'phase5_pairs_observed':len(paired),
       'candidate_sets_different':sum(p['candidates_differ'] for p in pairs),
       'candidate_counts_equal':sum(p['candidate_count_on']==p['candidate_count_off'] for p in pairs),
       'selected_schedules_different':sum(p['schedule_different'] for p in paired),
       'throughput_improved':sum(p['deltas']['throughput_uph']>0 for p in paired),
       'throughput_tied':sum(p['deltas']['throughput_uph']==0 for p in paired),
       'throughput_degraded':sum(p['deltas']['throughput_uph']<0 for p in paired),
       'mean_deltas':{k:sum(p['deltas'][k] for p in paired)/len(paired) if paired else None for k in ['throughput_uph','energy_kwh','deadline_gap_pct','vr']}}
    sequence_summaries=[]
    for seq in protocol['sequences']:
        pp=[p for p in paired if p['replicate']==seq['replicate']]
        sequence_summaries.append({'replicate':seq['replicate'],'matched_observations':len(pp),'mean_deltas':{k:sum(p['deltas'][k] for p in pp)/len(pp) if pp else None for k in ['throughput_uph','energy_kwh','deadline_gap_pct','vr']}})
    (out/'results.json').write_text(json.dumps({'protocol':protocol,'summary':summaries,'sequence_summaries':sequence_summaries,'pairs':pairs,'rows':rows},indent=2)+'\n')
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
