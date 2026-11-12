#!/usr/bin/env python3
"""Nominal capacity bound and explicitly counterfactual maintenance envelopes.

Never changes the live scenario or its retained hard contract. Target-aware
search adds a throughput criterion only to a local optimisation copy.
"""
import contextlib,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cbpa.config.scenario import CellConfig
from cbpa.models.contract import make_c1,HardConstraint
from cbpa.physics.cell_evaluator import CellEvaluator
from cbpa.baselines.optimizer_planner import BOUNDS,OptimiserPlanner
from cbpa.layer2_planning.vr_scorer import VRScorer
from cbpa.layer3_verification.constraint_checker import ConstraintChecker
from cbpa.layer4_execution.simulation import ManufacturingCell

def main():
    out=ROOT/'data/validation/capacity_target_audit_2026-09-14';out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/validation/semantic_followup_2026-09-11/cases.json'
    cases=[c for c in json.loads(source.read_text()) if c['category']=='target']
    cfg=CellConfig();bmin=BOUNDS['buf'][0]
    rows=[];bounds=[]
    for case in cases:
        target=case['reference']['kpi']['threshold']
        bounds.append({'case':case['id'],'target_uph':target,'r1_fraction_necessary_nominal':cfg.r1_cycle_time_s/(3600/target-bmin)})
        ref=make_c1();ref.hard_constraints=[HardConstraint(**c) for c in case['reference']['constraints']]
        for cap in [.5,.55,.6,.65]:
            for seed in [42,43,44]:
                ev=CellEvaluator(cfg,mode='analytical',r1_speed_cap=cap);checker=ConstraintChecker()
                search=ref.model_copy(deep=True)
                search.hard_constraints.append(HardConstraint(name='Throughput',operator='>=',limit=target,unit='u/h'))
                result=OptimiserPlanner(ev,VRScorer(mode='analytical'),checker,certify=True,seed=seed).plan(search,52,bounds=dict(BOUNDS,r1=(.3,cap)))
                row={'case':case['id'],'target_uph':target,'r1_fraction_cap':cap,'seed':seed,
                    'nominal_upper_bound_uph':3600/(cfg.r1_cycle_time_s/cap+bmin),'candidate_found':result.feasible_found}
                if result.schedule:
                    row['schedule']=result.schedule.model_dump(mode='json');row['nominal_metrics']=ev.evaluate(result.schedule).model_dump(mode='json');row['sampled_p']=result.p_feasible
                    # Independent diagnostic execution; no claim of live deployment or
                    # a calibrated physical plant. Evaluate observations explicitly.
                    sim=ManufacturingCell(cell=cfg,seed=seed+6).run(result.schedule)
                    row['simulation_metrics']=sim.metrics.model_dump(mode='json')
                    row['observed_hard_constraints_met']=checker.check(ref,sim.metrics).is_feasible
                    row['observed_target_met']=sim.metrics.throughput_uph>=target
                rows.append(row)
    summary={str(cap):{'attempts':sum(r['r1_fraction_cap']==cap for r in rows),
        'certified_target_candidates':sum(r['candidate_found'] for r in rows if r['r1_fraction_cap']==cap),
        'observed_target_and_constraints':sum(r.get('observed_target_met',False) and r.get('observed_hard_constraints_met',False) for r in rows if r['r1_fraction_cap']==cap)} for cap in [.5,.55,.6,.65]}
    result={'scope':__doc__,'current_cap':.5,'nominal_bound_at_current_cap':3600/(cfg.r1_cycle_time_s/.5+bmin),'necessary_capacity_bounds':bounds,
      'bound_scope':'Necessary nominal-model bound from R1 cycle time and minimum buffer. Not a sufficient scheduling condition, not a bound on every stochastic sample or physical production.',
      'counterfactual_scope':'Higher caps assume maintenance restores available capacity; no evidence establishes achievable repair effort or cost. Original runtime cap and historical contracts are unchanged.',
      'summary':summary,'rows':rows,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*.py'))},
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'bounds':bounds,'summary':summary},indent=2))
if __name__=='__main__':main()
