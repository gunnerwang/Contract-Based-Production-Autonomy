#!/usr/bin/env python3
"""Frozen offline semantic-to-planning diagnostic; never substitutes fake LLM results."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cbpa.baselines.elicitation_baselines import TemplateElicitor,RuleBasedElicitor
from cbpa.baselines.optimizer_planner import OptimiserPlanner
from cbpa.layer2_planning.vr_scorer import VRScorer
from cbpa.layer3_verification.constraint_checker import ConstraintChecker
from cbpa.layer1_outcome.contract_validator import ContractValidator
from cbpa.models.contract import make_c1,HardConstraint,KPITarget
from cbpa.physics.cell_evaluator import CellEvaluator

def tuples(c): return {(v.name,v.operator,v.limit) for v in c.hard_constraints}
def kpi(c):
    return next(((v.direction.value,v.threshold) for v in c.kpi_targets if v.name=='Throughput'),None)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'data/validation/semantic_followup_2026-09-11');a=parser.parse_args()
    protocol=json.loads((a.out/'protocol.json').read_text());raw=(a.out/'cases.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==protocol['cases_sha256'],'Frozen cases changed'
    cases=json.loads(raw);rows=[];ev=CellEvaluator(mode='analytical');checker=ConstraintChecker()
    for case in cases:
        ref=make_c1();ref.hard_constraints=[HardConstraint(**v) for v in case['reference']['constraints']]
        ref.kpi_targets=[KPITarget(**case['reference']['kpi'])]
        text=case['intent']+('\nManager clarification: '+case['answer'] if case['answer'] else '')
        for method in ['template','rule','structured_reference']:
            c=ref.model_copy(deep=True) if method=='structured_reference' else (TemplateElicitor() if method=='template' else RuleBasedElicitor()).elicit(text)
            predicted,truth=tuples(c),tuples(ref)
            for seed in protocol['fixed_settings']['seeds']:
                row={'case':case['id'],'category':case['category'],'method':method,'seed':seed,
                     'constraints_exact':predicted==truth,'constraint_precision':len(predicted&truth)/len(predicted) if predicted else 0,
                     'constraint_recall':len(predicted&truth)/len(truth),'kpi_exact':kpi(c)==kpi(ref),
                     'predicted_contract':c.model_dump(mode='json')}
                errors=ContractValidator().validate(c).errors;row['validation_errors']=errors
                start=time.perf_counter();result=None
                if not errors:
                    result=OptimiserPlanner(ev,VRScorer(mode='analytical'),checker,certify=True,seed=seed).plan(c,52)
                row['elapsed_s']=time.perf_counter()-start
                row['certified_candidate_found']=bool(result and result.feasible_found)
                row['reference_compliant']=False;row['target_met']=False
                if result and result.schedule:
                    metrics=ev.evaluate(result.schedule)
                    nominal=checker.check(ref,metrics)
                    stochastic=checker.check_stochastic(ref,ev.evaluate_monte_carlo(result.schedule,n_samples=200,seed=seed),seed=seed)
                    row.update(schedule=result.schedule.model_dump(mode='json'),metrics=metrics.model_dump(mode='json'),
                        predicted_p=result.p_feasible,reference_p=stochastic.p_feasible,
                        reference_compliant=nominal.is_feasible and not stochastic.evaluation_errors and stochastic.p_feasible>=.95)
                    target=case['reference']['kpi']['threshold']
                    row['target_met']=target is None or metrics.throughput_uph>=target
                row['task_success']=row['constraints_exact'] and row['kpi_exact'] and row['reference_compliant'] and row['target_met']
                rows.append(row)
        print(case['id'],flush=True)
    metrics=['constraints_exact','constraint_precision','constraint_recall','kpi_exact','certified_candidate_found','reference_compliant','target_met','task_success']
    summary={m:{k:sum(float(v[k]) for v in rows if v['method']==m)/sum(v['method']==m for v in rows) for k in metrics} for m in ['template','rule','structured_reference']}
    out={'scope':protocol['scope'],'llm_status':'not_run_offline','cases_sha256':protocol['cases_sha256'],'n_cases':len(cases),'n_rows':len(rows),
         'note':'Semantic scores repeat across seeds; effective semantic sample size is 18, not 54 per method. No inferential significance claim.',
         'summary':summary,'rows':rows,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*.py'))}}
    (a.out/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
