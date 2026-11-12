#!/usr/bin/env python3
"""Prospective model arm on frozen diagnostic cases; raw outputs and failures retained.

One independent model generation per intent, reused across three MC seeds.
The union arm uses the same generation, not a second model call.
"""
import argparse,concurrent.futures,copy,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from cbpa.llm.client import ClaudeClient,_extract_json
from cbpa.llm.prompts import ELICITATION_TOOL_SCHEMA
from cbpa.models.contract import OutcomeContract,HardConstraint,KPITarget,PriorityLevel,make_c1
from cbpa.baselines.elicitation_baselines import RuleBasedElicitor
from cbpa.baselines.optimizer_planner import OptimiserPlanner
from cbpa.layer1_outcome.contract_validator import ContractValidator
from cbpa.layer2_planning.vr_scorer import VRScorer
from cbpa.layer3_verification.constraint_checker import ConstraintChecker
from cbpa.physics.cell_evaluator import CellEvaluator
from semantic_followup import tuples,kpi

SYSTEM = """You compile a manager's natural language production request into a typed outcome contract.
Use the supplied plant ontology: hard constraints FatigueIndex (dimensionless), Noise (dB),
CyberRiskLevel (dimensionless), Energy (kWh per shift), DefectRate (fraction), Throughput (u/h),
DeadlineGap (percent). Supported KPI names: Throughput, DefectRate, ChangeoverTime, Energy.
Mandatory plant ceilings are FatigueIndex <= 0.4, Noise <= 80 dB and CyberRiskLevel <= 2.
A manager may request stricter constraints; never relax these ceilings. The shift is 8 hours.
Preserve the actual requirement, including comparison operator, units, target magnitudes,
negation and the ordering of manager updates. A later explicit clarification revises the earlier
request for the same field. Do not convert examples, rejected suggestions or soft goals into
hard requirements. If throughput is to be maximised, use maximize with no threshold; preserve
an explicit numerical throughput target as a lower_bound KPI. Do not invent additional KPIs.
Use priority labels from HumanWellbeing, Safety, Quality, Throughput, Cost, Flexibility.
If priorities are not specified, use HumanWellbeing, Safety, Quality, Throughput.
Return ONLY a JSON object matching the supplied schema. No tools or files are needed."""

def merge_union(llm,rule):
    """Union retains false positives; merge only compatible bounds, never drop obligations."""
    result=llm.model_copy(deep=True);cons={c.name:c.model_copy(deep=True) for c in result.hard_constraints}
    if len(cons)!=len(result.hard_constraints): raise ValueError('Duplicate LLM constraint')
    for c in rule.hard_constraints:
        old=cons.get(c.name)
        if old is None: cons[c.name]=c.model_copy(deep=True);continue
        if old.unit and c.unit and old.unit!=c.unit: raise ValueError('Incompatible units in union')
        if old.operator in ('<','<=') and c.operator in ('<','<='):
            if c.limit<old.limit or (c.limit==old.limit and c.operator=='<'): cons[c.name]=c.model_copy(deep=True)
        elif old.operator in ('>','>=') and c.operator in ('>','>='):
            if c.limit>old.limit or (c.limit==old.limit and c.operator=='>'): cons[c.name]=c.model_copy(deep=True)
        elif (old.operator,old.limit)!=(c.operator,c.limit): raise ValueError('Incompatible comparisons in union')
    result.hard_constraints=list(cons.values());return result

def parse(raw):
    import jsonschema
    value=_extract_json(raw);jsonschema.validate(value,ELICITATION_TOOL_SCHEMA)
    return OutcomeContract(name='semantic_model',kpi_targets=[KPITarget(**v) for v in value['kpi_targets']],
        hard_constraints=[HardConstraint(**v) for v in value['hard_constraints']],
        priority_order=[PriorityLevel(v) for v in value['priority_order']],
        context=value.get('context',{}),assumptions=value.get('assumptions',{}))

def infer(case,model,out):
    path=out/(case['id']+'.json')
    if path.exists(): return json.loads(path.read_text())
    message=case['intent']+('\nManager clarification: '+case['answer'] if case['answer'] else '')
    prompt=message+'\nOutput schema:\n'+json.dumps(ELICITATION_TOOL_SCHEMA,sort_keys=True)
    record={'id':case['id'],'model':model,'system':SYSTEM,'user_message':prompt,'status':'error','raw_response':None}
    start=time.perf_counter()
    try:
        record['raw_response']=ClaudeClient(model=model).query_text(SYSTEM,prompt)
        record['contract']=parse(record['raw_response']).model_dump(mode='json');record['status']='ok'
    except Exception as exc: record['error']=type(exc).__name__+': '+str(exc)[:500]
    record['latency_s']=time.perf_counter()-start;path.write_text(json.dumps(record,indent=2)+'\n')
    print(case['id'],record['status'],round(record['latency_s'],1),flush=True)
    return record

def evaluate(case,c,method,seed,error=None):
    ref=make_c1();ref.hard_constraints=[HardConstraint(**v) for v in case['reference']['constraints']]
    ref.kpi_targets=[KPITarget(**case['reference']['kpi'])]
    truth=tuples(ref);predicted=tuples(c) if c else set()
    row={'case':case['id'],'category':case['category'],'method':method,'seed':seed,
        'constraints_exact':predicted==truth,'constraint_precision':len(predicted&truth)/len(predicted) if predicted else 0,
        'constraint_recall':len(predicted&truth)/len(truth),'kpi_exact':c is not None and kpi(c)==kpi(ref),
        'certified_candidate_found':False,'reference_compliant':False,'target_met':False,'error':error,
        'predicted_contract':c.model_dump(mode='json') if c else None}
    errors=ContractValidator().validate(c).errors if c else ['model/union failure']
    if c:
        for name,ceiling in [('FatigueIndex',.4),('Noise',80),('CyberRiskLevel',2)]:
            hc=c.get_constraint(name)
            if hc is None or hc.operator not in ['<=','<','=='] or hc.limit>ceiling: errors.append('missing or relaxed plant ceiling: '+name)
    row['validation_errors']=errors
    if not errors:
        ev=CellEvaluator(mode='analytical');checker=ConstraintChecker()
        result=OptimiserPlanner(ev,VRScorer(mode='analytical'),checker,certify=True,seed=seed).plan(c,52)
        row['certified_candidate_found']=result.feasible_found
        if result.schedule:
            m=ev.evaluate(result.schedule);report=checker.check(ref,m)
            st=checker.check_stochastic(ref,ev.evaluate_monte_carlo(result.schedule,n_samples=200,seed=seed),seed=seed)
            row.update(schedule=result.schedule.model_dump(mode='json'),metrics=m.model_dump(mode='json'),
                predicted_p=result.p_feasible,reference_p=st.p_feasible,
                reference_compliant=report.is_feasible and not st.evaluation_errors and st.p_feasible>=.95)
            target=case['reference']['kpi']['threshold'];row['target_met']=target is None or m.throughput_uph>=target
    row['task_success']=row['constraints_exact'] and row['kpi_exact'] and row['reference_compliant'] and row['target_met']
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='claude-sonnet-4-6');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--score-only',action='store_true');a=p.parse_args()
    source=ROOT/'data/validation/semantic_followup_2026-09-11';out=ROOT/'data/validation/semantic_model_followup_2026-09-13';out.mkdir(parents=True,exist_ok=True)
    raw=(source/'cases.json').read_bytes();base=json.loads((source/'protocol.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==base['cases_sha256'];cases=json.loads(raw)
    protocol={'model':a.model,'n_intents':18,'calls_per_intent':1,'mc_seeds':[42,43,44],'cases_sha256':base['cases_sha256'],
        'system':SYSTEM,'schema':ELICITATION_TOOL_SCHEMA,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'baseline_results_sha256':hashlib.sha256((source/'results.json').read_bytes()).hexdigest(),
        'scope':'Authored frozen diagnostic; no independent annotation. Shared model output across LLM-only and union arms and three MC seeds. No retries or replacement of failed responses.',
        'union':'Union hard constraints; tighter compatible bound, strict wins at equality, incompatible comparisons refuse. Other fields retained from LLM.',
        'cost':'Wall-clock latency recorded; token/billing cost unavailable through current SDK transport.',
        'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'src').rglob('*.py'))}}
    dest=out/'protocol.json'
    if dest.exists(): assert json.loads(dest.read_text())==protocol,'Protocol/source changed after freeze; create a new experiment version'
    else: dest.write_text(json.dumps(protocol,indent=2)+'\n')
    if a.prepare_only: print('Frozen protocol',dest);return
    records=out/'responses';records.mkdir(exist_ok=True)
    if not a.score_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(lambda c:infer(c,a.model,records),cases))
    rows=[]
    for case in cases:
        path=records/(case['id']+'.json');record=json.loads(path.read_text()) if path.exists() else {'status':'not_run','error':'not run'}
        c=OutcomeContract.model_validate(record['contract']) if record['status']=='ok' else None
        text=case['intent']+('\nManager clarification: '+case['answer'] if case['answer'] else '')
        for method in ['llm','llm_rule_union']:
            pred=c;error=record.get('error')
            if c is not None and method=='llm_rule_union':
                try: pred=merge_union(c,RuleBasedElicitor().elicit(text))
                except Exception as exc: pred=None;error=str(exc)
            for seed in [42,43,44]: rows.append(evaluate(case,pred,method,seed,error))
    metrics=['constraints_exact','constraint_precision','constraint_recall','kpi_exact','certified_candidate_found','reference_compliant','target_met','task_success']
    summary={m:{k:sum(float(r[k]) for r in rows if r['method']==m)/54 for k in metrics} for m in ['llm','llm_rule_union']}
    result={'protocol_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'summary':summary,'rows':rows,
        'model_calls_recorded':len(list(records.glob('*.json'))),'note':protocol['scope']}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
