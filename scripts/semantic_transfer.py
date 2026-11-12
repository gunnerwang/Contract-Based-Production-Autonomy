#!/usr/bin/env python3
"""Prospective new-intent transfer check with unchanged parser prompt and baselines."""
import concurrent.futures,copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from semantic_model_followup import SYSTEM,ELICITATION_TOOL_SCHEMA,infer,evaluate,merge_union
from semantic_followup import tuples,kpi
from cbpa.baselines.elicitation_baselines import TemplateElicitor,RuleBasedElicitor
from cbpa.models.contract import OutcomeContract,HardConstraint,KPITarget,make_c1,PriorityLevel

UNITS={'Noise':{'','dB','db'},'Energy':{'','kWh','kwh'},'FatigueIndex':{'','dimensionless','index'},
       'CyberRiskLevel':{'','dimensionless','level'},'DefectRate':{'','fraction','dimensionless'},
       'Throughput':{'','u/h','units/h','units per hour'},'DeadlineGap':{'','%','percent'}}
def unit_errors(c):
    return [f'{h.name}: unexpected unit {h.unit}' for h in c.hard_constraints if h.name in UNITS and h.unit not in UNITS[h.name]] + [f'{k.name}: unexpected target unit {k.unit}' for k in c.kpi_targets if k.name=='Throughput' and k.threshold is not None and k.unit not in UNITS['Throughput']]

def score(case,c,method,seed,rep,error=None):
    units=unit_errors(c) if c else ['no parsed contract']
    row=evaluate(case,c if not units else None,method,seed,error or '; '.join(units) or None)
    if c and units:
        ref={ (h['name'],h['operator'],h['limit']) for h in case['reference']['constraints']}
        row['constraints_exact']=tuples(c)==ref
        row['kpi_exact']=kpi(c)==(case['reference']['kpi']['direction'],case['reference']['kpi']['threshold'])
        row['predicted_contract']=c.model_dump(mode='json')
    row['replicate']=rep;row['units_correct']=not units
    wanted=case['reference']['explicit_priority']
    row['priority_exact']=None if wanted is None else c is not None and [p.value for p in c.priority_order]==wanted
    row['exact_scored_semantics']=row['constraints_exact'] and row['kpi_exact'] and row['units_correct'] and row['priority_exact'] is not False
    return row

def main():
    out=ROOT/'data/validation/semantic_transfer_2026-09-14';cases=json.loads((out/'cases.json').read_text())
    old=ROOT/'data/validation/semantic_model_followup_2026-09-13/protocol.json'
    frozen=json.loads(old.read_text());assert SYSTEM==frozen['system'] and ELICITATION_TOOL_SCHEMA==frozen['schema']
    protocol={'model':'claude-sonnet-4-6','replicates':2,'new_cases':len(cases),'new_calls':2*len(cases),'mc_seeds':[42,43,44],
      'cases_sha256':hashlib.sha256((out/'cases.json').read_bytes()).hexdigest(),'system':SYSTEM,'schema':ELICITATION_TOOL_SCHEMA,
      'prior_prompt_protocol_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*.py'))},
      'shared_script_sha256':{name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in ['semantic_model_followup.py','semantic_followup.py']},
      'scope':'New authored intents written after parser prompt and implementations were frozen, evaluated without tuning. Distinct compositions, not three numeric clones per template. Independent annotation and external representativeness remain unestablished.',
      'unit_policy':'No conversion or correction by scorer. Explicit units must match canonical units or documented synonyms; absent units inherit named ontology fields. Mismatches refuse planning.',
      'refusal_policy':'No candidate found is compared with the same finite reference-contract search, not interpreted as global infeasibility.',
      'metrics':'Exact hard constraints incl operator/value, explicit canonical units, throughput KPI direction/target, complete order on three explicit-priority cases. Other semantic fields are not fully scored.',
      'sampling':'Two fresh generations per intent; same generation reused for LLM/union and MC seeds. Baselines evaluated once per intent; failed calls retained without replacement.'}
    p=out/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol,'Frozen transfer protocol changed'
    else:p.write_text(json.dumps(protocol,indent=2)+'\n')
    records=out/'responses';records.mkdir(exist_ok=True)
    tasks=[]
    for rep in [0,1]:
        for case in cases:
            c=copy.deepcopy(case);c['id']=f"rep{rep}_{case['id']}";tasks.append(c)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda c:infer(c,protocol['model'],records),tasks))
    rows=[]
    for case in cases:
        text=case['intent']+('\nManager clarification: '+case['answer'] if case['answer'] else '')
        ref=make_c1();ref.hard_constraints=[HardConstraint(**v) for v in case['reference']['constraints']];ref.kpi_targets=[KPITarget(**case['reference']['kpi'])]
        if case['reference']['explicit_priority']:ref.priority_order=[PriorityLevel(v) for v in case['reference']['explicit_priority']]
        for method,c in [('template',TemplateElicitor().elicit(text)),('rule',RuleBasedElicitor().elicit(text)),('structured_reference',ref)]:
            for seed in [42,43,44]:rows.append(score(case,c,method,seed,None))
        for rep in [0,1]:
            record=json.loads((records/f"rep{rep}_{case['id']}.json").read_text())
            raw=OutcomeContract.model_validate(record['contract']) if record['status']=='ok' else None
            for method in ['llm','llm_rule_union']:
                c=raw;error=record.get('error')
                if c and method=='llm_rule_union':
                    try:c=merge_union(c,RuleBasedElicitor().elicit(text))
                    except Exception as exc:c=None;error=str(exc)
                for seed in [42,43,44]:rows.append(score(case,c,method,seed,rep,error))
    ref={(r['case'],r['seed']):r for r in rows if r['method']=='structured_reference'}
    for r in rows:
        expected=ref[r['case'],r['seed']]['certified_candidate_found']
        r['reference_candidate_found']=expected
        r['agrees_with_reference_search']=r['certified_candidate_found']==expected
        r['semantics_and_search_success']=r['exact_scored_semantics'] and r['agrees_with_reference_search'] and (not expected or (r['reference_compliant'] and r['target_met'])) and not r['validation_errors']
    summary={}
    for method in ['template','rule','structured_reference','llm','llm_rule_union']:
        rr=[v for v in rows if v['method']==method];sem=[v for v in rr if v['seed']==42];pr=[v for v in sem if v['priority_exact'] is not None]
        summary[method]={'semantic_outputs':len(sem),'exact_constraints':sum(v['constraints_exact'] for v in sem),'exact_kpi':sum(v['kpi_exact'] for v in sem),
          'correct_units':sum(v['units_correct'] for v in sem),'priority_cases':len(pr),'exact_priority':sum(v['priority_exact'] for v in pr),
          'exact_scored_semantics':sum(v['exact_scored_semantics'] for v in sem),'planning_combinations':len(rr),
          'candidate_found':sum(v['certified_candidate_found'] for v in rr),'reference_compliant':sum(v['reference_compliant'] for v in rr),
          'matched_search_outcome':sum(v['agrees_with_reference_search'] for v in rr),'semantics_and_search_success':sum(v['semantics_and_search_success'] for v in rr)}
    (out/'results.json').write_text(json.dumps({'protocol':protocol,'summary':summary,'rows':rows},indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
