#!/usr/bin/env python3
"""Two additional fixed-prompt generations per intent; preserves original replicate."""
import concurrent.futures,hashlib,json,sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from semantic_model_followup import infer,evaluate,merge_union,SYSTEM,ELICITATION_TOOL_SCHEMA
from cbpa.models.contract import OutcomeContract
from cbpa.baselines.elicitation_baselines import RuleBasedElicitor

def main():
    old=ROOT/'data/validation/semantic_model_followup_2026-09-13'
    source=ROOT/'data/validation/semantic_followup_2026-09-11/cases.json'
    out=ROOT/'data/validation/semantic_repeats_v2_2026-09-13';out.mkdir(parents=True,exist_ok=True)
    cases=json.loads(source.read_text());model='claude-sonnet-4-6'
    protocol={'model':model,'system':SYSTEM,'schema':ELICITATION_TOOL_SCHEMA,'cases_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'replicates':3,'new_calls':36,'prior_calls':18,'mc_seeds':[42,43,44],
      'scope':'Two new generations with unchanged prompt, plus the retained original generation per authored intent. Repetitions assess within-set model variability, not generalisation.',
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'shared_script_sha256':hashlib.sha256((ROOT/'scripts/semantic_model_followup.py').read_bytes()).hexdigest(),
      'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*.py'))}}
    p=out/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol,'Protocol changed'
    else:p.write_text(json.dumps(protocol,indent=2)+'\n')
    records=out/'responses';records.mkdir(exist_ok=True)
    tasks=[]
    for rep in [1,2]:
        for case in cases:
            c=copy.deepcopy(case);c['id']=f"rep{rep}_{case['id']}";tasks.append(c)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda c:infer(c,model,records),tasks))
    rows=[];calls=[]
    for rep in [0,1,2]:
        for case in cases:
            path=old/'responses'/(case['id']+'.json') if rep==0 else records/(f"rep{rep}_{case['id']}.json")
            record=json.loads(path.read_text());calls.append({'replicate':rep,'case':case['id'],'status':record['status'],'latency_s':record['latency_s'],'response_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            contract=OutcomeContract.model_validate(record['contract']) if record['status']=='ok' else None
            text=case['intent']+('\nManager clarification: '+case['answer'] if case['answer'] else '')
            for method in ['llm','llm_rule_union']:
                c=contract;error=record.get('error')
                if c is not None and method=='llm_rule_union':
                    try:c=merge_union(c,RuleBasedElicitor().elicit(text))
                    except Exception as exc:c=None;error=str(exc)
                for seed in [42,43,44]:
                    row=evaluate(case,c,method,seed,error);row['replicate']=rep;rows.append(row)
    summary={}
    for method in ['llm','llm_rule_union']:
        selected=[v for v in rows if v['method']==method]
        semantics=[v for v in selected if v['seed']==42]
        summary[method]={'generations':len(semantics),'exact_constraints':sum(v['constraints_exact'] for v in semantics),
            'exact_kpi':sum(v['kpi_exact'] for v in semantics),'initial_combinations':len(selected),
            'reference_compliant':sum(v['reference_compliant'] for v in selected),'joint_success':sum(v['task_success'] for v in selected)}
    (out/'results.json').write_text(json.dumps({'protocol':protocol,'summary':summary,'calls':calls,'rows':rows},indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
