#!/usr/bin/env python3
"""Versioned post-hoc unit repair replay; no model calls and no changed labels."""
import copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from cbpa.models.contract import OutcomeContract
from cbpa.models.units import normalize_contract_units
from cbpa.baselines.elicitation_baselines import RuleBasedElicitor
from semantic_transfer import score
from semantic_model_followup import merge_union


def main():
    old=ROOT/'data/validation/semantic_transfer_2026-09-14'
    out=ROOT/'data/validation/unit_compilation_v2_2026-09-14'
    original=json.loads((old/'results.json').read_text())
    cases=json.loads((old/'cases.json').read_text())
    protocol={'revision':'Explicitly clear rejected union outputs; preceding v1 retained, no failed calls replaced', 'scope':__doc__,'source_experiment':'semantic_transfer_2026-09-14',
        'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [old/'cases.json',old/'protocol.json',old/'results.json',*sorted((old/'responses').glob('*.json'))]},
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*.py'))},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'shared_script_sha256':{name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in ['semantic_transfer.py','semantic_model_followup.py','semantic_followup.py']},
        'new_model_calls':0,'mc_seeds':[42,43,44],
        'intervention':'Explicit quantity normalization before identical scoring/planning; union normalizes both parser inputs before comparing numeric limits. No semantic requirement, label or prompt is repaired.',
        'interpretation':'Development replay on observed errors, not independent confirmation, a new LLM generation experiment, or full lifecycle validation.'}
    path=out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Create a new version after source changes'
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[];audits=[]
    for case in cases:
        text=case['intent']+('\nManager clarification: '+case['answer'] if case['answer'] else '')
        for method in ['template','rule','structured_reference','llm','llm_rule_union']:
            reps=[0,1] if method.startswith('llm') else [None]
            for rep in reps:
                if rep is None:
                    prior=next(v for v in original['rows'] if v['case']==case['id'] and v['method']==method and v['seed']==42)
                    raw=OutcomeContract.model_validate(prior['predicted_contract'])
                else:
                    record=json.loads((old/'responses'/f"rep{rep}_{case['id']}.json").read_text())
                    raw=OutcomeContract.model_validate(record['contract'])
                error=None;converted=None;changes=[];rule_changes=[]
                try:
                    converted,changes=normalize_contract_units(raw)
                    if method=='llm_rule_union':
                        rule,rule_changes=normalize_contract_units(RuleBasedElicitor().elicit(text))
                        converted=merge_union(converted,rule)
                except Exception as exc:
                    converted=None
                    error=f'{type(exc).__name__}: {exc}'
                audits.append({'case':case['id'],'method':method,'replicate':rep,
                    'input_contract':raw.model_dump(mode='json'),'conversions':changes,'rule_conversions':rule_changes,
                    'compiled_contract':converted.model_dump(mode='json') if converted else None,'error':error})
                for seed in protocol['mc_seeds']:rows.append(score(case,converted,method,seed,rep,error))
    ref={(v['case'],v['seed']):v for v in rows if v['method']=='structured_reference'}
    for row in rows:
        found=ref[row['case'],row['seed']]['certified_candidate_found']
        row['reference_candidate_found']=found
        row['agrees_with_reference_search']=row['certified_candidate_found']==found
        row['semantics_and_search_success']=row['exact_scored_semantics'] and row['agrees_with_reference_search'] and (not found or (row['reference_compliant'] and row['target_met'])) and not row['validation_errors']
    summary={}
    for method in ['template','rule','structured_reference','llm','llm_rule_union']:
        rr=[v for v in rows if v['method']==method];sem=[v for v in rr if v['seed']==42]
        summary[method]={'semantic_outputs':len(sem),'exact_constraints':sum(v['constraints_exact'] for v in sem),
            'exact_scored_semantics':sum(v['exact_scored_semantics'] for v in sem),'planning_combinations':len(rr),
            'candidate_found':sum(v['certified_candidate_found'] for v in rr),
            'reference_compliant':sum(v['reference_compliant'] for v in rr),
            'semantics_and_search_success':sum(v['semantics_and_search_success'] for v in rr)}
    (out/'results.json').write_text(json.dumps({'protocol':protocol,'summary':summary,'rows':rows,'conversion_audits':audits},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
