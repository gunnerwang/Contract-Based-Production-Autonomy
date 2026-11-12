#!/usr/bin/env python3
"""Offline review-gate replay; no manager answers and no new model calls."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from cbpa.models.contract import OutcomeContract
from cbpa.layer1_outcome.reconciliation import reconcile_for_review
from cbpa.layer1_outcome.contract_validator import ContractValidator
from cbpa.baselines.elicitation_baselines import RuleBasedElicitor
from semantic_transfer import score


def main():
    old=ROOT/'data/validation/semantic_transfer_2026-09-14'
    units=ROOT/'data/validation/unit_compilation_v2_2026-09-14/results.json'
    out=ROOT/'data/validation/parser_review_v3_2026-09-14'
    cases=json.loads((old/'cases.json').read_text());prior=json.loads(units.read_text())
    protocol={'scope':__doc__,'intervention':'Normalize units and compare hard requirements; disagreements pause without scheduling or invented manager answers. LLM input otherwise retained.',
      'scope_limits':'Hard-constraint review only; KPI, priority, context and assumptions are not compared by this gate. Authored known-case replay; not new language evidence or user-study workload.',
      'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*.py'))},
      'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [old/'cases.json',units,*sorted((old/'responses').glob('*.json'))]},
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'mc_seeds':[42,43,44],'new_model_calls':0,'manager_answers':0}
    path=out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    decisions=[];rows=[]
    for case in cases:
        intent=case['intent']+('\nManager clarification: '+case['answer'] if case['answer'] else '')
        for rep in [0,1]:
            record=json.loads((old/'responses'/f"rep{rep}_{case['id']}.json").read_text())
            llm=OutcomeContract.model_validate(record['contract'])
            result=reconcile_for_review(llm,RuleBasedElicitor().elicit(intent))
            pending=[f.field for f in result.ambiguities if not f.resolved]
            union=next(v for v in prior['rows'] if v['method']=='llm_rule_union' and v['case']==case['id'] and v['replicate']==rep and v['seed']==42)
            decisions.append({'case':case['id'],'replicate':rep,'pending_fields':pending,'requires_review':bool(pending),
                              'prior_union_semantics_correct':union['exact_scored_semantics'],
                              'review_contract':result.model_dump(mode='json')})
            if pending:
                assert not ContractValidator().validate(result).is_valid
                continue
            for seed in protocol['mc_seeds']:
                row=score(case,result,'llm_review_gate',seed,rep)
                reference=next(v for v in prior['rows'] if v['method']=='structured_reference' and v['case']==case['id'] and v['seed']==seed)
                row['automatic_joint_success']=row['exact_scored_semantics'] and not row['validation_errors'] and (row['certified_candidate_found']==reference['certified_candidate_found']) and (not reference['certified_candidate_found'] or (row['reference_compliant'] and row['target_met']))
                rows.append(row)
    bad=[d for d in decisions if not d['prior_union_semantics_correct']]
    summary={'model_outputs':len(decisions),'review_required_outputs':sum(d['requires_review'] for d in decisions),
      'automatically_cleared_outputs':sum(not d['requires_review'] for d in decisions),
      'unique_intents_requiring_review':len({d['case'] for d in decisions if d['requires_review']}),
      'prior_union_semantic_errors':len(bad),'prior_union_errors_routed_to_review':sum(d['requires_review'] for d in bad),
      'total_output_seed_inputs':len(decisions)*3,'paused_output_seed_inputs':sum(d['requires_review'] for d in decisions)*3,
      'attempted_planning_combinations':len(rows),'accepted_candidates':sum(v['certified_candidate_found'] for v in rows),
      'reference_compliant_candidates':sum(v['reference_compliant'] for v in rows),
      'automatic_joint_success':sum(v['automatic_joint_success'] for v in rows),'manager_answers':0}
    (out/'results.json').write_text(json.dumps({'protocol':protocol,'summary':summary,'decisions':decisions,'planning_rows':rows},indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
