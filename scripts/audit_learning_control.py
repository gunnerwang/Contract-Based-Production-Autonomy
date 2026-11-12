#!/usr/bin/env python3
"""Post-hoc integrity and verification-work audit of the frozen feedback study."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'data/validation/learning_control_2026-09-14';d=json.loads((out/'results.json').read_text())
    for name,h in d['protocol']['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    assert hashlib.sha256((ROOT/'scripts/learning_control_followup.py').read_bytes()).hexdigest()==d['protocol']['script_sha256']
    summaries={};records=[]
    for arm in ['feedback_on','feedback_off']:
        attempts=0;selected={};blocked=[]
        for row in [r for r in d['rows'] if r['arm']==arm]:
            folder=out/f"r{row['replicate']:02}_{arm}_s{row['shift']:02}"
            audit=json.loads((folder/'audit.json').read_text())
            if row['status']=='completed':
                item=next(e for e in audit if e['phase']==5 and e['action']=='pareto_frontier_computed')
                tried=item['details']['gate_tried'];name=item['details']['selected'];selected[name]=selected.get(name,0)+1
                assert item['details']['gate_certified']
            else:
                refusal=next(e for e in audit if e['phase']==5 and e['action']=='deployment_blocked')
                reason=refusal['details']['reason'];tried=ast.literal_eval(reason.split('; tried=',1)[1]);blocked.append({'replicate':row['replicate'],'shift':row['shift'],'tried':tried})
                assert not any(e['phase']>=5 and e['action']=='schedule_deployed' for e in audit)
            attempts+=len(tried);records.append({'arm':arm,'replicate':row['replicate'],'shift':row['shift'],'phase5_candidate_certifications':len(tried)})
        summaries[arm]={'phase5_candidate_certifications':attempts,'selected_certified_candidates':selected,'blocked':blocked}
    p=d['pairs'];assert len(p)==100 and all(v['candidate_count_on']==v['candidate_count_off']==6 for v in p)
    for x in p:
        assert x['on_status']==x['off_status']
        if x['phase5_pair_observed']:assert all(v==0 for v in x['deltas'].values())
    result={'scope':__doc__,'frozen_source_and_script_match':True,'paired_candidates_each':6,'no_dispatch_after_phase5_refusal':True,
      'summary':summaries,'rows':records,'interpretation':'Equal allocated candidate/search budgets do not imply equal realised verification work. Counts are candidate gate checks, not timing, full optimiser evaluation counts, physical cost or production benefit.',
      'result_sha256':hashlib.sha256((out/'results.json').read_bytes()).hexdigest(),'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'verification_work_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
