#!/usr/bin/env python3
"""Replay archived L3 decisions through the corrected runner boundary.

Each archived phase is an independent input, including phases that would be
unreachable after an earlier halt. This does not regenerate candidates, verify
physics, or estimate throughput of a corrected lifecycle. No API calls.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cbpa.models.metrics import FeasibilityReport, StochasticFeasibilityReport
from cbpa.layer3_verification.feasibility_cert import FeasibilityCertificate
from cbpa.runner.experiment import CBPAExperiment
from cbpa.runner.factory_experiment import FactoryExperiment
from cbpa.runner.lifecycle import DeploymentBlocked

BATCHES=[
 'single_full_claude-sonnet-4-6_integrated_v5',
 'single_full_nollm_integrated',
 'single_full_claude-sonnet-4-6_llmonly_integrated',
 'factory_claude-sonnet-4-6_analytical_v3',
 'factory_nollm_analytical',
 'factory_claude-sonnet-4-6_llmonly_analytical',
 'factory_gemini-3.8-flash_llmonly_analytical',
]

def main():
 output={'scope':'Independent replay of archived non-replan L3 nominal and sampled decisions; not a candidate, physics, or trajectory rerun.',
         'protocol':'200 samples, threshold 0.95; missing per-event sample count uses the archived runner protocol. Historical seed field is not used to resample.',
         'batches':{},'input_sha256':{},'rows':[]}
 for batch in BATCHES:
  single=batch.startswith('single')
  runner=CBPAExperiment(use_llm=False) if single else FactoryExperiment(use_llm=False)
  counts={'decisions':0,'allowed':0,'blocked_nominal':0,'blocked_sampled_only':0}
  folder=ROOT/'data/repeated'/batch
  paths=sorted(folder.glob('run_*/shift_*/audit_trail.json') if single else folder.glob('run_*/audit_trail.json'))
  if not paths: raise RuntimeError(f'No records for {batch}')
  for path in paths:
   output['input_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
   events=json.loads(path.read_text())
   seen=set()
   for event in events:
    phase=event['phase']; d=event['details']
    if event['layer']!='L3' or phase not in {1,5,6} or 'p_feasible' not in d: continue
    assert phase not in seen,(path,phase)
    seen.add(phase)
    nominal=d['feasible']; p=d['p_feasible']
    name=d.get('schedule',f'phase_{phase}')
    # Pass archived outcomes through the current boundary without re-evaluation.
    stoch=StochasticFeasibilityReport(p_feasible=p,n_samples=d.get('stochastic_n_samples',200),seed=42)
    cert=FeasibilityCertificate(schedule_name=name,is_certified=nominal and p>=.95,
        feasibility_report=FeasibilityReport(is_feasible=nominal,violations=d.get('violations',[])),
        p_feasible=p,stochastic_report=stoch,notes='Archived nominal/sample decision replay')
    try:
     runner._require_deployment_certificate(cert,phase)
     allowed=True
    except DeploymentBlocked:
     allowed=False
    expected=nominal and p>=.95
    assert allowed==expected,(path,phase,nominal,p)
    counts['decisions']+=1
    counts['allowed' if allowed else 'blocked_nominal' if not nominal else 'blocked_sampled_only']+=1
    output['rows'].append({'source':str(path.relative_to(ROOT)),'phase':phase,'nominal':nominal,'p_feasible':p,'allowed':allowed})
  output['batches'][batch]=counts
 output['totals']={key:sum(c[key] for c in output['batches'].values()) for key in counts}
 output['implementation_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
  ROOT/'src/cbpa/runner/lifecycle.py',ROOT/'src/cbpa/runner/experiment.py',ROOT/'src/cbpa/runner/factory_experiment.py',ROOT/'src/cbpa/physics/factory_evaluator.py']}
 dest=ROOT/'data/validation/fail_closed/replay_summary.json'
 dest.parent.mkdir(parents=True,exist_ok=True)
 dest.write_text(json.dumps(output,indent=2)+'\n')
 print(json.dumps({'batches':output['batches'],'totals':output['totals']},indent=2))

if __name__=='__main__': main()
