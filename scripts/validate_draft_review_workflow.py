#!/usr/bin/env python3
"""Scripted single-cell draft review; no model calls or human observations."""
import argparse
import contextlib
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cbpa.config.scenario import ScenarioConfig
from cbpa.layer1_outcome.draft_review import DraftDecision
from cbpa.models.contract import HardConstraint, make_c1
from cbpa.runner.experiment import CBPAExperiment
from cbpa.runner.lifecycle import DeploymentBlocked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing evidence is never overwritten')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__).resolve()]
    protocol = {
        'scope': __doc__, 'new_model_calls': 0, 'human_participants': 0,
        'draft_source': 'Authored fixture, recreated at each elicitation; not archived or live model output',
        'review_source': 'scripted; removes a spurious draft cap, preserves a requested infeasible cap, or deliberately attempts an invalid change',
        'seeds': [42, 43, 44], 'cases': ['corrected_draft', 'unanswered', 'requested_infeasible', 'active_change_rejected'],
        'configuration': {'optimiser_anchor': True, 'enable_macro_phase': True, 'use_simulation': True},
        'limits': 'Integration diagnostic, not parser accuracy, independent annotation, human workload, authenticated authority, crash recovery or physical safety',
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for p in sources:
        target = out / 'source_snapshot' / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    runs = []
    for case in protocol['cases']:
        for seed in protocol['seeds']:
            folder = out / f'{case}_{seed}'
            folder.mkdir()
            drafts, reviews, phases = [], [], []
            intent = 'Increase throughput. Keep energy as a soft objective, with no energy cap.'
            if case == 'requested_infeasible':
                intent = 'Increase throughput. Energy must be at most 3 kWh per shift.'
            if case == 'active_change_rejected':
                intent = 'Increase throughput. Energy must be at most 1000 kWh per shift.'
            def elicit(intent, contract_name):
                c = make_c1()
                c.name = contract_name
                c.hard_constraints.append(HardConstraint(name='Energy', limit=3, unit='kWh'))
                # This stricter unrequested noise bound ensures review even when
                # the energy cap is requested and must remain infeasible.
                c.get_constraint('Noise').limit = 79
                drafts.append(c.model_dump(mode='json'))
                return c
            def reviewer(request):
                if case == 'unanswered':
                    reviews.append({'phase': request.phase, 'decision': 'no answer'})
                    return None
                c = request.draft
                if case == 'active_change_rejected':
                    c.get_constraint('Energy').limit = 1000
                elif case != 'requested_infeasible':
                    c.hard_constraints = [h for h in c.hard_constraints if h.name != 'Energy']
                c.get_constraint('Noise').limit = 80
                if case == 'active_change_rejected' and request.phase != 1:
                    c.hard_constraints = [h for h in c.hard_constraints if h.name != 'Energy']
                decision = DraftDecision(request_id=request.request_id, digest=request.digest,
                    reviewer_id='offline-fixture', source='scripted', confirmed=True,
                    reason=f'Authored protocol condition: {case}; no human answer.', contract=c)
                reviews.append({'phase': request.phase, 'decision': decision.model_dump(mode='json')})
                return decision
            exp = CBPAExperiment(config=ScenarioConfig(**protocol['configuration']), use_llm=False,
                                 manager_intent=intent, contract_reviewer=reviewer)
            exp.mc_seed_base = seed
            # Exercise the LLM-draft reconciliation branch with a labelled
            # fixture. All actual agents/clients remain deterministic/offline.
            exp.elicitation.use_llm = True
            exp.elicitation.elicit = elicit
            exp.elicitation.elicit_with_refinement = lambda intent, name: (elicit(intent, name), [])
            status, reason = 'completed', None
            try:
                with (folder / 'execution.log').open('w') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    exp.run_all_phases(on_phase_complete=phases.append)
            except DeploymentBlocked as exc:
                status, reason = 'blocked', str(exc)
            finally:
                exp.shutdown()
                exp.audit.export_json(folder / 'audit_trail.json')
            dispatched = [e.phase for e in exp.audit.entries if e.action in {'schedule_deployed', 'next_shift_prepared'}]
            refused = [e.phase for e in exp.audit.entries if e.action in {'deployment_blocked', 'runtime_execution_blocked'}]
            confirmed = [e.phase for e in exp.audit.entries if e.action == 'contract_review_confirmed']
            row = dict(case=case, seed=seed, status=status, reason=reason,
                       completed_phases=[p.phase for p in phases], dispatch_phases=dispatched,
                       refusal_phases=refused, confirmed_review_phases=confirmed,
                       elicited_contracts=[d['name'] for d in drafts])
            (folder / 'drafts_and_decisions.json').write_text(json.dumps({'intent': intent, 'drafts': drafts, 'reviews': reviews}, indent=2) + '\n')
            runs.append(row)
            (folder / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
            # These assertions check boundaries, not an assumed positive result.
            assert len(row['elicited_contracts']) == len(set(row['elicited_contracts']))
            assert not refused or not any(p > min(refused) for p in dispatched)
            if case in {'unanswered', 'requested_infeasible'}:
                assert status == 'blocked' and refused == [1] and not dispatched
            if case == 'active_change_rejected':
                assert status == 'blocked' and refused == [4] and 4 not in confirmed
            print(json.dumps(row), flush=True)
    summary = {'runs': runs, 'completed': sum(r['status'] == 'completed' for r in runs),
               'blocked': sum(r['status'] == 'blocked' for r in runs),
               'new_model_calls': 0, 'human_participants': 0}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

if __name__ == '__main__':
    main()
