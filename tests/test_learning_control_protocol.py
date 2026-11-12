"""Verify feedback ablation retains histories and matches disturbance schedules."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from learning_control_followup import FeedbackStore,sequences


def test_disabled_feedback_preserves_writes_but_exposes_no_prior_information():
    on=FeedbackStore(True);off=FeedbackStore(False)
    record=dict(disturbance_type='demand_surge',disturbance_magnitude=20,
                rejected_policy='S2',rejection_reason='constraint rejection',
                accepted_policy='S3',adaptation_level='meso',outcome_metrics={'throughput_uph':35})
    for store in [on,off]:
        assert store.get_bias_for_planning('demand_surge')=={}
        store.record(**record)
        assert store.total_records==1 and store.records[0].outcome_metrics=={'throughput_uph':35}
    assert on.get_similar('demand_surge') and on.get_bias_for_planning('demand_surge')['parameter_hints']
    assert off.get_similar('demand_surge')==[] and off.get_bias_for_planning('demand_surge')=={}


def test_feedback_histories_are_independent_across_conditions():
    a=FeedbackStore(True);b=FeedbackStore(False)
    a.record('demand_surge',20,'S2','rejected','S3','meso',{'throughput_uph':35})
    assert a.total_records==1 and b.total_records==0


def test_frozen_sequences_balance_disturbances_and_do_not_share_seed_bases():
    a=sequences();assert a==sequences() and len(a)==10
    seeds=[]
    for seq in a:
        shifts=seq['shifts'];assert len(shifts)==10
        assert sorted(s['demand_spike_pct'] for s in shifts)==sorted([10,15,20,25,30]*2)
        seeds.extend(s['mc_seed'] for s in shifts)
    assert len(seeds)==len(set(seeds))==100
