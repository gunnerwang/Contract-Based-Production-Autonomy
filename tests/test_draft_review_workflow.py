"""Draft correction and in-process waiting preserve the acceptance boundary."""
import threading
import time
from unittest.mock import MagicMock

import pytest

from cbpa.layer1_outcome.draft_review import DraftDecision, DraftReviewRequest, ReviewInbox
from cbpa.layer1_outcome.reconciliation import reconcile_for_review
from cbpa.models.contract import AmbiguityFlag, HardConstraint, OutcomeContract, make_c1
from cbpa.runner.experiment import CBPAExperiment
from cbpa.runner.lifecycle import DeploymentBlocked


def request(active=None):
    draft = make_c1()
    draft.hard_constraints.append(HardConstraint(name="Energy", limit=3, unit="kWh"))
    draft.ambiguities = [AmbiguityFlag(field="crosscheck:Energy", question="Was this cap requested?", default_resolution="Review required")]
    return DraftReviewRequest(draft, "No hard energy cap", 1, active=active)


def decision(req, contract=None, **overrides):
    fields = dict(request_id=req.request_id, digest=req.digest, reviewer_id="test-script",
                  reason="Scripted test: the request has no energy cap.", confirmed=True,
                  source="scripted", contract=contract or make_c1())
    fields.update(overrides)
    return DraftDecision(**fields)


def test_spurious_draft_limit_can_be_removed_without_mutating_input():
    req = request()
    result = req.resolve(decision(req))
    assert result.get_constraint("Energy") is None
    assert req.draft.get_constraint("Energy").limit == 3
    assert result.ambiguities[0].resolved
    assert result.context["_draft_review"]["source"] == "scripted"


@pytest.mark.parametrize("kind", ["missing", "looser", "strictness", "direction"])
def test_active_obligation_cannot_be_removed_or_weakened(kind):
    active = make_c1()
    active.hard_constraints.append(HardConstraint(name="Energy", operator="<", limit=300, unit="kWh"))
    req = request(active)
    corrected = make_c1()
    if kind != "missing":
        corrected.hard_constraints.append(HardConstraint(name="Energy", operator=">=" if kind=="direction" else "<=", limit=301 if kind=="looser" else 300, unit="kWh"))
    with pytest.raises(ValueError, match="active constraint"):
        req.resolve(decision(req, corrected))
    assert req.draft.get_constraint("Energy").limit == 3


@pytest.mark.parametrize("field,limit", [("Noise",81),("FatigueIndex",.41),("CyberRiskLevel",3)])
def test_draft_correction_cannot_relax_plant_limits(field,limit):
    req=request();candidate=make_c1();candidate.get_constraint(field).limit=limit
    with pytest.raises(ValueError,match="mandatory bound"):
        req.resolve(decision(req,candidate))


def test_review_rejects_stale_unconfirmed_and_reused_decisions():
    req=request()
    for patch in [dict(confirmed=False),dict(digest="stale"),dict(request_id="other")]:
        with pytest.raises(ValueError):req.resolve(decision(req,**patch))
    req.resolve(decision(req))
    with pytest.raises(ValueError,match="already"):
        req.resolve(decision(req))


def test_units_are_compiled_before_active_constraint_comparison():
    active=make_c1();active.hard_constraints.append(HardConstraint(name="Energy",limit=300,unit="kWh"))
    req=request(active);candidate=make_c1();candidate.hard_constraints.append(HardConstraint(name="Energy",limit=300000,unit="Wh"))
    assert req.resolve(decision(req,candidate)).get_constraint("Energy").limit==300


def test_phase_one_waits_then_continues_without_reelicitation():
    inbox=ReviewInbox(3)
    exp=CBPAExperiment(use_llm=False, manager_intent="Energy is a soft objective; no energy cap.", contract_reviewer=inbox)
    exp.elicitation.use_llm=True
    draft=make_c1();draft.hard_constraints.append(HardConstraint(name="Energy",limit=3,unit="kWh"))
    exp.elicitation.elicit_with_refinement=MagicMock(return_value=(draft,[]))
    reached=threading.Event()
    class PlanningReached(Exception):pass
    def planning(*args,**kwargs):
        reached.set();raise PlanningReached()
    exp.planner.generate_initial=MagicMock(side_effect=planning)
    errors=[]
    def worker():
        try:exp._phase1()
        except Exception as exc:errors.append(exc)
    thread=threading.Thread(target=worker);thread.start()
    try:
        deadline=time.monotonic()+2
        while inbox.pending() is None and time.monotonic()<deadline:time.sleep(.005)
        pending=inbox.pending();assert pending is not None
        assert not reached.is_set()
        corrected=OutcomeContract.model_validate(pending['contract'])
        corrected.hard_constraints=[c for c in corrected.hard_constraints if c.name!='Energy']
        d=DraftDecision(request_id=pending['request_id'],digest=pending['digest'],reviewer_id='test-script',reason='No energy cap was requested.',confirmed=True,source='scripted',contract=corrected)
        with pytest.raises(ValueError):inbox.submit(d.model_copy(update={'digest':'stale'}))
        assert inbox.pending() is not None
        changed = d.model_copy(deep=True)
        changed.contract.typed_assumptions[0].expected_value = 999
        with pytest.raises(ValueError, match="typed clarification"):
            inbox.submit(changed)
        assert inbox.pending() is not None and not reached.is_set()
        assert inbox.pending()['contract']['typed_assumptions'][0]['expected_value'] == 52
        inbox.submit(d)
        thread.join(2)
        assert not thread.is_alive() and reached.is_set()
        assert len(errors)==1 and isinstance(errors[0],PlanningReached)
        assert exp.elicitation.elicit_with_refinement.call_count==1
        assert exp.planner.generate_initial.call_count==1
        assert len([e for e in exp.audit.entries if e.action=='contract_review_confirmed'])==1
        with pytest.raises(ValueError):inbox.submit(d)
    finally:
        thread.join(4);exp.shutdown()


def test_review_timeout_does_not_dispatch():
    exp=CBPAExperiment(use_llm=False,manager_intent='Noise below 77 dB.',contract_reviewer=ReviewInbox(.01))
    exp.elicitation.use_llm=True
    exp.elicitation.elicit_with_refinement=MagicMock(return_value=(make_c1(),[]))
    exp.planner.generate_initial=MagicMock()
    with pytest.raises(DeploymentBlocked,match='without confirmation'):exp._phase1()
    exp.planner.generate_initial.assert_not_called()


def test_instance_intents_do_not_cross_contaminate():
    a=CBPAExperiment(manager_intent='A');b=CBPAExperiment(manager_intent='B')
    assert (a.manager_intent,b.manager_intent)==('A','B')


def test_review_cannot_change_execution_horizon_or_assumptions():
    for field in ("context", "assumptions"):
        req = request()
        candidate = make_c1()
        if field == "context":
            candidate.context["shift_hours"] = 16
        else:
            candidate.assumptions = {"unreviewed": "new assumption"}
        with pytest.raises(ValueError, match="typed clarification"):
            req.resolve(decision(req, candidate))


@pytest.mark.parametrize("change", [
    "remove_one", "remove_all", "add", "reorder", "name", "description",
    "expected_value", "tolerance_pct", "unit", "current_value", "violated", "drift_pct",
])
def test_typed_assumptions_remain_bound_to_review_snapshot(change):
    req = request()
    original = req.draft.model_dump(mode="json")
    candidate = make_c1()
    assumptions = candidate.typed_assumptions
    if change == "remove_one":
        assumptions.pop()
    elif change == "remove_all":
        candidate.typed_assumptions = []
    elif change == "add":
        assumptions.append(assumptions[0].model_copy(update={"name": "extra"}))
    elif change == "reorder":
        candidate.typed_assumptions = list(reversed(assumptions))
    else:
        values = {"name": "changed", "description": "changed", "expected_value": 999,
                  "tolerance_pct": 99, "unit": "changed", "current_value": 999,
                  "violated": True, "drift_pct": 99}
        setattr(assumptions[0], change, values[change])
    with pytest.raises(ValueError, match="typed clarification"):
        req.resolve(decision(req, candidate))
    assert req.draft.model_dump(mode="json") == original
    # Invalid submissions do not consume the request or prevent a valid repair.
    accepted = req.resolve(decision(req))
    assert accepted.typed_assumptions == req.draft.typed_assumptions
    assert accepted.get_constraint("Energy") is None
