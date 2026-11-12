"""Mandatory single-cell floors under the default review policy."""
from cbpa.models.contract import HardConstraint, make_c1
from cbpa.runner.experiment import MANAGER_INTENT, _canonicalise_contract


def _draft(constraints):
    return make_c1().model_copy(update={"hard_constraints": constraints})


def test_review_policy_reinserts_omitted_cyber_risk_floor():
    # Archived LLM drafts name fatigue and noise in their own vocabulary and omit
    # CyberRiskLevel; the review gate requires all three plant floors.
    draft = _draft([
        HardConstraint(name="operator_fatigue", operator="<=", limit=0.4),
        HardConstraint(name="noise_level", operator="<=", limit=80.0),
    ])
    bound = {c.name: c for c in _canonicalise_contract(draft, MANAGER_INTENT).hard_constraints}
    assert bound["CyberRiskLevel"].operator == "<=" and bound["CyberRiskLevel"].limit == 2.0
    assert bound["FatigueIndex"].limit == 0.4 and bound["Noise"].limit == 80.0


def test_relaxed_cyber_risk_bound_is_clamped():
    draft = _draft([
        HardConstraint(name="FatigueIndex", operator="<=", limit=0.4),
        HardConstraint(name="Noise", operator="<=", limit=80.0),
        HardConstraint(name="cyber_risk_level", operator="<=", limit=3.0),
    ])
    bound = {c.name: c for c in _canonicalise_contract(draft, MANAGER_INTENT).hard_constraints}
    assert bound["CyberRiskLevel"].limit == 2.0
