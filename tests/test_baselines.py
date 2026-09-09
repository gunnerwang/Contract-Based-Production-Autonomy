"""Tests for the non-LLM baselines (template / rule-based elicitation, optimiser planner)."""

from __future__ import annotations

from cbpa.baselines.elicitation_baselines import RuleBasedElicitor, TemplateElicitor
from cbpa.baselines.optimizer_planner import OptimiserPlanner
from cbpa.config.scenario import CellConfig
from cbpa.layer2_planning.vr_scorer import VRScorer
from cbpa.layer3_verification.constraint_checker import ConstraintChecker
from cbpa.models.contract import PriorityLevel, make_c1
from cbpa.physics.cell_evaluator import CellEvaluator


def _limits(c):
    return {hc.name: hc.limit for hc in c.hard_constraints}


class TestTemplate:
    def test_defaults_and_numeric_override(self):
        t = TemplateElicitor()
        c = t.elicit("Increase output this shift; keep noise under 78 dB and fatigue below 0.35.")
        assert _limits(c)["Noise"] == 78.0 and _limits(c)["FatigueIndex"] == 0.35
        c2 = t.elicit("Maximise throughput, cap energy at 380 kWh, no overtime.")
        assert _limits(c2)["Noise"] == 80.0 and "Energy" not in _limits(c2)   # unanticipated field dropped


class TestRuleBased:
    def test_synonyms_numbers_and_priorities(self):
        r = RuleBasedElicitor()
        c = r.elicit("Raise productivity but keep the operators fresh; acoustic level not above 76 decibels; "
                     "energy capped at 380 kWh; put safety first; robot arm speed 1.2 m/s; no overtime on the 8-hour shift.")
        lim = _limits(c)
        assert lim["Noise"] == 76.0 and lim["FatigueIndex"] == 0.4 and lim["Energy"] == 380.0
        assert c.priority_order[0] == PriorityLevel.SAFETY
        assert any(a.name == "R1.speed" and a.expected_value == 1.2 for a in c.typed_assumptions)
        assert c.context.get("overtime_allowed") is False and c.context["shift_hours"] == 8

    def test_floor_clamping(self):
        c = RuleBasedElicitor().elicit("Noise under 90 dB is fine, fatigue under 0.6.")
        assert _limits(c)["Noise"] == 80.0 and _limits(c)["FatigueIndex"] == 0.4


class TestOptimiser:
    def test_finds_feasible_better_or_equal_vr(self):
        ev = CellEvaluator(cell=CellConfig(), mode="analytical")
        opt = OptimiserPlanner(ev, VRScorer(mode="analytical"), ConstraintChecker())
        r = opt.plan(make_c1(), 52.0, grid=7, refine=150)
        assert r.feasible_found and r.schedule.source == "optimiser"
        assert ConstraintChecker().check(make_c1(), ev.evaluate(r.schedule)).is_feasible
        assert r.objective >= 0.481 - 1e-9   # at least the anchor S1e's V/R
