"""Tests for the repeated-run harness: instrumented client, seed injection, aggregation."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from cbpa.llm.instrumented import FallbackLogCapture, InstrumentedLLMClient  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "r1_speed_fraction": {"type": "number"}},
                "required": ["name", "r1_speed_fraction"],
            },
        }
    },
    "required": ["candidates"],
}


class FakeClient:
    """Scripted inner client: each call pops the next behaviour."""

    model = "fake-model"

    def __init__(self, script):
        self.script = list(script)

    @property
    def is_available(self):
        return True

    def check_connection(self):
        return True, "fake connected"

    def query_text(self, system, user_message):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def query_structured(self, system, user_message, tool_name, tool_schema, tool_description=""):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class TestInstrumentedClient:
    def test_records_valid_empty_invalid_and_exception(self):
        inner = FakeClient([
            {"candidates": [{"name": "a", "r1_speed_fraction": 0.5}, {"name": "b", "r1_speed_fraction": 0.6}]},
            {},                                    # JSON extraction failed upstream
            {"candidates": [{"name": "c"}]},       # schema violation (missing field)
            RuntimeError("boom"),                  # transport failure
            "plain text answer",
        ])
        c = InstrumentedLLMClient(inner)
        out = c.query_structured("s", "u", "propose", SCHEMA)
        assert len(out["candidates"]) == 2
        assert c.query_structured("s", "u", "propose", SCHEMA) == {}
        c.query_structured("s", "u", "propose", SCHEMA)
        with pytest.raises(RuntimeError):
            c.query_structured("s", "u", "repair", SCHEMA)
        assert c.query_text("s", "u") == "plain text answer"

        recs = c.records
        assert [r.ok for r in recs] == [True, True, True, False, True]
        assert recs[0].schema_valid is True and recs[0].n_items == 2
        assert recs[1].empty_output is True and recs[1].schema_valid is False
        assert recs[2].empty_output is False and recs[2].schema_valid is False
        assert "r1_speed_fraction" in (recs[2].schema_error or "")
        assert recs[3].error.startswith("RuntimeError")
        assert recs[4].kind == "text"

        st = c.stats()
        assert st["total"]["calls"] == 5
        assert st["total"]["exceptions"] == 1
        assert st["total"]["empty_output"] == 1
        assert st["total"]["schema_invalid"] == 1
        assert st["by_tool"]["propose"]["calls"] == 3
        assert st["by_tool"]["repair"]["exceptions"] == 1
        assert c.model == "fake-model"
        assert "provider" in c.sampling_settings()

    def test_fallback_capture(self):
        cap = FallbackLogCapture().attach("cbpa")
        try:
            logging.getLogger("cbpa.layer2_planning.plan_generator").warning(
                "LLM planning failed: x, falling back to deterministic")
            logging.getLogger("cbpa.layer2_planning.plan_generator").info("not a fallback")
            logging.getLogger("cbpa.meta_layer.escalation_agent").warning("something else entirely")
        finally:
            cap.detach("cbpa")
        assert cap.by_component() == {"plan_generator": 1}


class TestSeedInjection:
    def test_factory_seed_base_propagates(self):
        from cbpa.runner.factory_experiment import FactoryExperiment

        exp = FactoryExperiment(use_llm=False, mc_seed_base=7)
        result = exp.run_all_phases()
        reports = [p.stochastic_report for p in result.phases if p.stochastic_report is not None]
        assert reports, "expected at least one stochastic verification"
        assert reports[0].seed == 7
        assert all(r.seed >= 7 for r in reports)

    def test_default_seed_unchanged(self):
        from cbpa.runner.factory_experiment import FactoryExperiment

        exp = FactoryExperiment(use_llm=False)
        assert exp.mc_seed_base == 42
        result = exp.run_all_phases()
        first = next(p.stochastic_report for p in result.phases if p.stochastic_report is not None)
        assert first.seed == 42


class TestAggregation:
    def test_describe_and_ci(self):
        from aggregate_repeated import describe, t_crit

        st = describe([1.0, 2.0, 3.0, 4.0])
        assert st["n"] == 4 and st["mean"] == 2.5
        assert abs(st["sd"] - 1.2909944) < 1e-6
        assert abs(st["ci95_half"] - t_crit(3) * st["sd"] / 2) < 1e-9
        assert describe([]) == {"n": 0}
        assert describe([5.0])["sd"] == 0.0

    def test_stability_and_firewall(self):
        from aggregate_repeated import firewall_stats, stability

        def rec(name, src, r1, thr, feasible=True, pool=None):
            return {
                "schedule": {"name": name, "source": src,
                             "params": {"r1_speed_fraction": r1, "r2_speed_fraction": 0.5,
                                        "human_cycle_rate_multiplier": 1.0, "buffer_time_s": 3.0}},
                "kpis": {"throughput_uph": thr}, "feasible": feasible, "p_feasible": 1.0,
                "pareto": pool,
            }
        pool = {
            "candidate_sources": {"A_llm": "llm", "B_det": "deterministic"},
            "scores": {"A_llm": {"feasible": 0.0}, "B_det": {"feasible": 1.0}},
            "non_dominated": ["B_det"], "selected": "B_det", "selected_source": "deterministic",
        }
        pool_llm = dict(pool, selected="A_llm", selected_source="llm")
        recs = [rec("B_det", "deterministic", 0.60, 40.0, pool=pool),
                rec("B_det", "deterministic", 0.61, 40.5, pool=pool),
                rec("Zed", "llm", 0.80, 47.0, pool=pool_llm)]
        stb = stability(recs)
        assert stb["modal_name"] == "B_det" and abs(stb["same_name_share"] - 2 / 3) < 1e-9
        assert abs(stb["same_params_share"] - 2 / 3) < 1e-9
        # selection provenance comes from the pareto record, not the deployed object
        assert abs(stb["llm_selected_share"] - 1 / 3) < 1e-9
        recs_nopool = [rec("Zed", "llm", 0.80, 47.0, pool=None)]
        assert stability(recs_nopool)["llm_selected_share"] == 1.0
        fw = firewall_stats(recs)
        assert fw["by_source"]["llm"]["generated"] == 3
        assert fw["by_source"]["llm"]["pass_deterministic_check"] == 0
        assert fw["by_source"]["deterministic"]["on_pareto_front"] == 3
        assert fw["selected_feasible_share"] == 1.0


class TestHarnessEndToEnd:
    def test_factory_no_llm_two_runs(self, tmp_path):
        from run_repeated import main

        root = tmp_path / "batch"
        rc = main(["--case", "factory", "--runs", "2", "--no-llm", "--quiet",
                   "--output-root", str(root)])
        assert rc == 0
        metas = [json.loads(p.read_text()) for p in sorted(root.glob("run_*/run_meta.json"))]
        assert [m["status"] for m in metas] == ["ok", "ok"]
        assert (root / "run_00" / "factory_table.json").exists()
        # deterministic mode must be identical across runs
        k0 = [p["kpis"] for p in metas[0]["shifts"][0]["phases"] if p["kpis"]]
        k1 = [p["kpis"] for p in metas[1]["shifts"][0]["phases"] if p["kpis"]]
        assert k0 == k1 and len(k0) >= 3
        summary = json.loads((root / "summary.json").read_text())
        assert summary["n_runs"] == 2 and summary["case"] == "factory"
        thr = summary["phases"][0]["kpis"]["throughput_uph"]
        assert thr["n"] == 2 and thr["sd"] == 0.0
        assert summary["phases"][0]["stability"]["same_params_share"] == 1.0
        assert (root / "summary_table.tex").exists()
        assert (root / "figures" / "variability_boxplots.pdf").exists()
        assert summary["llm_calls"] is None
        # resume skips completed runs
        rc = main(["--case", "factory", "--runs", "2", "--no-llm", "--quiet", "--resume",
                   "--no-aggregate", "--output-root", str(root)])
        assert rc == 0
        assert json.loads((root / "batch_meta.json").read_text())["runs_skipped"] == 2

    def test_single_cell_no_llm_one_shift(self, tmp_path):
        from run_repeated import main

        root = tmp_path / "single"
        rc = main(["--case", "single", "--runs", "1", "--shifts", "1", "--no-llm", "--quiet",
                   "--simulate", "--output-root", str(root)])
        assert rc == 1
        meta = json.loads((root / "run_00" / "run_meta.json").read_text())
        assert meta["status"] == "error" and "DeploymentBlocked" in meta["error"]
        events = json.loads((root / "run_00" / "audit_trail.json").read_text())
        assert events[-1]["action"] == "deployment_blocked"
        assert not any(e["phase"] == 5 and e["action"] == "schedule_deployed" for e in events)
        assert not (root / "run_00" / "table3.json").exists()


class TestContractCanonicalisation:
    def test_rename_clamp_and_reinsert(self):
        from cbpa.models.contract import HardConstraint, make_c1
        from cbpa.runner.experiment import _canonicalise_contract

        c = make_c1()
        c = c.model_copy(update={"hard_constraints": [
            HardConstraint(name="ambient_noise_level", operator="<=", limit=85.0),
            HardConstraint(name="CyberRiskLevel", operator="<=", limit=2.0),
        ]})
        out = _canonicalise_contract(c)
        names = {hc.name: hc.limit for hc in out.hard_constraints}
        assert names["Noise"] == 80.0          # renamed and clamped
        assert names["FatigueIndex"] == 0.4    # re-inserted
        assert names["CyberRiskLevel"] == 2.0  # untouched
        assert "ambient_noise_level" not in names

    def test_typed_assumptions_reinserted_for_llm_contract(self):
        """Defect (vi): LLM-elicited contracts carried no typed assumptions, so the
        Layer-4 tracker had nothing to check.  Canonicalisation must re-insert them."""
        from cbpa.models.contract import OutcomeContract, HardConstraint, make_c1, make_c1_factory, reinsert_typed_assumptions
        from cbpa.runner.experiment import _canonicalise_contract

        llm_like = OutcomeContract(
            name="C1",
            hard_constraints=[HardConstraint(name="operator_fatigue", operator="<=", limit=0.4)],
            assumptions={"demand": "about 52 units per hour"},
        )
        assert llm_like.typed_assumptions == []
        out = _canonicalise_contract(llm_like)
        names = {a.name for a in out.typed_assumptions}
        assert names == {a.name for a in make_c1().typed_assumptions}
        assert out.assumptions == {"demand": "about 52 units per hour"}  # free text kept
        # existing typed entries are preserved, not duplicated or overwritten
        kept = out.model_copy(update={"typed_assumptions": [out.typed_assumptions[0].model_copy(update={"expected_value": 60.0})]})
        again = reinsert_typed_assumptions(kept, make_c1())
        assert sum(a.name == kept.typed_assumptions[0].name for a in again.typed_assumptions) == 1
        assert next(a for a in again.typed_assumptions if a.name == kept.typed_assumptions[0].name).expected_value == 60.0
        fac = reinsert_typed_assumptions(OutcomeContract(name="C1_factory"), make_c1_factory())
        assert {a.name for a in fac.typed_assumptions} >= {"factory_demand_uph", "h2_availability", "supply_vc_available"}

    def test_tracker_fires_on_canonicalised_llm_contract(self):
        from cbpa.models.contract import OutcomeContract
        from cbpa.runner.experiment import _canonicalise_contract
        from cbpa.layer4_execution.assumption_tracker import AssumptionTracker

        c = _canonicalise_contract(OutcomeContract(name="C1"))
        tracker = AssumptionTracker(c.typed_assumptions)
        drift = tracker.update("demand_base_uph", 62.4)
        assert drift is not None and "demand_base_uph" in drift.attribution
        assert any("demand_base_uph" in a for a in tracker.attribute_constraint_breach("FatigueIndex", "hard"))
        # a contract with no typed assumptions gives the monitor nothing to check
        assert AssumptionTracker([]).update("demand_base_uph", 62.4) is None

    def test_tightest_duplicate_kept_and_canonical_preserved(self):
        from cbpa.models.contract import HardConstraint, make_c1
        from cbpa.runner.experiment import _canonicalise_contract

        c = make_c1().model_copy(update={"hard_constraints": [
            HardConstraint(name="operator_fatigue_index", operator="<=", limit=0.35),
            HardConstraint(name="FatigueIndex", operator="<=", limit=0.6),
            HardConstraint(name="noise_level", operator="<=", limit=78.0),
        ]})
        out = _canonicalise_contract(c)
        names = {hc.name: hc.limit for hc in out.hard_constraints}
        assert names["FatigueIndex"] == 0.35   # tightest duplicate wins; 0.6 clamped then deduped
        assert names["Noise"] == 78.0          # tighter than the floor stays

    def test_unmapped_constraint_rejects(self):
        from cbpa.models.contract import HardConstraint, make_c1
        from cbpa.models.metrics import SimulationMetrics
        from cbpa.layer3_verification.constraint_checker import ConstraintChecker
        c = make_c1().model_copy(update={"hard_constraints": [
            HardConstraint(name="totally_unknown_thing", operator="<=", limit=1.0),
        ]})
        m = SimulationMetrics(throughput_uph=40, defect_rate=0.008, noise_db=77,
                              fatigue_index=0.3, energy_kwh=400, deadline_gap_pct=10)
        report = ConstraintChecker().check(c, m)
        assert not report.is_feasible
        assert report.evaluation_errors == ["totally_unknown_thing: unsupported hard constraint"]

    def test_rule_based_cross_check_tightens_stated_limit(self):
        from cbpa.models.contract import HardConstraint, make_c1
        from cbpa.runner.experiment import _canonicalise_contract

        c = make_c1().model_copy(update={"hard_constraints": [
            HardConstraint(name="operator_fatigue", operator="<=", limit=0.4),
            HardConstraint(name="noise_level", operator="<=", limit=80.0),
        ]})
        out = _canonicalise_contract(c, intent="Increase output; keep fatigue below 0.35 and noise under 78 dB.")
        names = {hc.name: hc.limit for hc in out.hard_constraints}
        assert names["FatigueIndex"] == 0.35 and names["Noise"] == 78.0


class TestCertificationGate:
    def test_gate_certifies_or_blocks(self):
        import pytest
        from cbpa.runner.experiment import CBPAExperiment
        from cbpa.runner.lifecycle import DeploymentBlocked

        exp = CBPAExperiment(use_llm=False)
        with pytest.raises(DeploymentBlocked, match="Phase 5"):
            exp.run_all_phases()
        assert any(e.details.get("gate_certified") for e in exp.audit.entries if e.phase == 1)
        assert exp.audit.entries[-1].action == "deployment_blocked"
        assert not any(e.action == "schedule_deployed" and e.phase == 5 for e in exp.audit.entries)
