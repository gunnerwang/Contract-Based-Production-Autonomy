#!/usr/bin/env python3
"""H3 control (Section 5.4): is the next-shift gain due to the contract revision or to the anchors?

Evaluates the medoid run's deployed FS3 and FS4 schedules, and the deterministic
next-shift anchor pool, under the UNREVISED contract C2_fac (= C1_fac with the
Phase-4 acknowledgement) and under the revised C3_fac, with the same evaluator,
checker, and Monte Carlo settings as the runs.  If FS4 (and the best certified
next-shift anchor) are feasible and certifiable under C2_fac, the 75.8 -> 81.7 u/h
gain is attributable to the anchor pool calibrated for H2 absence rather than to
any change in the contract's constraints.

Usage: python scripts/h3_control.py --factory-root data/repeated/factory_claude-sonnet-4-6_analytical_v3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))

from cbpa.config.scenario import FactoryConfig, FactoryScenarioConfig  # noqa: E402
from cbpa.layer3_verification.constraint_checker import ConstraintChecker  # noqa: E402
from cbpa.models.contract import make_c1_factory, make_c3_factory  # noqa: E402
from cbpa.models.schedule import CellSchedule, FactorySchedule  # noqa: E402
from cbpa.physics.factory_evaluator import FactoryEvaluator  # noqa: E402
from cbpa.runner import factory_experiment as fe  # noqa: E402
from paper_values import medoid, phases_by_canonical  # noqa: E402

SEED = 42


def schedule_from_params(name: str, p: dict) -> FactorySchedule:
    cells = {}
    for cid, c in p["cells"].items():
        cells[cid] = CellSchedule(
            name=f"{name}_{cid}", cell_id=cid, r1_speed_fraction=c["r1_speed_fraction"],
            r2_speed_fraction=c["r2_speed_fraction"], human_cycle_rate_multiplier=c["human_cycle_rate_multiplier"],
            buffer_time_s=c["buffer_time_s"], demand_target_uph=c.get("demand_target_uph", 52.0),
            assigned_operator=c.get("assigned_operator", "H1"), assigned_variants=c.get("assigned_variants", ["V_A"]),
        )
    return FactorySchedule(name=name, source="deterministic", cell_schedules=cells,
                           variant_routing=p["variant_routing"], operator_assignments=p["operator_assignments"])


def verdict(ev, chk, contract, fs) -> dict:
    m = ev.evaluate(fs)
    det = chk.check_factory(contract, m)
    mc = ev.evaluate_monte_carlo(fs, n_samples=200, seed=SEED)
    st = chk.check_factory_stochastic(contract, mc, seed=SEED)
    return {"throughput": m.total_throughput_uph, "max_fatigue": round(max(m.operator_fatigue.values()), 4),
            "noise": m.factory_noise_db, "feasible": det.is_feasible, "p_feasible": round(st.p_feasible, 3),
            "certified": st.p_feasible >= 0.95, "violations": det.violations}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--factory-root", required=True)
    ap.add_argument("--out", default="data/baselines/h3_control.json")
    a = ap.parse_args()
    idx, m = medoid(Path(a.factory_root))
    ph = phases_by_canonical(m)
    c2 = make_c1_factory().model_copy(update={"name": "C2_factory"})
    c3 = make_c3_factory()
    same_constraints = [(h.name, h.operator, h.limit) for h in c2.hard_constraints] == \
                       [(h.name, h.operator, h.limit) for h in c3.hard_constraints]
    ev = FactoryEvaluator(factory=FactoryConfig(), mode="analytical"); chk = ConstraintChecker()
    out = {"factory_root": a.factory_root, "medoid_run": idx, "c2_c3_same_hard_constraints": same_constraints,
           "c3_changes": {"assumptions": {x.name: (x.expected_value, x.tolerance_pct) for x in c3.typed_assumptions
                                          if any(y.name == x.name and (y.expected_value, y.tolerance_pct) != (x.expected_value, x.tolerance_pct)
                                                 for y in c2.typed_assumptions)},
                          "context": {k: v for k, v in c3.context.items() if c2.context.get(k) != v}},
           "deployed": {}, "anchor_pool_under_C2": {}}
    for name in ("FS3", "FS4"):
        fs = schedule_from_params(name, ph[name]["schedule"]["params"])
        out["deployed"][name] = {"under_C2": verdict(ev, chk, c2, fs), "under_C3": verdict(ev, chk, c3, fs)}
    # the deterministic next-shift anchor pool, evaluated under C2 (no revision)
    planner_cls = [c for c in vars(fe).values() if isinstance(c, type) and hasattr(c, "generate_next_shift")][0]
    planner = planner_cls(client=None, use_llm=False, config=FactoryScenarioConfig())
    routing = {"V_A": ["A", "B"], "V_B": ["A"], "V_C": ["A"]}; assignments = {"H1": "A", "H3": "B"}
    pool = planner.generate_next_shift(c2, routing, assignments)
    res = {fs.name: verdict(ev, chk, c2, fs) for fs in pool}
    out["anchor_pool_under_C2"] = res
    cert = [(v["throughput"], k) for k, v in res.items() if v["certified"]]
    out["best_certified_next_shift_anchor_under_C2"] = max(cert) if cert else None
    # the balanced (Phase-5) anchor pool under C2, for reference
    pool5 = planner.generate_balanced(c2, routing, assignments, ["V_A", "V_B", "V_C"], ["V_A"])
    res5 = {fs.name: verdict(ev, chk, c2, fs) for fs in pool5}
    cert5 = [(v["throughput"], k) for k, v in res5.items() if v["certified"]]
    out["best_certified_balanced_anchor_under_C2"] = max(cert5) if cert5 else None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("medoid_run", "c2_c3_same_hard_constraints", "c3_changes", "deployed",
                                          "best_certified_next_shift_anchor_under_C2", "best_certified_balanced_anchor_under_C2")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
