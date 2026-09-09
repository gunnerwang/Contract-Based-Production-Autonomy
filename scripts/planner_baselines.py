#!/usr/bin/env python3
"""Layer-2 baseline comparison: optimisation-based planner vs. the hybrid
LLM + anchor planner and the no-LLM anchors, per planning phase.

No LLM calls: the hybrid results are read from the repeated-run batches
(medoid single-cell run, factory batch) and the no-LLM values from the
deterministic runs; the optimiser is run here on the same physics, contract,
and decision space, once with deterministic feasibility only and once with
Monte-Carlo certification (P >= 0.95) as an additional constraint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from cbpa.baselines.optimizer_planner import FactoryOptimiserPlanner, OptimiserPlanner  # noqa: E402
from cbpa.config.scenario import CellConfig, FactoryConfig  # noqa: E402
from cbpa.layer2_planning.vr_scorer import VRScorer  # noqa: E402
from cbpa.layer3_verification.constraint_checker import ConstraintChecker  # noqa: E402
from cbpa.models.contract import make_c1, make_c1_factory, make_c2  # noqa: E402
from cbpa.physics.cell_evaluator import CellEvaluator  # noqa: E402
from cbpa.physics.factory_evaluator import FactoryEvaluator  # noqa: E402

# Hybrid (medoid run, corrected code) and no-LLM values from the reported runs
SINGLE_REF = {
    "S1": {"demand": 52.0, "contract": "C1", "hybrid": (40.7, 0.481, 0.350, 77.8, 0.995), "nollm": (40.7, 0.481, 0.350, 77.8, 0.995)},
    "S2": {"demand": 62.4, "contract": "C1", "hybrid": (46.2, 0.495, 0.591, 83.8, 0.0), "nollm": (48.2, 0.475, 0.600, 84.9, 0.0)},
    "S3": {"demand": 62.4, "contract": "C2", "hybrid": (42.3, 0.452, 0.397, 78.7, 0.585), "nollm": (42.3, 0.452, 0.397, 78.7, 0.585)},
}
FACTORY_REF = {
    "FS1": dict(hybrid=(84.4, 0.358, 80.0, 0.99), nollm=(84.4, 0.358, 80.0, 0.99), assignments={"H1": "A", "H2": "B"},
                routing={"V_A": ["A", "B"], "V_B": ["A", "B"], "V_C": ["A", "B"]}, va=["V_A", "V_B", "V_C"], vb=["V_A", "V_B", "V_C"], da=52.0, db=48.0),
    "FS2": dict(hybrid=(99.7, 0.825, 86.6, 0.0), nollm=(86.8, 0.494, 81.9, 0.0), assignments={"H1": "A", "H3": "B"},
                routing={"V_A": ["A", "B"], "V_B": ["A"], "V_C": ["A"]}, va=["V_A", "V_B", "V_C"], vb=["V_A"], da=62.4, db=57.6),
    "FS3": dict(hybrid=(75.8, 0.348, 78.8, 1.0), nollm=(75.8, 0.348, 78.8, 1.0), assignments={"H1": "A", "H3": "B"},
                routing={"V_A": ["A", "B"], "V_B": ["A"], "V_C": ["A"]}, va=["V_A", "V_B", "V_C"], vb=["V_A"], da=52.0, db=48.0),
    "FS4": dict(hybrid=(81.7, 0.388, 80.6, 0.955), nollm=(81.7, 0.388, 80.6, 0.955), assignments={"H1": "A", "H3": "B"},
                routing={"V_A": ["A", "B"], "V_B": ["A"], "V_C": ["A"]}, va=["V_A", "V_B", "V_C"], vb=["V_A"], da=56.0, db=44.0),
}


def load_refs(single_root=None, factory_root=None, single_nollm=None, factory_nollm=None) -> None:
    """Replace the hard-coded hybrid / no-LLM reference values by those of the medoid runs of the given batches."""
    from paper_values import factory_ref, single_ref
    if single_root:
        for name, tup in single_ref(single_root).items():
            if name in SINGLE_REF: SINGLE_REF[name]["hybrid"] = tup
    if single_nollm:
        for name, tup in single_ref(single_nollm).items():
            if name in SINGLE_REF: SINGLE_REF[name]["nollm"] = tup
    if factory_root:
        for name, tup in factory_ref(factory_root).items():
            if name in FACTORY_REF: FACTORY_REF[name]["hybrid"] = tup
    if factory_nollm:
        for name, tup in factory_ref(factory_nollm).items():
            if name in FACTORY_REF: FACTORY_REF[name]["nollm"] = tup


def run(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    checker = ConstraintChecker()
    ev = CellEvaluator(cell=CellConfig(), mode="analytical")
    scorer = VRScorer(mode="analytical")
    res: dict = {"single": {}, "factory": {}}
    for name, ref in SINGLE_REF.items():
        contract = make_c1() if ref["contract"] == "C1" else make_c2()
        row = {"hybrid": ref["hybrid"], "nollm": ref["nollm"]}
        for mode, certify in (("opt_det", False), ("opt_cert", True)):
            r = OptimiserPlanner(ev, scorer, checker, certify=certify).plan(contract, ref["demand"], name=f"{name}_opt")
            if r.schedule is None:
                row[mode] = None; continue
            m = ev.evaluate(r.schedule)
            row[mode] = {"throughput": m.throughput_uph, "vr": r.objective, "fatigue": m.fatigue_index, "noise": m.noise_db,
                         "p_feasible": r.p_feasible, "evaluations": r.evaluations, "seconds": r.seconds,
                         "params": {"r1": round(r.schedule.r1_speed_fraction, 3), "r2": round(r.schedule.r2_speed_fraction, 3),
                                    "hr": round(r.schedule.human_cycle_rate_multiplier, 3), "buf": round(r.schedule.buffer_time_s, 2)}}
        res["single"][name] = row
        print(name, json.dumps(row, default=str)[:400])
    fev = FactoryEvaluator(factory=FactoryConfig(), mode="analytical")
    fc = make_c1_factory()
    for name, ref in FACTORY_REF.items():
        row = {"hybrid": ref["hybrid"], "nollm": ref["nollm"]}
        for mode, certify in (("opt_det", False), ("opt_cert", True)):
            r = FactoryOptimiserPlanner(fev, checker, certify=certify).plan(
                fc, ref["routing"], ref["assignments"], ref["va"], ref["vb"], ref["da"], ref["db"], name=f"{name}_opt")
            if r.schedule is None:
                row[mode] = None; continue
            m = fev.evaluate(r.schedule)
            fat = m.operator_fatigue
            row[mode] = {"throughput": m.total_throughput_uph, "fatigue": max(fat.values()), "noise": m.factory_noise_db,
                         "p_feasible": r.p_feasible, "evaluations": r.evaluations, "seconds": r.seconds,
                         "params": {cid: [round(cs.r1_speed_fraction, 3), round(cs.r2_speed_fraction, 3),
                                          round(cs.human_cycle_rate_multiplier, 3), round(cs.buffer_time_s, 2)]
                                    for cid, cs in r.schedule.cell_schedules.items()}}
        res["factory"][name] = row
        print(name, json.dumps(row, default=str)[:400])
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Layer-2 optimiser baseline vs. hybrid and no-LLM planners")
    ap.add_argument("out", nargs="?", default=str(_HERE.parent / "data" / "baselines" / "planner"))
    ap.add_argument("--single-root", help="single-cell repeated batch (medoid run gives the hybrid values)")
    ap.add_argument("--factory-root", help="factory repeated batch")
    ap.add_argument("--single-nollm-root", help="single-cell no-LLM batch")
    ap.add_argument("--factory-nollm-root", help="factory no-LLM batch")
    a = ap.parse_args()
    load_refs(a.single_root, a.factory_root, a.single_nollm_root, a.factory_nollm_root)
    run(Path(a.out))
