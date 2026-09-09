#!/usr/bin/env python3
"""Print the values the paper's single-run tables need from a repeated-run batch.

Selects the medoid run (closest to the per-phase medians of throughput,
fatigue, and noise over all shifts/phases) and prints, for its first shift,
the rows of the numerical-comparison table, the provenance table, the
Pareto/gate details, plus the batch-level summary lines.  Usage:

    python scripts/paper_values.py data/repeated/<batch_dir> [--shift 1]
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path


def medoid(root: Path) -> tuple[int, dict]:
    runs: dict[int, dict] = {}
    for p in sorted(glob.glob(str(root / "run_*/run_meta.json"))):
        m = json.load(open(p))
        if m.get("status") != "ok":
            continue
        recs = {}
        for sh in m["shifts"]:
            for ph in sh["phases"]:
                if ph.get("schedule") and ph.get("canonical") and ph["kpis"]:
                    recs[(sh["shift"], ph["canonical"])] = ph
        runs[m["run_index"]] = (m, recs)
    keys = sorted(set().union(*[set(r[1]) for r in runs.values()]))
    scale = {"throughput_uph": 1.0, "fatigue_index": 0.02, "noise_db": 0.3}
    med = {k: {f: statistics.median(runs[i][1][k]["kpis"][f] for i in runs if k in runs[i][1]) for f in scale} for k in keys}
    def dist(i):
        d = 0.0
        for k in keys:
            if k not in runs[i][1]:
                d += 10; continue
            for f, sc in scale.items():
                d += abs(runs[i][1][k]["kpis"][f] - med[k][f]) / sc
        return d
    best = min(runs, key=dist)
    return best, runs[best][0]


def phases_by_canonical(m: dict, shift: int = 1) -> dict:
    """Canonical name -> phase record of the given shift of a run (phases with a schedule)."""
    sh = next(s for s in m["shifts"] if s["shift"] == shift)
    return {ph["canonical"]: ph for ph in sh["phases"] if ph.get("canonical") and ph.get("schedule") and ph.get("kpis")}


def single_ref(root: Path, shift: int = 1) -> dict:
    """(throughput, V/R, fatigue, noise, P) per canonical phase of the medoid run (single cell)."""
    _, m = medoid(Path(root)); out = {}
    for name, ph in phases_by_canonical(m, shift).items():
        k = ph["kpis"]
        out[name] = (k["throughput_uph"], k.get("vr_score"), round(k["fatigue_index"], 3), k["noise_db"], ph.get("p_feasible"))
    return out


def factory_ref(root: Path) -> dict:
    """(throughput, max operator fatigue, noise, P) per canonical phase of the medoid factory run."""
    _, m = medoid(Path(root)); out = {}
    for name, ph in phases_by_canonical(m).items():
        k = ph["kpis"]
        out[name] = (k["throughput_uph"], round(k["fatigue_index"], 3), k["noise_db"], ph.get("p_feasible"))
    return out


def single_params(root: Path, shift: int = 1) -> dict:
    """Deployed/rejected schedule parameters of the medoid run, in the sensitivity script's format."""
    _, m = medoid(Path(root)); out = {}
    for name, ph in phases_by_canonical(m, shift).items():
        p = ph["schedule"]["params"]
        out[name] = dict(r1=p["r1_speed_fraction"], r2=p["r2_speed_fraction"], hr=p["human_cycle_rate_multiplier"],
                         buf=p["buffer_time_s"], demand=p["demand_target_uph"])
    return out


def factory_params(root: Path) -> dict:
    _, m = medoid(Path(root)); out = {}
    for name, ph in phases_by_canonical(m).items():
        p = ph["schedule"]["params"]; cells = p["cells"]
        tup = lambda c: (c["r1_speed_fraction"], c["r2_speed_fraction"], c["human_cycle_rate_multiplier"], c["buffer_time_s"])  # noqa: E731
        out[name] = dict(A=tup(cells["A"]), B=tup(cells["B"]), ops=p["operator_assignments"], routing=p["variant_routing"],
                         va=cells["A"]["assigned_variants"], vb=cells["B"]["assigned_variants"])
    return out


def export_stochastic(root: Path, out_path: Path, shift: int = 1) -> dict:
    """Write the stochastic-verification file (Figure 11) from the medoid run's per-shift audit trail."""
    idx, m = medoid(Path(root))
    run_dir = Path(root) / f"run_{idx:02d}"
    audit = run_dir / f"shift_{shift:02d}" / "audit_trail.json"
    if not audit.exists():
        audit = run_dir / "audit_trail.json"  # legacy batches: last shift only
    a = json.load(open(audit)); ents = a if isinstance(a, list) else a.get("entries", a)
    canon = {1: "S1", 3: "S2", 5: "S3"}
    scheds = {}
    for e in ents:
        if e["action"] in ("verification_passed", "schedule_rejected") and e["phase"] in canon:
            d = e["details"]; p = d.get("p_feasible")
            scheds[canon[e["phase"]]] = {
                "p_feasible": p, "violation_probs": d.get("constraint_violation_probabilities", {}),
                "worst_case_margins": d.get("worst_case_margins", {}), "certified": bool(p is not None and p >= 0.95)}
    # Rejected schedules carry no worst-case margins in the audit record; recompute them
    # with the same evaluator, contract, sample count, and seed.
    missing = [n for n, v in scheds.items() if not v.get("worst_case_margins")]
    if missing:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from cbpa.config.scenario import CellConfig
        from cbpa.layer3_verification.constraint_checker import ConstraintChecker
        from cbpa.models.contract import make_c1, make_c2
        from cbpa.models.schedule import Schedule
        from cbpa.physics.cell_evaluator import CellEvaluator
        ev = CellEvaluator(cell=CellConfig(), mode="analytical"); chk = ConstraintChecker()
        ph = phases_by_canonical(m, shift)
        for n in missing:
            if n not in ph:
                continue
            q = ph[n]["schedule"]["params"]
            sched = Schedule(name=n, r1_speed_fraction=q["r1_speed_fraction"], r2_speed_fraction=q["r2_speed_fraction"],
                             human_cycle_rate_multiplier=q["human_cycle_rate_multiplier"], buffer_time_s=q["buffer_time_s"],
                             demand_target_uph=q["demand_target_uph"])
            rep = chk.check_stochastic(make_c2() if n == "S3" else make_c1(), ev.evaluate_monte_carlo(sched, n_samples=200, seed=42), seed=42)
            scheds[n]["worst_case_margins"] = rep.worst_case_margins
            scheds[n]["worst_case_margins_recomputed"] = True
    payload = {"description": f"Stochastic verification (n=200, seed=42) of the medoid run {idx}, shift {shift}, from {audit}",
               "n_samples": 200, "seed": 42, "schedules": scheds}
    Path(out_path).write_text(json.dumps(payload, indent=1))
    return payload


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    root = Path(argv[1]); shift = int(argv[argv.index("--shift") + 1]) if "--shift" in argv else 1
    idx, m = medoid(root)
    print(f"medoid run: {idx}  ({root})")
    if "--export-stochastic" in argv:
        out = argv[argv.index("--export-stochastic") + 1]
        print("stochastic file:", json.dumps(export_stochastic(root, Path(out), shift)["schedules"]))
    sh = next(s for s in m["shifts"] if s["shift"] == shift)
    print(f"\n== shift {shift}: numerical table rows ==")
    for ph in sh["phases"]:
        if not (ph.get("schedule") and ph.get("canonical")):
            continue
        k = ph["kpis"]; pa = ph.get("pareto") or {}
        vr = f"{k['vr_score']:.3f}" if k.get("vr_score") is not None else "--"
        print(f"{ph['canonical']}: thr {k['throughput_uph']} defect {k.get('defect_rate')} noise {k['noise_db']} fatigue {k['fatigue_index']:.3f} "
              f"energy {k.get('energy_kwh')} gap {k.get('deadline_gap_pct')} vr {vr} feas {ph['feasible']} P {ph.get('p_feasible')} "
              f"| selected {pa.get('selected')} ({pa.get('selected_source')}) certified={pa.get('gate_certified')} tried={pa.get('gate_tried')} "
              f"params {ph['schedule']['params']}")
        src = pa.get("candidate_sources", {}); nd = set(pa.get("non_dominated", []))
        llm = [c for c, s in src.items() if s == "llm"]; det = [c for c, s in src.items() if s != "llm"]
        if src:
            print(f"    provenance: llm {len(llm)} det {len(det)} | on front llm {len([c for c in llm if c in nd])} det {len([c for c in det if c in nd])}")
        if ph.get("violations"):
            print(f"    violations: {ph['violations']}")
    if "summary" in sh:
        print("   shift summary:", sh["summary"])
    lc = json.load(open(root / "summary.json"))
    print("\n== batch summary ==")
    for ph in lc["phases"]:
        k = ph["kpis"]; st = ph["stability"]; fw = ph["firewall"]
        print(f"{ph['label']:14s} thr {k['throughput_uph']['mean']:.1f}±{k['throughput_uph']['sd']:.1f} "
              f"fat {k['fatigue_index']['mean']:.3f}±{k['fatigue_index']['sd']:.3f} noise {k['noise_db']['mean']:.1f}±{k['noise_db']['sd']:.1f} "
              f"feas {fw['selected_feasible_share']} cert {fw.get('certified_share')} P {round(fw['p_feasible']['mean'], 3) if fw['p_feasible'].get('n') else '--'} "
              f"llm_sel {st['llm_selected_share']:.1f} same {st['same_params_share']:.1f}")
    t = lc["llm_calls"]["total"]
    print("calls", t["calls"], "exc", t["exceptions"], "empty", t["empty_output"], "schema_inv", t["schema_invalid"], "lat", round(t["mean_latency_s"], 1),
          "| warnings", lc["llm_calls"]["warnings_by_category"], "| fallbacks", lc["llm_calls"]["fallbacks_by_component"])
    print("reinserted/canonicalised:", lc["llm_calls"]["contract_constraints_reinserted"])
    print("by_tool:", {k: (v["calls"], v["schema_invalid"]) for k, v in lc["llm_calls"]["by_tool"].items()})


if __name__ == "__main__":
    main(sys.argv)
