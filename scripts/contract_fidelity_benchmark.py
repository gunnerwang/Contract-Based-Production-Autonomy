#!/usr/bin/env python3
"""Contract-fidelity benchmark for Layer-1 elicitation.

Thirty manager intents in six categories, each with a ground-truth contract
(KPIs, hard constraints with limits, priority order, assumptions, and whether
the intent is deliberately ambiguous), are run through three elicitors:

* template  — fixed form with plant defaults and explicit numeric overrides;
* rule_based — keyword-to-ontology parser with synonyms and numeric extraction;
* llm — the CBPA elicitation agent (propose-and-refine), followed by the same
  ontology canonicalisation the prototype applies.

All three outputs pass through one canonicaliser (constraint and KPI names
mapped to the plant ontology, safety limits clamped to the plant floors) so
that vocabulary alone does not decide the score.  Metrics per elicitor:
constraint recall, limit fidelity, hard-constraint preservation (limit not
looser than the truth), KPI F1, top-priority accuracy, assumption recall,
and ambiguity detection on the ambiguous subset (plus false flags elsewhere).

Usage:  python scripts/contract_fidelity_benchmark.py [--llm] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from cbpa.baselines.elicitation_baselines import PLANT_FLOORS, RuleBasedElicitor, TemplateElicitor  # noqa: E402
from cbpa.models.contract import KPIDirection, OutcomeContract, PriorityLevel  # noqa: E402

# ── benchmark intents and ground truth ──────────────────────────────────
HW, SA, QU, TH, CO, FL = (PriorityLevel.HUMAN_WELLBEING, PriorityLevel.SAFETY, PriorityLevel.QUALITY,
                          PriorityLevel.THROUGHPUT, PriorityLevel.COST, PriorityLevel.FLEXIBILITY)
DEF_PRIO = [HW, SA, QU, TH]


def gt(kpis, constraints, priority=None, assumptions=None, ambiguous=False, category=""):
    return {"kpis": kpis, "constraints": {**{"FatigueIndex": 0.4, "Noise": 80.0}, **constraints},
            "priority": priority or DEF_PRIO, "assumptions": assumptions or {}, "ambiguous": ambiguous,
            "category": category}


INTENTS = [
    # A — explicit numeric limits
    ("A", "Maximise throughput this shift. Keep the operator fatigue index below 0.35 and noise under 78 dB. Defect rate must stay under 0.5%.",
     gt({"Throughput": "max", "DefectRate": "bound"}, {"FatigueIndex": 0.35, "Noise": 78.0})),
    ("A", "Increase output; fatigue index at most 0.30; noise at most 75 dB.",
     gt({"Throughput": "max"}, {"FatigueIndex": 0.30, "Noise": 75.0})),
    ("A", "Run the cell at full pace but keep noise under 79 dB and fatigue under 0.38, and hold the defect rate under 1%.",
     gt({"Throughput": "max", "DefectRate": "bound"}, {"FatigueIndex": 0.38, "Noise": 79.0})),
    ("A", "Target maximum units per hour with the fatigue index no more than 0.25 and the noise level no more than 77 dB.",
     gt({"Throughput": "max"}, {"FatigueIndex": 0.25, "Noise": 77.0})),
    ("A", "Maximise delivered value per resource for this shift; noise under 80 dB; fatigue under 0.4; cyber risk level at most 2.",
     gt({"Throughput": "max"}, {"CyberRiskLevel": 2.0})),
    # B — implicit / qualitative wording
    ("B", "Push output as hard as the team can take, but nobody should finish the shift exhausted and the hall must stay quiet enough for normal conversation.",
     gt({"Throughput": "max"}, {}, [HW, SA, QU, TH])),
    ("B", "We need more units out today. Keep the operators fresh and the cell reasonably quiet.",
     gt({"Throughput": "max"}, {})),
    ("B", "Get the most out of the line without wearing the operator down or making the workplace loud.",
     gt({"Throughput": "max"}, {})),
    ("B", "Raise productivity for the afternoon; operator well-being comes first, noise must stay within the usual limit.",
     gt({"Throughput": "max"}, {}, [HW, SA, QU, TH])),
    ("B", "Increase delivered value per resource consumed for this shift, while keeping operator fatigue low and noise under 80dB.",
     gt({"Throughput": "max"}, {})),
    # C — paraphrase / synonyms
    ("C", "Increase productivity; acoustic level not above 76 decibels; keep operator strain within 0.30.",
     gt({"Throughput": "max"}, {"FatigueIndex": 0.30, "Noise": 76.0})),
    ("C", "Boost the production rate; sound level no more than 74 dB; the workload index must stay within 0.32.",
     gt({"Throughput": "max"}, {"FatigueIndex": 0.32, "Noise": 74.0})),
    ("C", "Lift units per hour; keep the decibel reading under 78; operator exhaustion index under 0.36.",
     gt({"Throughput": "max"}, {"FatigueIndex": 0.36, "Noise": 78.0})),
    ("C", "More output per hour, scrap under 0.8%, acoustic exposure not above 77 dB.",
     gt({"Throughput": "max", "DefectRate": "bound"}, {"Noise": 77.0})),
    ("C", "Improve first-pass yield to at least 99.5% while lifting the production rate; loudness under 79 dB.",
     gt({"Throughput": "max", "DefectRate": "bound"}, {"Noise": 79.0})),
    # D — unanticipated constraints (energy, robot speed, overtime)
    ("D", "Maximise output; energy no more than 380 kWh for the shift; robot arms never faster than 1.2 m/s; no overtime.",
     gt({"Throughput": "max"}, {"Energy": 380.0, "ShiftHours": 8.0}, assumptions={"R1.speed": 1.2})),
    ("D", "Increase throughput but cap energy at 400 kWh and keep noise under 78 dB.",
     gt({"Throughput": "max"}, {"Energy": 400.0, "Noise": 78.0})),
    ("D", "Run faster, but the robot speed must stay under 1.0 m/s near the operator and network latency under 15 ms.",
     gt({"Throughput": "max"}, {}, assumptions={"R1.speed": 1.0, "NetworkLatency": 15.0})),
    ("D", "Maximise units; no overtime beyond the 8-hour shift; fatigue under 0.35; energy capped at 360 kWh.",
     gt({"Throughput": "max"}, {"FatigueIndex": 0.35, "Energy": 360.0, "ShiftHours": 8.0})),
    ("D", "Deliver as much as possible with the changeover time minimised; electricity below 390 kWh; noise below 80 dB.",
     gt({"Throughput": "max", "ChangeoverTime": "min"}, {"Energy": 390.0})),
    # E — priority statements
    ("E", "Hit the delivery deadline even if it costs some quality, but never trade operator well-being.",
     gt({"Throughput": "max"}, {}, [HW, SA, TH, QU])),
    ("E", "Quality over throughput this shift: zero tolerance on defects, output second, operators protected as always.",
     gt({"Throughput": "max", "DefectRate": "bound"}, {}, [HW, SA, QU, TH])),
    ("E", "Prioritise safety above all; then throughput; cost does not matter today.",
     gt({"Throughput": "max"}, {}, [SA, HW, TH, QU])),
    ("E", "Throughput first today; quality second; keep fatigue under 0.4 and noise under 80 dB as hard limits.",
     gt({"Throughput": "max"}, {}, [HW, SA, TH, QU])),
    ("E", "Put operator comfort ahead of the deadline; noise under 75 dB; fatigue under 0.3.",
     gt({"Throughput": "max"}, {"FatigueIndex": 0.30, "Noise": 75.0}, [HW, SA, QU, TH])),
    # F — ambiguous / conflicting
    ("F", "Maximise throughput and minimise energy at the same time, and keep the deadline.",
     gt({"Throughput": "max", "Energy": "min"}, {}, ambiguous=True)),
    ("F", "Increase output by 20% today; operators are already tired.",
     gt({"Throughput": "max"}, {}, ambiguous=True)),
    ("F", "Run the cell as fast as possible; noise is not a big concern; fatigue should be fine.",
     gt({"Throughput": "max"}, {}, ambiguous=True)),
    ("F", "Meet the rush order; quality can slip a little; not sure how far.",
     gt({"Throughput": "max", "DefectRate": "bound"}, {}, [HW, SA, TH, QU], ambiguous=True)),
    ("F", "Maximise value per resource this shift.",
     gt({"Throughput": "max"}, {}, ambiguous=True)),
]

# ── canonicalisation shared by all elicitors ──────────────────────────
_CANON_C = [("fatigue", "FatigueIndex"), ("strain", "FatigueIndex"), ("exhaust", "FatigueIndex"), ("workload", "FatigueIndex"),
            ("noise", "Noise"), ("acoustic", "Noise"), ("sound", "Noise"), ("decibel", "Noise"), ("loud", "Noise"), ("db", "Noise"),
            ("energy", "Energy"), ("kwh", "Energy"), ("electric", "Energy"), ("power", "Energy"),
            ("cyber", "CyberRiskLevel"), ("security", "CyberRiskLevel"),
            ("overtime", "ShiftHours"), ("shift", "ShiftHours"), ("hour", "ShiftHours"),
            ("speed", "R1.speed"), ("m/s", "R1.speed"), ("velocity", "R1.speed"),
            ("latency", "NetworkLatency"),
            ("defect", "DefectRate"), ("scrap", "DefectRate"), ("yield", "DefectRate"), ("quality", "DefectRate")]
_CANON_K = [("throughput", "Throughput"), ("output", "Throughput"), ("units", "Throughput"), ("productiv", "Throughput"),
            ("production", "Throughput"), ("value", "Throughput"), ("uph", "Throughput"),
            ("defect", "DefectRate"), ("scrap", "DefectRate"), ("yield", "DefectRate"), ("quality", "DefectRate"),
            ("changeover", "ChangeoverTime"), ("setup", "ChangeoverTime"),
            ("energy", "Energy"), ("kwh", "Energy"), ("cost", "Cost"), ("downtime", "Downtime")]
ASSUMPTION_NAMES = {"R1.speed", "NetworkLatency"}


def canon_name(name: str, table) -> str | None:
    low = name.lower()
    for kw, canon in table:
        if kw in low:
            return canon
    return None


def canonicalise(c: OutcomeContract) -> dict:
    """Return {kpis: {canon: dir}, constraints: {canon: limit}, assumptions: {canon: value}, priority: [...]}."""
    kpis: dict[str, str] = {}
    for k in c.kpi_targets:
        cn = canon_name(k.name, _CANON_K)
        if cn is None:
            continue
        d = "max" if k.direction == KPIDirection.MAXIMIZE else ("min" if k.direction == KPIDirection.MINIMIZE else "bound")
        kpis.setdefault(cn, d)
    cons: dict[str, float] = {}
    assum: dict[str, float] = {}
    for hc in c.hard_constraints:
        cn = canon_name(hc.name, _CANON_C)
        if cn is None:
            continue
        if cn in ASSUMPTION_NAMES:
            assum[cn] = min(assum.get(cn, hc.limit), hc.limit); continue
        lim = hc.limit
        if cn == "DefectRate":  # a defect-rate hard constraint is a bounded KPI in the ontology
            kpis.setdefault("DefectRate", "bound"); continue
        if cn in PLANT_FLOORS:
            lim = min(lim, PLANT_FLOORS[cn])
        cons[cn] = min(cons.get(cn, lim), lim)
    for a in c.typed_assumptions:
        cn = canon_name(a.name, _CANON_C)
        if cn in ASSUMPTION_NAMES:
            assum[cn] = a.expected_value
    for k, v in (c.assumptions or {}).items():
        cn = canon_name(str(k), _CANON_C)
        if cn in ASSUMPTION_NAMES:
            try:
                assum[cn] = float(re.findall(r"\d+(?:\.\d+)?", str(v))[0])
            except (IndexError, ValueError):
                pass
    if c.context.get("overtime_allowed") is False and "ShiftHours" not in cons:
        cons["ShiftHours"] = float(c.context.get("shift_hours", 8))
    for canon, floor in PLANT_FLOORS.items():   # ontology validation re-inserts mandatory floors
        cons.setdefault(canon, floor)
    return {"kpis": kpis, "constraints": cons, "assumptions": assum, "priority": [p for p in c.priority_order]}


# ── scoring ──────────────────────────────────────────────────────────
def score(pred: dict, truth: dict, n_ambig: int) -> dict:
    tc, pc = truth["constraints"], pred["constraints"]
    found = [n for n in tc if n in pc]
    exact = [n for n in found if abs(pc[n] - tc[n]) <= 0.01 * max(abs(tc[n]), 1e-9) + 1e-9]
    preserved = [n for n in found if pc[n] <= tc[n] + 1e-9]
    # explicit (non-floor) constraints are the discriminating ones
    explicit = [n for n, v in tc.items() if not (n in PLANT_FLOORS and v == PLANT_FLOORS[n])]
    exp_found = [n for n in explicit if n in pc]
    exp_exact = [n for n in exp_found if abs(pc[n] - tc[n]) <= 0.01 * max(abs(tc[n]), 1e-9) + 1e-9]
    tk, pk = truth["kpis"], pred["kpis"]
    def dir_ok(a, b):
        return a == b or {a, b} <= {"min", "bound"}
    kpi_tp = [n for n in tk if n in pk and dir_ok(tk[n], pk[n])]
    kpi_prec = len(kpi_tp) / len(pk) if pk else 0.0
    kpi_rec = len(kpi_tp) / len(tk) if tk else 1.0
    kpi_f1 = (2 * kpi_prec * kpi_rec / (kpi_prec + kpi_rec)) if (kpi_prec + kpi_rec) else 0.0
    ta, pa = truth["assumptions"], pred["assumptions"]
    a_found = [n for n in ta if n in pa and abs(pa[n] - ta[n]) <= 0.01 * ta[n] + 1e-9]
    tp_, pp_ = truth["priority"], pred["priority"]
    return {
        "constraint_recall": len(found) / len(tc),
        "limit_fidelity": len(exact) / len(tc),
        "preservation": len(preserved) / len(tc),
        "explicit_recall": (len(exp_found) / len(explicit)) if explicit else None,
        "explicit_fidelity": (len(exp_exact) / len(explicit)) if explicit else None,
        "constraint_precision": (len([n for n in pc if n in tc]) / len(pc)) if pc else 0.0,
        "kpi_f1": kpi_f1,
        "top_priority": float(bool(pp_) and pp_[0] == tp_[0]),
        "priority_exact": float(list(pp_[:len(tp_)]) == list(tp_)),
        "assumption_recall": (len(a_found) / len(ta)) if ta else None,
        "ambiguities_flagged": n_ambig,
    }


def aggregate(rows: list[dict]) -> dict:
    keys = ["constraint_recall", "limit_fidelity", "preservation", "explicit_recall", "explicit_fidelity",
            "constraint_precision", "kpi_f1", "top_priority", "priority_exact", "assumption_recall"]
    out = {}
    for k in keys:
        vals = [r["scores"][k] for r in rows if r["scores"][k] is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    amb = [r for r in rows if r["truth"]["ambiguous"]]
    non = [r for r in rows if not r["truth"]["ambiguous"]]
    out["ambiguity_detection_rate"] = (sum(1 for r in amb if r["scores"]["ambiguities_flagged"] > 0) / len(amb)) if amb else None
    out["ambiguity_flag_rate_nonambiguous"] = (sum(1 for r in non if r["scores"]["ambiguities_flagged"] > 0) / len(non)) if non else None
    out["mean_ambiguities_per_intent"] = sum(r["scores"]["ambiguities_flagged"] for r in rows) / len(rows)
    out["n"] = len(rows)
    by_cat = {}
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        by_cat[cat] = {k: (sum(r["scores"][k] for r in sub if r["scores"][k] is not None) /
                           max(1, len([r for r in sub if r["scores"][k] is not None]))) for k in ("constraint_recall", "limit_fidelity", "preservation", "kpi_f1", "top_priority")}
    out["by_category"] = by_cat
    return out


def run(out_dir: Path, use_llm: bool, provider: str, model: str | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    elicitors: dict = {"template": TemplateElicitor(), "rule_based": RuleBasedElicitor()}
    llm_client = None
    if use_llm:
        from cbpa.layer1_outcome.elicitation_agent import ElicitationAgent
        from cbpa.llm.client import create_client
        from cbpa.llm.instrumented import InstrumentedLLMClient
        raw = create_client(provider=provider, model=model)
        ok, msg = raw.check_connection(); print("LLM:", msg)
        if not ok:
            sys.exit(2)
        llm_client = InstrumentedLLMClient(raw)
        elicitors["llm"] = ElicitationAgent(llm_client, use_llm=True)
    results: dict = {"intents": [], "summary": {}, "llm": None}
    per_elicitor: dict[str, list] = {k: [] for k in elicitors}
    for i, (cat, intent, truth) in enumerate(INTENTS):
        entry = {"id": i, "category": cat, "intent": intent, "truth": {**truth, "priority": [p.value for p in truth["priority"]]}, "elicitors": {}}
        for name, el in elicitors.items():
            t0 = time.perf_counter(); n_amb = 0; err = None
            try:
                if name == "llm":
                    c, amb = el.elicit_with_refinement(intent, "C1"); n_amb = len(amb)
                else:
                    c = el.elicit(intent, "C1")
                pred = canonicalise(c)
                raw_names = [hc.name for hc in c.hard_constraints]
            except Exception as exc:  # keep going; count as empty contract
                err = f"{type(exc).__name__}: {exc}"; pred = {"kpis": {}, "constraints": dict(PLANT_FLOORS), "assumptions": {}, "priority": []}; raw_names = []
            sc = score(pred, {**truth, "priority": truth["priority"]}, n_amb)
            rec = {"scores": sc, "pred": {**pred, "priority": [getattr(p, "value", str(p)) for p in pred["priority"]]},
                   "raw_constraint_names": raw_names, "seconds": round(time.perf_counter() - t0, 1), "error": err,
                   "truth": truth, "category": cat}
            per_elicitor[name].append(rec)
            entry["elicitors"][name] = {k: v for k, v in rec.items() if k != "truth"}
            print(f"[{i:02d}/{cat}] {name:10s} recall={sc['constraint_recall']:.2f} fid={sc['limit_fidelity']:.2f} "
                  f"pres={sc['preservation']:.2f} kpiF1={sc['kpi_f1']:.2f} top={sc['top_priority']:.0f} amb={n_amb} {('ERR ' + err) if err else ''}")
        results["intents"].append(entry)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    results["summary"] = {name: aggregate(rows) for name, rows in per_elicitor.items()}
    if llm_client is not None:
        results["llm"] = {**llm_client.sampling_settings(), "calls": llm_client.stats()}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results["summary"], indent=1, default=str))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true"); ap.add_argument("--provider", default="claude")
    ap.add_argument("--model", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(Path(a.out) if a.out else _HERE.parent / "data" / "baselines" / ("fidelity_llm" if a.llm else "fidelity_nollm"), a.llm, a.provider, a.model)
