#!/usr/bin/env python3
"""Aggregate repeated CBPA runs produced by ``run_repeated.py``.

Outputs (in the batch root):

* ``summary.json``        — everything below in machine-readable form
* ``summary_table.csv``   — per phase x KPI: n, mean, SD, 95 % CI, min, max
* ``summary_table.tex``   — booktabs table (mean ± SD, feasibility, LLM share,
                            strategy stability) ready to paste into the paper
* ``figures/variability_boxplots.{pdf,png}``
* ``figures/strategy_stability.{pdf,png}``

Statistics
----------
* mean, sample SD, and a t-based 95 % confidence interval of every KPI of the
  *selected* schedule, per planning phase (and per shift for multi-shift runs);
* **strategy stability**: share of runs selecting (a) the same candidate name,
  (b) a parameter vector within tolerance of the medoid (|Δ| ≤ 0.02 on speed
  fractions and cycle-rate multipliers, ≤ 0.5 s on buffer times), and (c) a
  KPI-equivalent schedule (throughput within ±2 % of the median);
* **firewall statistics** per provenance (LLM vs deterministic): candidates
  generated, passing the deterministic constraint check, on the Pareto front,
  selected; Monte-Carlo ``p_feasible`` of the selected schedule;
* **LLM call statistics**: calls, transport exceptions, empty outputs,
  schema violations, latency, and deterministic fallbacks by component.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cbpa.llm.instrumented import FallbackLogCapture  # noqa: E402

# Student-t 97.5 % quantiles (df 1..30); beyond, the normal quantile is used.
_T975 = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
]

KPI_ORDER = [
    "throughput_uph", "vr_score", "fatigue_index", "noise_db", "energy_kwh",
    "defect_rate", "deadline_gap_pct", "cell_balance_loss_pct", "agv_utilization",
]
KPI_LABEL = {
    "throughput_uph": "Throughput (u/h)",
    "vr_score": "V/R score",
    "fatigue_index": "Fatigue index (max)",
    "noise_db": "Noise (dB)",
    "energy_kwh": "Energy (kWh)",
    "defect_rate": "Defect rate",
    "deadline_gap_pct": "Deadline gap (%)",
    "cell_balance_loss_pct": "Cell balance loss (%)",
    "agv_utilization": "AGV utilisation",
}
PARAM_TOL = {
    "r1_speed_fraction": 0.02,
    "r2_speed_fraction": 0.02,
    "human_cycle_rate_multiplier": 0.02,
    "buffer_time_s": 0.5,
}
# Hard limits known from the scenario configuration (drawn as dashed lines)
LIMITS = {
    "factory": {"fatigue_index": 0.40, "noise_db": 82.0},
    "single": {},
}


# ── basic statistics ─────────────────────────────────────────────────────

def t_crit(df: int) -> float:
    if df <= 0:
        return float("nan")
    return _T975[df - 1] if df <= len(_T975) else 1.96


def describe(values: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    half = t_crit(n - 1) * se if n > 1 else 0.0
    return {
        "n": n, "mean": mean, "sd": sd, "se": se,
        "ci95_low": mean - half, "ci95_high": mean + half, "ci95_half": half,
        "min": min(vals), "max": max(vals), "median": statistics.median(vals),
        "cv_pct": (100.0 * sd / mean) if mean else None,
    }


# ── loading ───────────────────────────────────────────────────────────────

def _pareto_from_audit_file(audit_path: Path) -> dict[int, dict]:
    """Rebuild per-phase candidate pools from an exported audit_trail.json."""
    pools: dict[int, dict] = {}
    try:
        entries = json.loads(audit_path.read_text())
    except Exception:
        return pools
    for e in entries:
        d = e.get("details", {}) or {}
        if "candidate_sources" not in d:
            continue
        scores = d.get("scores") or {}
        pools[int(e["phase"])] = {
            "candidates": list(scores.keys()) or list(d.get("candidate_sources", {}).keys()),
            "non_dominated": list(d.get("non_dominated") or d.get("pareto_front") or []),
            "selected": d.get("selected"),
            "selected_source": d.get("selected_source"),
            "candidate_sources": dict(d.get("candidate_sources", {})),
            "scores": {k: dict(v) for k, v in scores.items()},
            "gate_certified": d.get("gate_certified"),
            "gate_p_feasible": d.get("gate_p_feasible"),
            "gate_tried": list(d.get("gate_tried", []) or []),
        }
    return pools


def load_runs(root: Path) -> tuple[list[dict], list[dict]]:
    ok, bad = [], []
    for meta_path in sorted(root.glob("run_*/run_meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:
            bad.append({"path": str(meta_path), "error": str(exc)})
            continue
        if meta.get("status") != "ok":
            bad.append(meta)
            continue
        # Repair candidate pools recorded before the audit keys were unified
        # (single-shift runs only: the exported audit trail is the last shift's).
        shifts = meta.get("shifts", [])
        if len(shifts) == 1:
            pools = _pareto_from_audit_file(meta_path.parent / "audit_trail.json")
            for rec in shifts[0].get("phases", []):
                pool = pools.get(int(rec["phase"]))
                cur = rec.get("pareto") or {}
                if pool and (not cur.get("non_dominated") or not cur.get("scores")):
                    rec["pareto"] = pool
        ok.append(meta)
    return ok, bad


def phase_groups(runs: list[dict]) -> dict[tuple[int, int], list[tuple[int, dict]]]:
    """(shift, phase) -> [(run_index, phase_record)] for phases with a schedule."""
    groups: dict[tuple[int, int], list[tuple[int, dict]]] = {}
    for meta in runs:
        for sh in meta.get("shifts", []):
            for rec in sh.get("phases", []):
                # Only planning phases (those with a canonical schedule label);
                # monitoring / escalation phases merely carry the deployed schedule.
                if rec.get("schedule") is None or rec.get("kpis") is None or not rec.get("canonical"):
                    continue
                groups.setdefault((int(sh["shift"]), int(rec["phase"])), []).append(
                    (int(meta["run_index"]), rec)
                )
    return dict(sorted(groups.items()))


# ── strategy stability ───────────────────────────────────────────────────

def _flatten_params(params: dict | None) -> list[tuple[str, float]]:
    """Ordered (name, value) pairs of the numeric decision variables."""
    if not params:
        return []
    if "cells" in params:  # factory
        out: list[tuple[str, float]] = []
        for cid, cp in sorted(params["cells"].items()):
            for k in PARAM_TOL:
                if k in cp:
                    out.append((f"{cid}.{k}", float(cp[k])))
        return out
    return [(k, float(params[k])) for k in PARAM_TOL if k in params]


def _tol_of(name: str) -> float:
    return PARAM_TOL[name.split(".")[-1]]


def _within_tol(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> bool:
    if len(a) != len(b) or not a:
        return False
    return all(abs(x - y) <= _tol_of(n) for (n, x), (_, y) in zip(a, b))


def _scaled_l1(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> float:
    if len(a) != len(b):
        return float("inf")
    return sum(abs(x - y) / _tol_of(n) for (n, x), (_, y) in zip(a, b))


def selection_source(rec: dict) -> str:
    """Provenance of the Pareto-selected candidate (authoritative: the audit
    record; the deployed schedule object's ``source`` field is not reliably
    propagated by the single-cell runner)."""
    pareto = rec.get("pareto") or {}
    src = pareto.get("selected_source")
    if src in ("llm", "deterministic"):
        return src
    return rec["schedule"].get("source", "deterministic")


def stability(records: list[dict]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0}
    names = [r["schedule"]["name"] for r in records]
    sources = [selection_source(r) for r in records]
    modal_name = max(set(names), key=names.count)
    vecs = [_flatten_params(r["schedule"].get("params")) for r in records]
    medoid_idx = min(range(n), key=lambda i: sum(_scaled_l1(vecs[i], v) for v in vecs))
    same_params = sum(1 for v in vecs if _within_tol(v, vecs[medoid_idx]))
    thr = [r["kpis"].get("throughput_uph") for r in records if r["kpis"].get("throughput_uph") is not None]
    med_thr = statistics.median(thr) if thr else None
    kpi_equiv = (
        sum(1 for t in thr if med_thr and abs(t - med_thr) <= 0.02 * abs(med_thr)) if thr else 0
    )
    return {
        "n": n,
        "modal_name": modal_name,
        "same_name_share": names.count(modal_name) / n,
        "distinct_names": len(set(names)),
        "medoid_run_offset": medoid_idx,
        "medoid_params": dict(vecs[medoid_idx]),
        "same_params_share": same_params / n,
        "kpi_equivalent_share": (kpi_equiv / len(thr)) if thr else None,
        "selected_source_counts": {s: sources.count(s) for s in sorted(set(sources))},
        "llm_selected_share": sources.count("llm") / n,
    }


# ── firewall / provenance statistics ─────────────────────────────────────

def firewall_stats(records: list[dict]) -> dict[str, Any]:
    agg = {
        src: {"generated": 0, "pass_deterministic_check": 0, "on_pareto_front": 0, "selected": 0}
        for src in ("llm", "deterministic")
    }
    pf_vals = [r.get("p_feasible") for r in records if r.get("p_feasible") is not None]
    feas = [r.get("feasible") for r in records if r.get("feasible") is not None]
    n_with_pool = 0
    for r in records:
        pareto = r.get("pareto")
        if not pareto:
            continue
        n_with_pool += 1
        srcs = pareto.get("candidate_sources", {})
        scores = pareto.get("scores", {})
        nd = set(pareto.get("non_dominated", []))
        for name, src in srcs.items():
            src = src if src in agg else "deterministic"
            agg[src]["generated"] += 1
            sc = scores.get(name, {})
            if bool(sc.get("feasible", 0)):
                agg[src]["pass_deterministic_check"] += 1
            if name in nd:
                agg[src]["on_pareto_front"] += 1
        sel_src = pareto.get("selected_source")
        if sel_src in agg:
            agg[sel_src]["selected"] += 1
    has_scores = any((r.get("pareto") or {}).get("scores") for r in records)
    for src, d in agg.items():
        g = d["generated"]
        d["pass_rate"] = (d["pass_deterministic_check"] / g) if (g and has_scores) else None
        if not has_scores:
            d["pass_deterministic_check"] = None
        d["front_rate"] = d["on_pareto_front"] / g if g else None
    cert = [r["pareto"].get("gate_certified") for r in records if (r.get("pareto") or {}).get("gate_certified") is not None]
    return {
        "runs_with_candidate_pool": n_with_pool,
        "certified_share": (sum(1 for c in cert if c) / len(cert)) if cert else None,
        "by_source": agg,
        "selected_feasible_share": (sum(1 for f in feas if f) / len(feas)) if feas else None,
        "p_feasible": describe(pf_vals),
        "p_feasible_certified_share": (sum(1 for p in pf_vals if p >= 0.95) / len(pf_vals)) if pf_vals else None,
    }


# ── LLM call statistics ──────────────────────────────────────────────────

def llm_call_stats(runs: list[dict]) -> dict[str, Any] | None:
    totals: dict[str, Any] = {"calls": 0, "exceptions": 0, "empty_output": 0, "schema_invalid": 0,
                              "schema_checked": 0, "total_latency_s": 0.0}
    by_tool: dict[str, dict[str, Any]] = {}
    fallbacks: dict[str, int] = {}
    categories: dict[str, int] = {}
    repaired: dict[str, int] = {}
    any_llm = False
    per_run_calls: list[int] = []
    for meta in runs:
        calls = meta.get("llm_calls")
        if not calls:
            continue
        any_llm = True
        t = calls.get("total", {})
        per_run_calls.append(int(t.get("calls", 0)))
        for k in list(totals):
            totals[k] += t.get(k, 0) or 0
        for tool, d in calls.get("by_tool", {}).items():
            acc = by_tool.setdefault(tool, {"calls": 0, "exceptions": 0, "empty_output": 0,
                                            "schema_invalid": 0, "total_latency_s": 0.0})
            for k in list(acc):
                acc[k] += d.get(k, 0) or 0
        for ev in meta.get("fallbacks", {}).get("events") or []:
            cat = ev.get("category") or FallbackLogCapture.classify(ev.get("message", "")) or "other"
            categories[cat] = categories.get(cat, 0) + 1
            if cat == "fallback":
                comp = ev["logger"].split(".")[-1]
                fallbacks[comp] = fallbacks.get(comp, 0) + 1
            elif cat == "contract_repair":
                key = ev["message"].split("constraint", 1)[-1].split(";")[0].strip()
                repaired[key] = repaired.get(key, 0) + 1
    if not any_llm:
        return None
    ok_calls = totals["calls"] - totals["exceptions"]
    for d in list(by_tool.values()) + [totals]:
        okc = d["calls"] - d["exceptions"]
        d["mean_latency_s"] = (d["total_latency_s"] / okc) if okc else None
        d["exception_rate"] = d["exceptions"] / d["calls"] if d["calls"] else None
        d["empty_rate"] = d["empty_output"] / okc if okc else None
        d["schema_invalid_rate"] = d["schema_invalid"] / okc if okc else None
    totals["calls_per_run"] = describe(per_run_calls)
    totals["fallback_rate_per_call"] = (sum(fallbacks.values()) / totals["calls"]) if totals["calls"] else None
    return {"total": totals, "by_tool": dict(sorted(by_tool.items())),
            "fallbacks_by_component": dict(sorted(fallbacks.items())),
            "fallbacks_total": sum(fallbacks.values()), "ok_calls": ok_calls,
            "warnings_by_category": dict(sorted(categories.items())),
            "contract_constraints_reinserted": dict(sorted(repaired.items()))}


# ── outputs ───────────────────────────────────────────────────────────────

def _label(shift: int, phase: int, canonical: str | None, multi: bool) -> str:
    base = canonical or f"P{phase}"
    return f"{base} (shift {shift})" if multi else base


def write_csv(summary: dict, path: Path) -> None:
    rows = []
    for ph in summary["phases"]:
        for kpi, st in ph["kpis"].items():
            if st.get("n", 0) == 0:
                continue
            rows.append({
                "shift": ph["shift"], "phase": ph["phase"], "label": ph["label"], "kpi": kpi,
                "n": st["n"], "mean": st["mean"], "sd": st["sd"], "ci95_half": st["ci95_half"],
                "min": st["min"], "max": st["max"], "median": st["median"], "cv_pct": st["cv_pct"],
            })
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["shift"])
        w.writeheader()
        w.writerows(rows)


def _fmt(st: dict | None, digits: int = 1, pct: bool = False) -> str:
    if not st or st.get("n", 0) == 0:
        return "--"
    m, s = st["mean"], st["sd"]
    if pct:
        return f"{100 * m:.{digits}f} $\\pm$ {100 * s:.{digits}f}"
    return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def write_tex(summary: dict, path: Path) -> None:
    case = summary["case"]
    has_vr = any(ph["kpis"].get("vr_score", {}).get("n", 0) for ph in summary["phases"])
    cols = ["Phase", "Throughput (u/h)"]
    if has_vr:
        cols.append("V/R")
    cols += ["Fatigue", "Noise (dB)", "Energy (kWh)", "Feasible (\\%)", "$P_{\\text{feas}}$ (\\%)",
             "LLM-sel.\\ (\\%)", "Same strategy (\\%)"]
    n_runs = summary["n_runs"]
    model = (summary.get("llm") or {}).get("model") or "none"
    lines = [
        "% Auto-generated by scripts/aggregate_repeated.py — do not edit by hand",
        f"% case={case}  runs={n_runs}  model={model}",
        "\\begin{table}[!t]",
        "\\centering",
        "\\footnotesize",
        f"\\caption{{Variability of the selected schedule across $n={n_runs}$ independent "
        f"{'factory' if case == 'factory' else 'single-cell'} lifecycle runs (mean $\\pm$ SD). "
        "``Same strategy'' is the share of runs whose selected parameter vector lies within "
        "tolerance of the medoid run.}",
        f"\\label{{tab:repeated-{case}}}",
        "\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}",
        "\\toprule",
        " & ".join(cols) + " \\\\",
        "\\midrule",
    ]
    for ph in summary["phases"]:
        k = ph["kpis"]
        fw, stb = ph["firewall"], ph["stability"]
        feas = fw.get("selected_feasible_share")
        pf = fw.get("p_feasible", {})
        row = [ph["label"], _fmt(k.get("throughput_uph"), 1)]
        if has_vr:
            row.append(_fmt(k.get("vr_score"), 3))
        row += [
            _fmt(k.get("fatigue_index"), 3),
            _fmt(k.get("noise_db"), 1),
            _fmt(k.get("energy_kwh"), 0),
            "--" if feas is None else f"{100 * feas:.0f}",
            _fmt(pf, 0, pct=True) if pf.get("n") else "--",
            f"{100 * stb['llm_selected_share']:.0f}",
            f"{100 * stb['same_params_share']:.0f}",
        ]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines))


def make_figures(summary: dict, groups: dict, root: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D
    except Exception:  # pragma: no cover
        return []
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5, "legend.fontsize": 9.5, "figure.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })
    C_DET, C_LLM, C_BOX, C_REF = "#2196F3", "#EF6C00", "#555555", "#111111"
    case = summary["case"]
    keys = list(groups.keys())
    multi = len({k[0] for k in keys}) > 1
    if multi:
        labels = [groups[k][0][1].get("canonical") or f"P{k[1]}" for k in keys]
    else:
        labels = [_label(sh, ph, groups[(sh, ph)][0][1].get("canonical"), multi) for sh, ph in keys]
    kpis = [k for k in ("throughput_uph", "vr_score", "fatigue_index", "noise_db")
            if any(rec["kpis"].get(k) is not None for recs in groups.values() for _, rec in recs)]
    ref = summary.get("reference") or {}
    out_dir = root / "figures"
    out_dir.mkdir(exist_ok=True)
    paths: list[Path] = []

    # ── Figure A: box plots + points per phase ──────────────────────────
    fig, axes = plt.subplots(1, len(kpis), figsize=(3.4 * len(kpis), 3.6), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, kpi in zip(axes[0], kpis):
        data, xs = [], []
        for i, key in enumerate(keys):
            vals = [rec["kpis"].get(kpi) for _, rec in groups[key]]
            data.append([v for v in vals if v is not None])
            xs.append(i + 1)
        ax.boxplot(data, positions=xs, widths=0.5, showfliers=False,
                   medianprops={"color": C_BOX, "linewidth": 1.4},
                   boxprops={"color": C_BOX, "linewidth": 1.0},
                   whiskerprops={"color": C_BOX, "linewidth": 1.0},
                   capprops={"color": C_BOX, "linewidth": 1.0})
        for i, key in enumerate(keys):
            for _, rec in groups[key]:
                v = rec["kpis"].get(kpi)
                if v is None:
                    continue
                src = selection_source(rec)
                ax.plot(i + 1 + rng.uniform(-0.16, 0.16), v, marker="o", markersize=4.5,
                        color=C_LLM if src == "llm" else C_DET, markeredgecolor="white",
                        markeredgewidth=0.6, linestyle="none", alpha=0.9, zorder=3)
            canon = groups[key][0][1].get("canonical")
            if canon in ref and kpi in ref[canon]:
                ax.plot(i + 1, ref[canon][kpi], marker="_", markersize=16, markeredgewidth=2.0,
                        color=C_REF, linestyle="none", zorder=4)
        lim = LIMITS.get(case, {}).get(kpi)
        if lim is not None:
            ax.axhline(lim, color="#9e9e9e", linestyle="--", linewidth=1.0, zorder=1)
            ax.annotate("limit", xy=(len(keys) + 0.45, lim), fontsize=8.5, color="#616161",
                        ha="right", va="bottom")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel(KPI_LABEL[kpi])
        ax.set_xlim(0.4, len(keys) + 0.6)
        if multi:
            shifts_seq = [k[0] for k in keys]
            start = 0
            for i in range(1, len(keys) + 1):
                if i == len(keys) or shifts_seq[i] != shifts_seq[start]:
                    ax.annotate(f"shift {shifts_seq[start]}",
                                xy=((start + i + 1) / 2, -0.16), xycoords=("data", "axes fraction"),
                                ha="center", fontsize=8.5, color="#616161")
                    if i < len(keys):
                        ax.axvline(i + 0.5, color="#d0d0d0", linewidth=0.7, zorder=0)
                    start = i
    handles = [
        Line2D([], [], marker="o", color=C_DET, linestyle="none", markersize=5,
               label="selected: deterministic anchor"),
        Line2D([], [], marker="o", color=C_LLM, linestyle="none", markersize=5,
               label="selected: LLM candidate"),
    ]
    if ref:
        handles.append(Line2D([], [], marker="_", color=C_REF, linestyle="none", markersize=12,
                              markeredgewidth=2, label="single run reported in the paper"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{'Factory' if case == 'factory' else 'Single-cell'} case: selected-schedule KPIs "
                 f"over $n={summary['n_runs']}$ runs", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    for ext in ("pdf", "png"):
        p = out_dir / f"variability_boxplots.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)

    # ── Figure B: strategy stability per phase ──────────────────────────
    fig, ax = plt.subplots(figsize=(max(4.5, 1.3 * len(keys) + 1.5), 3.2))
    same = [100 * ph["stability"]["same_params_share"] for ph in summary["phases"]]
    llm = [100 * ph["stability"]["llm_selected_share"] for ph in summary["phases"]]
    x = np.arange(len(keys))
    w = 0.36
    b1 = ax.bar(x - w / 2, same, w, color=C_DET, label="same strategy (parameter vector within tolerance)")
    b2 = ax.bar(x + w / 2, llm, w, color=C_LLM, label="LLM candidate selected")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8.5,
                        color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of runs (%)")
    ax.set_ylim(0, 112)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"strategy_stability.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# ── main aggregation ─────────────────────────────────────────────────────

def load_reference(path: Path | None, case: str) -> dict[str, dict[str, float]]:
    """Map the paper's single-run table (factory_table.json / table3.json) onto KPI names."""
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text())
    ref: dict[str, dict[str, float]] = {}
    for name, row in raw.items():
        if case == "factory":
            fat = [row.get(k) for k in ("fatigue_h1", "fatigue_h2", "fatigue_h3") if row.get(k) is not None]
            ref[name] = {
                "throughput_uph": row.get("total_uph"), "noise_db": row.get("noise_db"),
                "fatigue_index": max(fat) if fat else None, "energy_kwh": row.get("energy_kwh"),
                "defect_rate": row.get("defect_rate"),
            }
        else:
            ref[name] = {k: row.get(k) for k in ("throughput_uph", "vr_score", "fatigue_index",
                                                 "noise_db", "energy_kwh", "defect_rate")}
        ref[name] = {k: v for k, v in ref[name].items() if v is not None}
    return ref


def _default_reference(root: Path, case: str) -> Path | None:
    here = Path(__file__).resolve().parent.parent
    cand = {
        "factory": here / "data/results_factory_integrated/factory_table.json",
        "single": here / "data/results_full_integrated_3shifts/table3.json",
    }.get(case)
    return cand if cand and cand.exists() else None


def aggregate(root: Path, reference: Path | None = None) -> dict[str, Any]:
    root = Path(root)
    runs, bad = load_runs(root)
    batch = {}
    if (root / "batch_meta.json").exists():
        batch = json.loads((root / "batch_meta.json").read_text())
    case = batch.get("case") or (runs[0]["case"] if runs else "unknown")
    if reference is None:
        reference = _default_reference(root, case)

    groups = phase_groups(runs)
    multi = len({k[0] for k in groups}) > 1
    phases = []
    for (shift, phase), items in groups.items():
        recs = [rec for _, rec in items]
        kpis = {}
        for kpi in KPI_ORDER:
            vals = [rec["kpis"].get(kpi) for rec in recs]
            if any(v is not None for v in vals):
                kpis[kpi] = describe(vals)
        phases.append({
            "shift": shift, "phase": phase, "canonical": recs[0].get("canonical"),
            "label": _label(shift, phase, recs[0].get("canonical"), multi),
            "runs": [ri for ri, _ in items],
            "kpis": kpis,
            "stability": stability(recs),
            "firewall": firewall_stats(recs),
            "selected": [{"run": ri, "name": rec["schedule"]["name"],
                          "source": rec["schedule"].get("source"),
                          "throughput_uph": rec["kpis"].get("throughput_uph")} for ri, rec in items],
        })

    # multi-shift learning summaries (single-cell)
    shift_summ: dict[tuple[int, str], list[float]] = {}
    for meta in runs:
        for sh in meta.get("shifts", []):
            if "summary" in sh:
                for k, v in sh["summary"].items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        shift_summ.setdefault((int(sh["shift"]), k), []).append(v)
    learning: dict[str, dict] = {}
    for (sh, k), vals in sorted(shift_summ.items()):
        learning.setdefault(str(sh), {})[k] = describe(vals)

    summary = {
        "case": case,
        "output_root": str(root),
        "n_runs": len(runs),
        "n_failed_runs": len(bad),
        "failed_runs": [{"run_index": b.get("run_index"), "error": b.get("error")} for b in bad],
        "llm": batch.get("llm") or (runs[0].get("llm") if runs else None),
        "git_commit": batch.get("git_commit"),
        "prompt_hash": batch.get("prompt_hash"),
        "mc_seed_bases": sorted({int(m.get("mc_seed_base", 42)) for m in runs}),
        "execution": sorted({m.get("execution", "?") for m in runs}),
        "duration_s": describe([m.get("duration_s") for m in runs]),
        "phases": phases,
        "learning_curve": learning or None,
        "llm_calls": llm_call_stats(runs),
        "reference": load_reference(reference, case),
        "reference_path": str(reference) if reference else None,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_csv(summary, root / "summary_table.csv")
    write_tex(summary, root / "summary_table.tex")
    figs = make_figures(summary, groups, root) if groups else []
    print_report(summary)
    print(f"  wrote summary.json, summary_table.csv, summary_table.tex" +
          (f", {len(figs)} figure files" if figs else "") + f"  →  {root}")
    return summary


def print_report(summary: dict) -> None:
    print("\n" + "=" * 78)
    print(f"  REPEATED-RUN SUMMARY — {summary['case']} case, n={summary['n_runs']} runs"
          f" ({summary['n_failed_runs']} failed)")
    llm = summary.get("llm") or {}
    if llm:
        print(f"  model={llm.get('model')}  temperature={llm.get('temperature')}  "
              f"MC seed bases={summary['mc_seed_bases']}")
    print("=" * 78)
    print(f"{'Phase':<16}{'Throughput':>16}{'Fatigue':>16}{'Noise':>14}{'Feas%':>7}{'LLM%':>7}{'Same%':>7}")

    def ms(st, d):
        return f"{st['mean']:.{d}f}±{st['sd']:.{d}f}" if st and st.get("n") else "--"

    for ph in summary["phases"]:
        k, stb, fw = ph["kpis"], ph["stability"], ph["firewall"]
        feas = fw.get("selected_feasible_share")
        print(f"{ph['label']:<16}{ms(k.get('throughput_uph'), 1):>16}{ms(k.get('fatigue_index'), 3):>16}"
              f"{ms(k.get('noise_db'), 1):>14}{('--' if feas is None else f'{100*feas:.0f}'):>7}"
              f"{100 * stb['llm_selected_share']:>7.0f}{100 * stb['same_params_share']:>7.0f}")
    lc = summary.get("llm_calls")
    if lc:
        t = lc["total"]
        print(f"\n  LLM calls: {t['calls']} total ({t['calls_per_run']['mean']:.1f}/run), "
              f"exceptions={t['exceptions']}, empty={t['empty_output']}, "
              f"schema-invalid={t['schema_invalid']}, mean latency={t['mean_latency_s'] or 0:.1f}s, "
              f"fallbacks={lc['fallbacks_total']} {lc['fallbacks_by_component']}; "
              f"contract fields re-inserted/canonicalised by validator: {lc['contract_constraints_reinserted']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate repeated CBPA runs")
    p.add_argument("--root", required=True, help="batch root containing run_XX/ folders")
    p.add_argument("--reference", default=None,
                   help="paper single-run table (factory_table.json / table3.json) to overlay")
    args = p.parse_args(argv)
    aggregate(Path(args.root), Path(args.reference) if args.reference else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
