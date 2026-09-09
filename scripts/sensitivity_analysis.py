#!/usr/bin/env python3
"""Sensitivity of the verification firewall to twin mismatch and sensor noise.

Runs without any LLM call.  The analytical models with
their default coefficients are the *nominal twin* used for certification;
a copy with perturbed coefficients plays the *true plant*.  For every
schedule that appears in the two case studies (single-cell S1--S4 from the
medoid run, factory FS1--FS4) the script reports:

* the nominal verdict (deterministic feasibility, Monte-Carlo P(feasible),
  certificate) and the true-plant verdict under each perturbation;
* false-accept (deployed but infeasible on the true plant) and
  false-reject counts per perturbation family and magnitude;
* the mismatch tolerance of each deployed schedule: the smallest adverse
  coefficient error at which it violates a hard constraint (bisection);
* under sensor noise, the false-alarm rate of the runtime guards for
  deployed schedules and the missed-detection rate for rejected ones.

Outputs go to data/sensitivity/ (summary.json, sensitivity_table.tex,
figures/sensitivity.{pdf,png}).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from cbpa.config.defaults import (  # noqa: E402
    FATIGUE_ALPHA, FATIGUE_BETA, NOISE_BASE_DB, NOISE_K1, NOISE_K2,
)
from cbpa.config.scenario import CellConfig, FactoryConfig  # noqa: E402
from cbpa.layer3_verification.constraint_checker import ConstraintChecker  # noqa: E402
from cbpa.models.contract import make_c1, make_c1_factory  # noqa: E402
from cbpa.models.schedule import CellSchedule, FactorySchedule, Schedule  # noqa: E402
from cbpa.physics.cell_evaluator import CellEvaluator  # noqa: E402
from cbpa.physics.factory_evaluator import FactoryEvaluator  # noqa: E402
from cbpa.physics.fatigue_model import FatigueModel  # noqa: E402
from cbpa.physics.noise_model import NoiseModel  # noqa: E402

DELTAS = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
FAMILIES = ["fatigue_alpha", "fatigue_beta", "noise_coeffs", "cycle_times"]
FAMILY_LABEL = {
    "fatigue_alpha": "fatigue coefficient $\\alpha$",
    "fatigue_beta": "fatigue exponent $\\beta$",
    "noise_coeffs": "noise coefficients $k_1,k_2$",
    "cycle_times": "cycle times",
}
CERT_THRESHOLD = 0.95
MC_SEED = 42
MC_N = 200

# Schedules as deployed / rejected in the reported runs
SINGLE = {
    "S1": dict(r1=0.70, r2=0.65, hr=1.00, buf=2.7, demand=52.0),
    "S2": dict(r1=0.90, r2=1.00, hr=1.15, buf=0.85, demand=62.4),
    "S3": dict(r1=0.65, r2=0.68, hr=0.90, buf=8.0, demand=62.4),      # deployed (LLM, certified)
    "S3d": dict(r1=0.72, r2=0.70, hr=1.05, buf=1.7, demand=62.4),     # anchor passed over by the gate (uncertified)
    "S4": dict(r1=0.425, r2=0.65, hr=1.05, buf=2.5, demand=52.0),
}
FACTORY = {
    "FS1": dict(A=(0.69, 0.64, 1.02, 2.76), B=(0.64, 0.59, 1.02, 2.76), ops={"H1": "A", "H2": "B"},
                routing={"V_A": ["A", "B"], "V_B": ["A", "B"], "V_C": ["A", "B"]},
                va=["V_A", "V_B", "V_C"], vb=["V_A", "V_B", "V_C"]),
    "FS2": dict(A=(0.95, 0.93, 1.38, 5.0), B=(0.90, 0.88, 1.25, 8.0), ops={"H1": "A", "H3": "B"},
                routing={"V_A": ["A", "B"], "V_B": ["A"], "V_C": ["A"]}, va=["V_A", "V_B", "V_C"], vb=["V_A"]),
    "FS3": dict(A=(0.66, 0.62, 1.02, 3.22), B=(0.54, 0.49, 0.87, 3.22), ops={"H1": "A", "H3": "B"},
                routing={"V_A": ["A", "B"], "V_B": ["A"], "V_C": ["A"]}, va=["V_A", "V_B", "V_C"], vb=["V_A"]),
    "FS4": dict(A=(0.72, 0.69, 1.04, 2.76), B=(0.64, 0.59, 0.92, 3.22), ops={"H1": "A", "H3": "B"},
                routing={"V_A": ["A", "B"], "V_B": ["A"], "V_C": ["A"]}, va=["V_A", "V_B", "V_C"], vb=["V_A"]),
}


# ── model construction ────────────────────────────────────────────────

def _fatigue(family: str, d: float) -> FatigueModel:
    a = FATIGUE_ALPHA * (1 + d) if family == "fatigue_alpha" else FATIGUE_ALPHA
    b = FATIGUE_BETA * (1 + d) if family == "fatigue_beta" else FATIGUE_BETA
    return FatigueModel(alpha=a, beta=b)


def _noise(family: str, d: float) -> NoiseModel:
    if family == "noise_coeffs":
        return NoiseModel(base_db=NOISE_BASE_DB, k1=NOISE_K1 * (1 + d), k2=NOISE_K2 * (1 + d))
    return NoiseModel()


def _cell_cfg(base: CellConfig, family: str, d: float) -> CellConfig:
    if family != "cycle_times":
        return base
    return CellConfig(
        **{**base.__dict__,
           "r1_cycle_time_s": base.r1_cycle_time_s * (1 + d),
           "r2_cycle_time_s": base.r2_cycle_time_s * (1 + d),
           "human_insertion_time_s": base.human_insertion_time_s * (1 + d),
           "human_inspection_time_s": base.human_inspection_time_s * (1 + d)}
    )


def single_evaluator(family: str = "fatigue_alpha", d: float = 0.0) -> CellEvaluator:
    ev = CellEvaluator(cell=_cell_cfg(CellConfig(), family, d), mode="analytical")
    ev.fatigue_model = _fatigue(family, d)
    ev.noise_model = _noise(family, d)
    return ev


def factory_evaluator(family: str = "fatigue_alpha", d: float = 0.0) -> FactoryEvaluator:
    cfg = FactoryConfig()
    cfg.cells = {cid: _cell_cfg(c, family, d) for cid, c in cfg.cells.items()}
    ev = FactoryEvaluator(factory=cfg, mode="analytical")
    for cev in ev.cell_evaluators.values():
        cev.fatigue_model = _fatigue(family, d)
        cev.noise_model = _noise(family, d)
    ev.fatigue_model = _fatigue(family, d)
    return ev


def single_schedule(name: str) -> Schedule:
    p = SINGLE[name]
    return Schedule(name=name, r1_speed_fraction=p["r1"], r2_speed_fraction=p["r2"],
                    human_cycle_rate_multiplier=p["hr"], buffer_time_s=p["buf"],
                    demand_target_uph=p["demand"])


def factory_schedule(name: str) -> FactorySchedule:
    p = FACTORY[name]
    cells = {}
    for cid, key, demand in (("A", "A", 52.0), ("B", "B", 48.0)):
        r1, r2, hr, buf = p[key]
        cells[cid] = CellSchedule(
            name=f"{name}_{cid}", cell_id=cid, r1_speed_fraction=r1, r2_speed_fraction=r2,
            human_cycle_rate_multiplier=hr, buffer_time_s=buf, demand_target_uph=demand,
            assigned_operator=next((o for o, c in p["ops"].items() if c == cid), "H1"),
            assigned_variants=p["va"] if cid == "A" else p["vb"],
        )
    return FactorySchedule(name=name, cell_schedules=cells, variant_routing=p["routing"],
                           operator_assignments=p["ops"], agv_priority="balanced")


# ── verdicts ─────────────────────────────────────────────────────────

def verdict_single(name: str, family: str, d: float, checker: ConstraintChecker, contract) -> dict:
    ev = single_evaluator(family, d)
    sched = single_schedule(name)
    m = ev.evaluate(sched)
    rep = checker.check(contract, m)
    return {"feasible": rep.is_feasible, "fatigue": m.fatigue_index, "noise": m.noise_db,
            "throughput": m.throughput_uph, "margins": rep.constraint_margins,
            "violations": rep.violations}


def verdict_factory(name: str, family: str, d: float, checker: ConstraintChecker, contract) -> dict:
    ev = factory_evaluator(family, d)
    sched = factory_schedule(name)
    m = ev.evaluate(sched)
    rep = checker.check_factory(contract, m)
    fat = dict(m.operator_fatigue)
    return {"feasible": rep.is_feasible, "fatigue": max(fat.values()) if fat else 0.0,
            "operator_fatigue": fat, "noise": m.factory_noise_db,
            "throughput": m.total_throughput_uph, "margins": rep.constraint_margins,
            "violations": rep.violations}


def nominal_certificate(case: str, name: str, checker: ConstraintChecker, contract) -> dict:
    if case == "single":
        ev = single_evaluator(); sched = single_schedule(name)
        det = checker.check(contract, ev.evaluate(sched))
        mc = ev.evaluate_monte_carlo(sched, n_samples=MC_N, seed=MC_SEED)
        st = checker.check_stochastic(contract, mc, seed=MC_SEED)
    else:
        ev = factory_evaluator(); sched = factory_schedule(name)
        det = checker.check_factory(contract, ev.evaluate(sched))
        mc = ev.evaluate_monte_carlo(sched, n_samples=MC_N, seed=MC_SEED)
        st = checker.check_factory_stochastic(contract, mc, seed=MC_SEED)
    return {"det_feasible": det.is_feasible, "p_feasible": st.p_feasible,
            "certified": st.p_feasible >= CERT_THRESHOLD, "deployed": det.is_feasible,
            "margins": det.constraint_margins}


def tolerance(case: str, name: str, family: str, checker, contract, adverse_sign: float) -> float | None:
    """Smallest |d| (adverse direction) at which the schedule becomes infeasible; None if > 100 %."""
    fn = verdict_single if case == "single" else verdict_factory
    if not fn(name, family, 0.0, checker, contract)["feasible"]:
        return 0.0
    lo, hi = 0.0, 1.0
    if fn(name, family, adverse_sign * hi, checker, contract)["feasible"]:
        return None
    for _ in range(30):
        mid = (lo + hi) / 2
        if fn(name, family, adverse_sign * mid, checker, contract)["feasible"]:
            lo = mid
        else:
            hi = mid
    return hi


# ── sensor noise ────────────────────────────────────────────────────

FATIGUE_SIGMAS = [0.01, 0.02, 0.05]
NOISE_SIGMAS = [0.5, 1.0, 2.0]


def sensor_rates(true_fatigue: float, true_noise: float, fat_limit: float, noise_limit: float,
                 n: int = 20000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    out = {"fatigue": {}, "noise": {}}
    for s in FATIGUE_SIGMAS:
        meas = true_fatigue + rng.normal(0, s, n)
        flagged = (meas > fat_limit).mean()
        warned = (meas > 0.9 * fat_limit).mean()
        out["fatigue"][str(s)] = {"guard_rate": float(flagged), "warn_rate": float(warned)}
    for s in NOISE_SIGMAS:
        meas = true_noise + rng.normal(0, s, n)
        out["noise"][str(s)] = {"guard_rate": float((meas > noise_limit).mean()),
                                "warn_rate": float((meas > noise_limit - 0.1 * noise_limit).mean())}
    return out


# ── main ────────────────────────────────────────────────────────────

def run(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    checker = ConstraintChecker()
    cases = {"single": (SINGLE, make_c1(), verdict_single),
             "factory": (FACTORY, make_c1_factory(), verdict_factory)}
    summary: dict = {"deltas": DELTAS, "families": FAMILIES, "cases": {}}
    for case, (scheds, contract, fn) in cases.items():
        cs: dict = {"nominal": {}, "grid": {}, "tolerance": {}, "sensor": {}, "false_accept": {}, "false_reject": {}}
        for name in scheds:
            cs["nominal"][name] = nominal_certificate(case, name, checker, contract)
        for fam in FAMILIES:
            cs["grid"][fam] = {}
            for d in DELTAS:
                cs["grid"][fam][str(d)] = {name: fn(name, fam, d, checker, contract) for name in scheds}
            fa = {}; fr = {}
            for d in DELTAS:
                fa[str(d)] = [n for n in scheds if cs["nominal"][n]["deployed"] and not cs["grid"][fam][str(d)][n]["feasible"]]
                fr[str(d)] = [n for n in scheds if not cs["nominal"][n]["deployed"] and cs["grid"][fam][str(d)][n]["feasible"]]
            cs["false_accept"][fam] = fa; cs["false_reject"][fam] = fr
            # adverse direction: coefficients up (more fatigue / noise); cycle times do not touch K
            if fam != "cycle_times":
                # adverse direction: coefficients up for alpha and noise; the fatigue
                # exponent acts on an effective load < 1, so a *smaller* beta raises fatigue
                sign = -1.0 if fam == "fatigue_beta" else +1.0
                cs["tolerance"][fam] = {n: tolerance(case, n, fam, checker, contract, sign) for n in scheds}
        # throughput misestimation from cycle-time mismatch
        cs["throughput_error_pct"] = {}
        for name in scheds:
            nom = cs["grid"]["cycle_times"]["0.0"][name]["throughput"]
            cs["throughput_error_pct"][name] = {str(d): round(100 * (cs["grid"]["cycle_times"][str(d)][name]["throughput"] - nom) / nom, 1) for d in DELTAS}
        fat_limit = 0.4; noise_limit = 80.0 if case == "single" else 82.0
        for name in scheds:
            v = cs["grid"]["fatigue_alpha"]["0.0"][name]
            if case == "factory":
                # the operator closest to its own limit
                fat_by_op = v["operator_fatigue"]; lim = {"H1": 0.4, "H2": 0.35, "H3": 0.4}
                op = max(fat_by_op, key=lambda o: fat_by_op[o] - lim[o])
                true_f, fat_limit_o = fat_by_op[op], lim[op]
            else:
                true_f, fat_limit_o = v["fatigue"], fat_limit
            cs["sensor"][name] = {"true_fatigue": true_f, "fatigue_limit": fat_limit_o, "true_noise": v["noise"],
                                  "noise_limit": noise_limit,
                                  **sensor_rates(true_f, v["noise"], fat_limit_o, noise_limit)}
        summary["cases"][case] = cs
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_tex(summary, out_dir / "sensitivity_table.tex")
    make_figure(summary, out_dir / "figures")
    print_report(summary)
    return summary


def _tol(x):
    return "$>$100" if x is None else ("0" if x == 0 else f"{100 * x:.1f}")


def write_tex(summary: dict, path: Path) -> None:
    lines = [
        "% Auto-generated by scripts/sensitivity_analysis.py",
        "\\begin{table}[!t]", "\\centering", "\\footnotesize",
        "\\caption{Sensitivity of the firewall's verdicts to twin mismatch and sensor noise. Tolerance: smallest adverse coefficient error (\\%) at which a deployed schedule violates a hard constraint on the true plant (0 for schedules that are rejected nominally). Guard false-alarm / missed-detection rates at the stated measurement noise.}",
        "\\label{tab:sensitivity}", "\\renewcommand{\\arraystretch}{1.2}",
        "\\begin{tabular}{@{}l l c c c c c c@{}}", "\\toprule",
        "Case & Schedule & $P(\\text{feas.})$ & Tol.\\ $\\alpha$ (\\%) & Tol.\\ $\\beta$ (\\%) & Tol.\\ noise (\\%) & Guard FA, $\\sigma_F{=}0.02$ & Guard FA, $\\sigma_N{=}1$\\,dB \\\\",
        "\\midrule",
    ]
    for case, cs in summary["cases"].items():
        for name, nom in cs["nominal"].items():
            sen = cs["sensor"][name]
            fa_f = sen["fatigue"]["0.02"]["guard_rate"]; fa_n = sen["noise"]["1.0"]["guard_rate"]
            if nom["deployed"]:
                fa_txt_f = f"{100 * fa_f:.0f}\\%"; fa_txt_n = f"{100 * fa_n:.0f}\\%"
            else:
                fa_txt_f = f"MD {100 * (1 - fa_f):.0f}\\%"; fa_txt_n = f"MD {100 * (1 - fa_n):.0f}\\%"
            label = f"${name[0]}_{{{name[1:]}}}$"
            lines.append(f"{case} & {label} & {100 * nom['p_feasible']:.0f}\\% & "
                         f"{_tol(cs['tolerance']['fatigue_alpha'][name])} & {_tol(cs['tolerance']['fatigue_beta'][name])} & "
                         f"{_tol(cs['tolerance']['noise_coeffs'][name])} & {fa_txt_f} & {fa_txt_n} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines))


def make_figure(summary: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 10.5, "axes.labelsize": 10.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "figure.dpi": 150, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e5", "axes.axisbelow": True})
    C_OK, C_REJ = "#2196F3", "#EF6C00"
    styles = ["-", "--", "-.", ":", (0, (1, 1))]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    panels = [("single", "fatigue_alpha", "fatigue", 0.4, "Fatigue index"),
              ("single", "noise_coeffs", "noise", 80.0, "Noise (dB)"),
              ("factory", "fatigue_alpha", "fatigue", 0.4, "Max operator fatigue"),
              ("factory", "noise_coeffs", "noise", 82.0, "Factory noise (dB)")]
    xs = [100 * d for d in DELTAS]
    for ax, (case, fam, key, limit, ylabel) in zip(axes.flat, panels):
        cs = summary["cases"][case]
        for i, name in enumerate(cs["nominal"]):
            ys = [cs["grid"][fam][str(d)][name][key] for d in DELTAS]
            rejected = not cs["nominal"][name]["deployed"]
            ax.plot(xs, ys, linestyle=styles[i % 4], color=C_REJ if rejected else C_OK, linewidth=1.6,
                    marker="o", markersize=3.2, label=f"${name[0]}_{{{name[1:]}}}$" + (" (rejected)" if rejected else ""))
            ax.annotate(f"${name[0]}_{{{name[1:]}}}$", xy=(xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8.5, va="center", color="#333333")
        ax.axhline(limit, color="#9e9e9e", linestyle="--", linewidth=1.0)
        ax.annotate("limit", xy=(xs[0], limit), xytext=(0, 3), textcoords="offset points", fontsize=8, color="#616161")
        ax.set_xlabel(f"Error in {FAMILY_LABEL[fam]} (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{'Single-cell' if case == 'single' else 'Factory'} case", fontsize=10.5)
        ax.set_xlim(xs[0] - 3, xs[-1] + 12)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C_OK, linewidth=1.6, label="deployed schedule"),
               Line2D([], [], color=C_REJ, linewidth=1.6, label="rejected schedule"),
               Line2D([], [], color="#9e9e9e", linestyle="--", label="contract limit")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"sensitivity.{ext}", bbox_inches="tight")
    plt.close(fig)


def print_report(summary: dict) -> None:
    for case, cs in summary["cases"].items():
        print(f"\n== {case} ==")
        for name, nom in cs["nominal"].items():
            t = cs["tolerance"]
            print(f"  {name}: det={nom['det_feasible']} P={nom['p_feasible']:.3f} cert={nom['certified']} | "
                  f"tol alpha={_tol(t['fatigue_alpha'][name])} beta={_tol(t['fatigue_beta'][name])} noise={_tol(t['noise_coeffs'][name])} | "
                  f"thr err @±30% ct: {cs['throughput_error_pct'][name]['-0.3']}/{cs['throughput_error_pct'][name]['0.3']}%")
        for fam in FAMILIES:
            fa = {d: v for d, v in cs["false_accept"][fam].items() if v}
            fr = {d: v for d, v in cs["false_reject"][fam].items() if v}
            print(f"  {fam}: false-accept {fa or '-'} | false-reject {fr or '-'}")
        for name, s in cs["sensor"].items():
            print(f"  sensor {name}: true fat {s['true_fatigue']:.3f}/{s['fatigue_limit']} guard@σ0.01/0.02/0.05 = "
                  + "/".join(f"{100*s['fatigue'][k]['guard_rate']:.0f}%" for k in ('0.01', '0.02', '0.05'))
                  + f" | noise {s['true_noise']}/{s['noise_limit']} guard@σ0.5/1/2 = "
                  + "/".join(f"{100*s['noise'][k]['guard_rate']:.0f}%" for k in ('0.5', '1.0', '2.0')))


def load_schedules(single_root=None, factory_root=None, single_nollm=None) -> None:
    """Replace the hard-coded schedules by the medoid run's deployed/rejected schedules.

    ``S3d`` (the anchor the certification gate passed over) is taken from the
    no-LLM batch, where that anchor is what gets deployed.
    """
    from paper_values import factory_params, single_params
    if single_root:
        for name, p in single_params(single_root).items():
            SINGLE[name] = p
    if single_nollm:
        SINGLE["S3d"] = single_params(single_nollm)["S3"]
    if factory_root:
        for name, p in factory_params(factory_root).items():
            FACTORY[name] = p
    # keep the display order S1, S2, S3, S3d, S4
    order = [n for n in ("S1", "S2", "S3", "S3d", "S4") if n in SINGLE]
    for n in order:
        SINGLE[n] = SINGLE.pop(n)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Firewall sensitivity to twin mismatch and sensor noise")
    ap.add_argument("out", nargs="?", default=str(_HERE.parent / "data" / "sensitivity"))
    ap.add_argument("--single-root"); ap.add_argument("--factory-root"); ap.add_argument("--single-nollm-root")
    a = ap.parse_args()
    load_schedules(a.single_root, a.factory_root, a.single_nollm_root)
    run(Path(a.out))
