#!/usr/bin/env python3
"""Repeated-run harness for the CBPA case studies.

Runs the single-cell or factory lifecycle *N* independent times, each with a
fresh LLM client, and records per run:

* the selected schedule (name, provenance, parameter vector) and its KPIs for
  every planning phase, the firewall verdicts (deterministic feasibility and
  Monte-Carlo ``p_feasible``), and the full candidate pool with provenance;
* every LLM call (latency, exceptions, empty / schema-invalid outputs) and the
  deterministic fallbacks they triggered;
* the exact model, sampling settings, Monte-Carlo seed base, prompt-file hash
  and git commit, so that the run is reproducible and reportable.

Per-run artefacts go to ``<output-root>/run_XX/`` (the usual exporter files
plus ``run_meta.json``); ``aggregate_repeated.py`` then produces mean / SD /
95 % CI tables, strategy-stability measures and figures.

Examples
--------
    # 10 LLM runs of the factory lifecycle, fixed Monte-Carlo seeds
    python scripts/run_repeated.py --case factory --runs 10

    # 10 runs of the 3-shift single-cell full demo, varying MC seeds too
    python scripts/run_repeated.py --case single --full-demo --runs 10 --vary-mc-seed

    # deterministic reference (no LLM) — should be identical across runs
    python scripts/run_repeated.py --case factory --runs 3 --no-llm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))

from cbpa.llm.client import create_client  # noqa: E402
from cbpa.llm.instrumented import FallbackLogCapture, InstrumentedLLMClient  # noqa: E402

logger = logging.getLogger("run_repeated")

# Phase -> canonical schedule label used in the paper tables.
CANONICAL = {
    "single": {1: "S1", 3: "S2", 5: "S3", 6: "S4"},
    "factory": {1: "FS1", 3: "FS2", 5: "FS3", 6: "FS4"},
}

DEFAULT_MC_SEED = 42


# ── provenance helpers ───────────────────────────────────────────────────

def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_HERE, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _prompt_hash() -> str:
    h = hashlib.sha256()
    for name in ("prompts.py", "schemas.py"):
        p = _HERE.parent / "src" / "cbpa" / "llm" / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


# ── record extraction ────────────────────────────────────────────────────

def _params_of(sched: Any) -> dict[str, Any] | None:
    """Flatten a Schedule / FactorySchedule into a JSON-friendly dict."""
    if sched is None:
        return None
    if hasattr(sched, "cell_schedules"):  # FactorySchedule
        return {
            "cells": {
                cid: {
                    "r1_speed_fraction": cs.r1_speed_fraction,
                    "r2_speed_fraction": cs.r2_speed_fraction,
                    "human_cycle_rate_multiplier": cs.human_cycle_rate_multiplier,
                    "buffer_time_s": cs.buffer_time_s,
                    "assigned_operator": cs.assigned_operator,
                    "assigned_variants": list(cs.assigned_variants),
                }
                for cid, cs in sorted(sched.cell_schedules.items())
            },
            "variant_routing": {k: list(v) for k, v in sched.variant_routing.items()},
            "operator_assignments": dict(sched.operator_assignments),
            "agv_priority": getattr(sched, "agv_priority", None),
        }
    return {
        "r1_speed_fraction": sched.r1_speed_fraction,
        "r2_speed_fraction": sched.r2_speed_fraction,
        "human_cycle_rate_multiplier": sched.human_cycle_rate_multiplier,
        "buffer_time_s": sched.buffer_time_s,
        "demand_target_uph": sched.demand_target_uph,
    }


def _pareto_from_audit(audit: Any, phase: int) -> dict[str, Any] | None:
    """Last audit entry of *phase* that carries a candidate pool."""
    found = None
    for e in audit.entries:
        if e.phase == phase and "candidate_sources" in e.details:
            found = e
    if found is None:
        return None
    d = found.details
    scores = d.get("scores", {}) or {}
    return {
        "candidates": list(scores.keys()) or list(d.get("candidate_sources", {}).keys()),
        "non_dominated": list(d.get("non_dominated") or d.get("pareto_front") or []),
        "selected": d.get("selected"),
        "selected_source": d.get("selected_source"),
        "candidate_sources": dict(d.get("candidate_sources", {})),
        "scores": {k: dict(v) for k, v in scores.items()},
        "gate_certified": d.get("gate_certified"),
        "gate_no_feasible_candidate": d.get("gate_no_feasible_candidate"),
        "gate_p_feasible": d.get("gate_p_feasible"),
        "gate_tried": list(d.get("gate_tried", []) or []),
    }


def _p_feasible_from_audit(audit: Any, phase: int) -> float | None:
    val = None
    for e in audit.entries:
        if e.phase == phase and "p_feasible" in e.details:
            val = e.details["p_feasible"]
    return None if val is None else float(val)


def _single_phase_record(pr: Any, audit: Any) -> dict[str, Any]:
    m, vr, feas, sched = pr.metrics, pr.vr_score, pr.feasibility, pr.schedule
    mode = pr.working_mode.value if hasattr(pr.working_mode, "value") else str(pr.working_mode)
    kpis = None
    if m is not None:
        kpis = {
            "throughput_uph": m.throughput_uph,
            "defect_rate": m.defect_rate,
            "noise_db": m.noise_db,
            "fatigue_index": m.fatigue_index,
            "energy_kwh": m.energy_kwh,
            "deadline_gap_pct": m.deadline_gap_pct,
        }
        if vr is not None:
            kpis["vr_score"] = vr.vr_score
    pareto = _pareto_from_audit(audit, pr.phase)
    if pareto is None and pr.pareto_front is not None:
        pf = pr.pareto_front
        pareto = {
            "candidates": list(pf.candidates),
            "non_dominated": list(pf.non_dominated),
            "selected": pf.selected,
            "selected_source": pf.candidate_sources.get(pf.selected),
            "candidate_sources": dict(pf.candidate_sources),
            "scores": {k: dict(v) for k, v in pf.scores.items()},
        }
    return {
        "phase": pr.phase,
        "canonical": CANONICAL["single"].get(pr.phase),
        "working_mode": mode,
        "description": pr.description,
        "schedule": None if sched is None else {
            "name": sched.name, "source": getattr(sched, "source", "deterministic"),
            "params": _params_of(sched),
        },
        "kpis": kpis,
        "feasible": None if feas is None else feas.is_feasible,
        "violations": [] if feas is None else list(feas.violations),
        "p_feasible": _p_feasible_from_audit(audit, pr.phase),
        "manager_decision": None if pr.decision is None else pr.decision.selected_option,
        "pareto": pareto,
    }


def _factory_phase_record(pr: Any, audit: Any) -> dict[str, Any]:
    fm, feas, sched = pr.factory_metrics, pr.feasibility, pr.factory_schedule
    kpis = None
    if fm is not None:
        fat = dict(fm.operator_fatigue or {})
        kpis = {
            "throughput_uph": fm.total_throughput_uph,
            "noise_db": fm.factory_noise_db,
            "fatigue_index": max(fat.values()) if fat else 0.0,
            "operator_fatigue": fat,
            "energy_kwh": fm.factory_energy_kwh,
            "defect_rate": fm.factory_defect_rate,
            "cell_balance_loss_pct": fm.cell_balance_loss_pct,
            "agv_utilization": fm.agv_utilization,
            "cell_uph": {cid: cm.throughput_uph for cid, cm in fm.cell_metrics.items()},
        }
    p_feas = pr.stochastic_report.p_feasible if pr.stochastic_report is not None else None
    if p_feas is None:
        p_feas = _p_feasible_from_audit(audit, pr.phase)
    return {
        "phase": pr.phase,
        "canonical": CANONICAL["factory"].get(pr.phase),
        "phase_name": pr.phase_name,
        "working_mode": pr.working_mode,
        "schedule": None if sched is None else {
            "name": sched.name, "source": getattr(sched, "source", "deterministic"),
            "params": _params_of(sched),
        },
        "kpis": kpis,
        "feasible": None if feas is None else feas.is_feasible,
        "violations": [] if feas is None else list(feas.violations),
        "p_feasible": p_feas,
        "manager_decision": pr.manager_decision,
        "notes": list(pr.notes),
        "pareto": _pareto_from_audit(audit, pr.phase),
    }


# ── one run ───────────────────────────────────────────────────────────────

def _build_client(args: argparse.Namespace) -> InstrumentedLLMClient | None:
    if args.no_llm:
        return None
    raw = create_client(provider=args.provider, model=args.model)
    return InstrumentedLLMClient(raw)


def run_one(args: argparse.Namespace, run_idx: int, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    use_llm = not args.no_llm
    mc_seed_base = DEFAULT_MC_SEED + (1000 * run_idx if args.vary_mc_seed else 0)
    client = _build_client(args)
    capture = FallbackLogCapture()  # attached once the experiment (and all layer modules) exists

    meta: dict[str, Any] = {
        "run_index": run_idx,
        "case": args.case,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "use_llm": use_llm,
        "llm": None if client is None else client.sampling_settings(),
        "mc_seed_base": mc_seed_base,
        "mc_samples": 200,
        "execution": "integrated" if args.integrated else ("simpy" if args.simulate else "analytical"),
    }
    t0 = time.perf_counter()
    exp = None
    try:
        if args.case == "factory":
            from cbpa.config.scenario import FactoryScenarioConfig
            from cbpa.runner.factory_experiment import FactoryExperiment
            from cbpa.runner.factory_results_exporter import FactoryResultsExporter

            config = FactoryScenarioConfig(
                llm_only=bool(getattr(args, "llm_only", False)),
                optimiser_anchor=bool(getattr(args, "optimiser_anchor", False)),
                enable_operator_absence=not args.no_operator_absence,
                enable_supply_delay=args.supply_delay,
                use_integrated=args.integrated,
            )
            exp = FactoryExperiment(
                config=config,
                mode="deterministic" if args.no_llm else "analytical",
                use_llm=use_llm, llm_client=client, mc_seed_base=mc_seed_base,
            )
            exp.factory_planner.llm_only = bool(getattr(args, "llm_only", False))
            meta["llm_only"] = exp.factory_planner.llm_only
            meta["optimiser_anchor"] = exp.factory_planner.optimiser_anchor
            capture.attach("cbpa")
            result = exp.run_all_phases()
            meta["shifts"] = [{
                "shift": 1,
                "phases": [_factory_phase_record(pr, exp.audit) for pr in result.phases],
            }]
            fm = result.final_metrics
            meta["final_kpis"] = None if fm is None else meta["shifts"][0]["phases"][-1]["kpis"]
            FactoryResultsExporter(str(out_dir), clean=False).export_all(result, audit=exp.audit)
        else:
            from cbpa.config.scenario import ScenarioConfig
            from cbpa.runner.experiment import CBPAExperiment
            from cbpa.runner.results_exporter import ResultsExporter

            n_shifts = max(args.shifts, 3) if args.full_demo else args.shifts
            config = ScenarioConfig(
                llm_only=bool(getattr(args, "llm_only", False)),
                optimiser_anchor=bool(getattr(args, "optimiser_anchor", False)),
                enable_macro_phase=args.full_demo,
                use_simulation=args.simulate and not args.integrated,
                use_integrated=args.integrated,
            )
            exp = CBPAExperiment(
                config=config, use_llm=use_llm, llm_client=client, mc_seed_base=mc_seed_base,
            )
            exp.planner.llm_only = bool(getattr(args, "llm_only", False))
            meta["llm_only"] = exp.planner.llm_only
            meta["optimiser_anchor"] = exp.planner.optimiser_anchor
            capture.attach("cbpa")
            exporter = ResultsExporter(str(out_dir), clean=False)
            if n_shifts > 1:
                multi = exp.run_multi_shift(n_shifts=n_shifts)
                meta["shifts"] = [
                    {
                        "shift": i + 1,
                        "summary": {
                            "demand_spike_pct": s.demand_spike_pct,
                            "phases_to_resolution": s.phases_to_resolution,
                            "vr_score_accepted": s.vr_score_accepted,
                            "total_constraint_violations": s.total_constraint_violations,
                            "learning_bias_available": s.learning_bias_available,
                            "candidates_skipped": s.candidates_skipped,
                            "experience_records_after": s.experience_records_after,
                        },
                        "phases": [_single_phase_record(pr, r.audit) for pr in r.phases],
                    }
                    for i, (s, r) in enumerate(zip(multi.shift_summaries, multi.shift_results))
                ]
                exporter.export_multi_shift(multi.shift_summaries)
                # Per-shift artefacts (audit trail, execution trace, Table-3 values):
                # the top-level files below hold the LAST shift only, which the
                # revision's read-through found misleading for the first-shift trace.
                for i, r in enumerate(multi.shift_results):
                    ResultsExporter(str(out_dir / f"shift_{i + 1:02d}"), clean=False).export_all(r)
                if multi.shift_results:
                    exporter.export_all(multi.shift_results[-1])
            else:
                result = exp.run_all_phases()
                meta["shifts"] = [{
                    "shift": 1,
                    "phases": [_single_phase_record(pr, result.audit) for pr in result.phases],
                }]
                exporter.export_all(result)
        meta["status"] = "ok"
    except Exception as exc:  # keep the batch going; record the failure
        meta["status"] = "error"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["traceback"] = traceback.format_exc()[-4000:]
        logger.error("run %d failed: %s", run_idx, exc)
    finally:
        capture.detach("cbpa")
        if exp is not None:
            try:
                exp.shutdown()
            except Exception:
                pass

    meta["duration_s"] = round(time.perf_counter() - t0, 1)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["fallbacks"] = {"count": capture.count("fallback"), "by_component": capture.by_component("fallback"),
                         "by_category": capture.by_category(), "events": capture.events}
    if client is not None:
        meta["llm_calls"] = client.stats()
        (out_dir / "llm_calls.json").write_text(json.dumps(client.export_records(), indent=2))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Repeated-run harness for the CBPA case studies")
    p.add_argument("--case", choices=["single", "factory"], required=True)
    p.add_argument("--runs", type=int, default=10, help="number of independent runs")
    p.add_argument("--start-index", type=int, default=0, help="first run index (for resuming)")
    p.add_argument("--no-llm", action="store_true", help="deterministic mode (no LLM calls)")
    p.add_argument("--optimiser-anchor", action="store_true",
                   help="add the certified constrained-optimiser schedule to the anchor pool of the initial, stabilisation, and next-shift phases")
    p.add_argument("--llm-only", action="store_true",
                   help="ablation: withhold the deterministic anchors so the pool holds LLM candidates only (H2 LLM-only condition)")
    p.add_argument("--provider", default="claude", choices=["claude", "openai"])
    p.add_argument("--model", default=None, help="model override (claude-sonnet-4-6 / gpt-4o)")
    p.add_argument("--shifts", type=int, default=3, help="single-cell: number of shifts")
    p.add_argument("--full-demo", action="store_true",
                   help="single-cell: enable macro-adaptation phase and >=3 shifts (paper setting)")
    p.add_argument("--simulate", action="store_true", help="single-cell: SimPy execution")
    p.add_argument("--integrated", action="store_true", help="Isaac Sim + BaSyx + OPC-UA execution")
    p.add_argument("--supply-delay", action="store_true", help="factory: enable V_C supply delay")
    p.add_argument("--no-operator-absence", action="store_true", help="factory: disable H2 absence")
    p.add_argument("--vary-mc-seed", action="store_true",
                   help="use a different Monte-Carlo seed base per run (default: fixed 42, "
                        "so that only LLM sampling varies)")
    p.add_argument("--output-root", default=None, help="directory for run_XX/ folders")
    p.add_argument("--resume", action="store_true", help="skip runs that already completed")
    p.add_argument("--no-aggregate", action="store_true", help="do not aggregate at the end")
    p.add_argument("--quiet", action="store_true", help="only warnings from the framework")
    return p


def default_output_root(args: argparse.Namespace) -> Path:
    parts = ["data/repeated", args.case]
    if args.case == "single":
        parts.append("full" if args.full_demo else f"{max(args.shifts, 1)}shifts")
    if args.no_llm:
        parts.append("nollm")
    elif getattr(args, "llm_only", False):
        parts.append("llmonly")
    elif getattr(args, "optimiser_anchor", False):
        parts.append("optanchor")
    else:
        model = args.model or ("claude-sonnet-4-6" if args.provider == "claude" else "gpt-4o")
        parts.append(model.replace("/", "-"))
    parts.append("integrated" if args.integrated else ("sim" if args.simulate else "analytical"))
    if args.vary_mc_seed:
        parts.append("varyseed")
    return Path("_".join(parts[:2]) + "_" + "_".join(parts[2:]))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    if args.case == "single" and not (args.simulate or args.integrated):
        print("  ERROR: the single-cell evaluator returns fixed Table-3 constants unless the schedule is "
              "executed through SimPy (--simulate) or the integrated stack (--integrated, analytical "
              "fallback when services are absent). Re-run with one of these flags.")
        return 2
    root = Path(args.output_root) if args.output_root else default_output_root(args)
    root.mkdir(parents=True, exist_ok=True)

    # One connectivity check up-front (outside the per-run instrumentation)
    llm_info: dict[str, Any] | None = None
    if not args.no_llm:
        probe = create_client(provider=args.provider, model=args.model)
        ok, msg = probe.check_connection()
        print(f"  LLM check: {msg}")
        if not ok:
            print("  ERROR: LLM unreachable; refusing to start a repeated-run batch "
                  "(every call would silently fall back to deterministic).")
            return 2
        llm_info = InstrumentedLLMClient(probe).sampling_settings()

    batch = {
        "case": args.case,
        "args": vars(args),
        "output_root": str(root),
        "llm": llm_info,
        "git_commit": _git_commit(),
        "prompt_hash": _prompt_hash(),
        "default_mc_seed": DEFAULT_MC_SEED,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
    }
    (root / "batch_meta.json").write_text(json.dumps(batch, indent=2, default=str))
    print(f"  Batch root: {root}   (commit {batch['git_commit']}, prompts {batch['prompt_hash']})")

    n_ok = n_err = n_skip = 0
    for run_idx in range(args.start_index, args.start_index + args.runs):
        out_dir = root / f"run_{run_idx:02d}"
        meta_path = out_dir / "run_meta.json"
        if args.resume and meta_path.exists():
            try:
                if json.loads(meta_path.read_text()).get("status") == "ok":
                    n_skip += 1
                    print(f"  [run {run_idx:02d}] already complete — skipped")
                    continue
            except Exception:
                pass
        print(f"\n{'=' * 70}\n  RUN {run_idx:02d} / {args.start_index + args.runs - 1}   "
              f"({args.case}, {'no-LLM' if args.no_llm else 'LLM'})\n{'=' * 70}")
        meta = run_one(args, run_idx, out_dir)
        if meta["status"] == "ok":
            n_ok += 1
            calls = (meta.get("llm_calls") or {}).get("total", {})
            print(f"  [run {run_idx:02d}] ok in {meta['duration_s']}s; "
                  f"LLM calls={calls.get('calls', 0)} exceptions={calls.get('exceptions', 0)} "
                  f"empty={calls.get('empty_output', 0)} schema_invalid={calls.get('schema_invalid', 0)} "
                  f"fallbacks={meta['fallbacks']['count']} llm_warnings={meta['fallbacks']['by_category']}")
        else:
            n_err += 1
            print(f"  [run {run_idx:02d}] ERROR: {meta.get('error')}")

    batch["finished_at"] = datetime.now(timezone.utc).isoformat()
    batch["runs_ok"], batch["runs_error"], batch["runs_skipped"] = n_ok, n_err, n_skip
    (root / "batch_meta.json").write_text(json.dumps(batch, indent=2, default=str))
    print(f"\n  Batch done: ok={n_ok} error={n_err} skipped={n_skip}  →  {root}")

    if not args.no_aggregate:
        try:
            from aggregate_repeated import aggregate
            aggregate(root)
        except Exception as exc:
            logger.warning("aggregation failed: %s", exc)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
