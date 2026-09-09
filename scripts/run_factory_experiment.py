#!/usr/bin/env python3
"""CLI entry point for the multi-cell CBPA factory experiment.

Usage:
    python scripts/run_factory_experiment.py --no-llm
    python scripts/run_factory_experiment.py --no-llm --supply-delay
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cbpa.config.scenario import FactoryScenarioConfig
from cbpa.llm.client import create_client
from cbpa.runner.factory_experiment import FactoryExperiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CBPA multi-cell factory experiment"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run in deterministic mode (no LLM calls)",
    )
    parser.add_argument(
        "--supply-delay",
        action="store_true",
        help="Enable V_C supply delay disturbance",
    )
    parser.add_argument(
        "--no-operator-absence",
        action="store_true",
        help="Disable operator absence disturbance",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: auto-generated from flags)",
    )
    parser.add_argument(
        "--integrated",
        action="store_true",
        help="Use Isaac Sim + BaSyx AAS + OPC-UA for execution",
    )
    parser.add_argument(
        "--isaac-url",
        type=str,
        default=None,
        help="Isaac Sim REST URL (default: http://localhost:8211)",
    )
    parser.add_argument(
        "--basyx-registry",
        type=str,
        default=None,
        help="BaSyx AAS registry URL (default: http://localhost:9082)",
    )
    parser.add_argument(
        "--opcua-endpoint",
        type=str,
        default=None,
        help="OPC-UA server endpoint (default: opc.tcp://localhost:4840/cbpa/)",
    )
    parser.add_argument(
        "--with-dashboard",
        action="store_true",
        help="Enable Streamlit dashboard live updates",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="claude",
        choices=["claude", "openai"],
        help="LLM provider (default: claude)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model override (e.g. claude-sonnet-4-6, gpt-4o)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    config_kwargs: dict = {
        "enable_operator_absence": not args.no_operator_absence,
        "enable_supply_delay": args.supply_delay,
        "use_integrated": args.integrated,
    }
    if args.isaac_url:
        config_kwargs["isaac_sim_url"] = args.isaac_url
    if args.basyx_registry:
        config_kwargs["basyx_registry_url"] = args.basyx_registry
    if args.opcua_endpoint:
        config_kwargs["opcua_endpoint"] = args.opcua_endpoint
    config = FactoryScenarioConfig(**config_kwargs)

    use_llm = not args.no_llm
    mode = "deterministic" if args.no_llm else "analytical"

    # Build LLM client
    llm_client = None
    if use_llm:
        model_name = args.model or (
            "claude-sonnet-4-6" if args.provider == "claude" else "gpt-4o"
        )
        print(f"\n  Connecting to LLM: {args.provider} ({model_name}) ...")
        llm_client = create_client(provider=args.provider, model=args.model)
        ok, msg = llm_client.check_connection()
        if ok:
            print(f"  LLM ready: {msg}")
        else:
            print(f"\n  WARNING: {msg}")
            print("  The experiment will fall back to DETERMINISTIC mode")
            print("  for any LLM call that fails.\n")
            if args.provider == "claude":
                print("  To fix: run 'claude login' and try again.")
            else:
                print("  To fix: set OPENAI_API_KEY and try again.")
            print()

    # Auto-generate output directory, mirroring the single-cell convention exactly:
    #   single-cell: data/results[_full][_nollm][_analytical|_sim|_integrated][_Nshifts]
    #   factory:     data/results_factory[_nollm][_analytical|_integrated][_supply_delay][_no_op_absence]
    # All parts are joined with "_" on the same base "data/results", so factory
    # results live alongside single-cell results in data/ at the same level.
    if args.output_dir is None:
        parts = ["data/results", "factory"]
        if args.no_llm:
            parts.append("nollm")
        if args.integrated:
            parts.append("integrated")
        else:
            parts.append("analytical")
        if args.supply_delay:
            parts.append("supply_delay")
        if args.no_operator_absence:
            parts.append("no_op_absence")
        args.output_dir = "_".join(parts)

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    exp = FactoryExperiment(config=config, mode=mode, use_llm=use_llm, llm_client=llm_client)

    # Ensure OPC-UA / integration bridges are released on exit
    import atexit
    atexit.register(exp.shutdown)

    # Live progress file — the Streamlit dashboard polls this
    live_progress_path = Path("/tmp/cbpa_factory_live.json")

    def _write_live_progress(pr) -> None:
        """Write incremental progress so the dashboard can pick it up."""
        # Read existing or start fresh
        existing = []
        if live_progress_path.exists():
            try:
                data = json.loads(live_progress_path.read_text())
                existing = data.get("phases", [])
            except Exception:
                existing = []
        existing.append({
            "phase": pr.phase,
            "name": pr.phase_name,
            "working_mode": pr.working_mode,
            "notes": pr.notes,
            "manager_decision": pr.manager_decision,
            "feasibility": {
                "is_feasible": pr.feasibility.is_feasible,
                "violations": pr.feasibility.violations,
            } if pr.feasibility else None,
            "stochastic_p_feasible": pr.stochastic_report.p_feasible if pr.stochastic_report else None,
            "factory_metrics": pr.factory_metrics.model_dump() if pr.factory_metrics else None,
        })
        live_progress_path.write_text(json.dumps({
            "state": "running",
            "current_phase": pr.phase,
            "total_phases": exp.total_phases,
            "phases": existing,
        }, indent=2, default=str))

    # Clear stale progress from previous run
    live_progress_path.write_text(json.dumps({"state": "starting", "phases": []}, indent=2))

    result = exp.run_all_phases(on_phase_complete=_write_live_progress)

    # Mark as completed
    try:
        data = json.loads(live_progress_path.read_text())
        data["state"] = "completed"
        if result.final_metrics:
            data["final_metrics"] = result.final_metrics.model_dump()
        live_progress_path.write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        pass

    # Print summary
    print("\n" + "=" * 70)
    print("CBPA FACTORY EXPERIMENT RESULTS")
    print("=" * 70)

    for phase_result in result.phases:
        print(f"\n--- Phase {phase_result.phase}: {phase_result.phase_name} ---")
        print(f"    Working mode: {phase_result.working_mode}")
        for note in phase_result.notes:
            print(f"    • {note}")

    if result.final_metrics:
        fm = result.final_metrics
        print(f"\n--- Final Factory Metrics ---")
        print(f"    Total throughput:    {fm.total_throughput_uph} uph")
        print(f"    Factory noise:       {fm.factory_noise_db} dB")
        print(f"    Cell balance loss:   {fm.cell_balance_loss_pct}%")
        print(f"    AGV utilization:     {fm.agv_utilization}")
        print(f"    Factory energy:      {fm.factory_energy_kwh} kWh")
        print(f"    Factory defect rate: {fm.factory_defect_rate}")
        print(f"    Operator fatigue:    {fm.operator_fatigue}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        # ── Identity metadata ──────────────────────────────────────────
        # These fields let any downstream tool (visualise_results.py,
        # paper scripts, CI checks) identify what produced this file
        # without relying on the directory name alone.
        "experiment_type": "factory",
        "mode": mode,                       # "analytical" | "deterministic"
        "timestamp": run_ts,                # ISO-8601 UTC
        "config": {
            "enable_operator_absence": config.enable_operator_absence,
            "enable_supply_delay":     config.enable_supply_delay,
            "use_integrated":          config.use_integrated,
        },
        # ── Phase trace ───────────────────────────────────────────────
        "phases": [
            {
                "phase": p.phase,
                "name": p.phase_name,
                "working_mode": p.working_mode,
                "notes": p.notes,
                "manager_decision": p.manager_decision,
            }
            for p in result.phases
        ],
    }
    if result.final_metrics:
        summary["final_metrics"] = result.final_metrics.model_dump()

    out_file = output_dir / "factory_experiment_result.json"
    out_file.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nResults saved to {out_file}")

    # ── Export structured artefacts (table, trace, modes, audit trail) ──
    try:
        from cbpa.runner.factory_results_exporter import FactoryResultsExporter
        exporter = FactoryResultsExporter(args.output_dir, clean=False)
        audit = getattr(exp, "audit", None)
        exporter.export_all(result, audit=audit)
        print(f"  factory_table.{{csv,tex,json}}, execution_trace.json,")
        print(f"  working_modes.json, audit_trail.json  →  {args.output_dir}/")
    except Exception as _exc:
        logging.getLogger(__name__).warning("Factory export failed: %s", _exc)

    # ── Generate figures ─────────────────────────────────────────────────
    viz_script = Path(__file__).resolve().parent / "visualize_factory_results.py"
    if viz_script.exists():
        try:
            subprocess.run(
                [sys.executable, str(viz_script), "--data-dir", args.output_dir],
                check=True,
            )
            print(f"Figures saved to {args.output_dir}/figures/")
        except Exception as _exc:
            logging.getLogger(__name__).warning("Figure generation failed: %s", _exc)


if __name__ == "__main__":
    main()
