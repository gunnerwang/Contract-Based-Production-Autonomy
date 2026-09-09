#!/usr/bin/env python3
"""Summarize existing ablation records without changing or rerunning evidence.

Single-cell rows use shift_01; run-root single-cell tables describe the last shift.
SD is the sample SD (ddof=1); the deterministic no-LLM condition has n=1,
so no SD is estimated. Certification is the recorded gate decision, cross-checked
against L3 where that event contains an explicit certified field. These are
empirical model checks, not real-world authorization or confidence guarantees.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCHES = {
    "single_hybrid": "single_full_claude-sonnet-4-6_integrated_v5",
    "single_nollm": "single_full_nollm_integrated",
    "single_llmonly": "single_full_claude-sonnet-4-6_llmonly_integrated",
    "factory_hybrid": "factory_claude-sonnet-4-6_analytical_v3",
    "factory_nollm": "factory_nollm_analytical",
    "factory_llmonly": "factory_claude-sonnet-4-6_llmonly_analytical",
    "factory_gemini_llmonly": "factory_gemini-3.8-flash_llmonly_analytical",
}


def stats(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def summarize():
    result = {
        "scope": "Existing first-shift single-cell and four-phase factory records; no evidence regenerated.",
        "acceptance": "Selected candidate is nominally feasible and recorded gate_certified is true.",
        "warning": "Selected-throughput statistics include rejected and uncertified candidates. Conditional throughput excludes those candidates; its denominator is the certified count, not total runs.",
        "batches": {},
    }
    for label, batch in BATCHES.items():
        single = label.startswith("single")
        path = ROOT / "data/repeated" / batch
        runs = sorted(path.glob("run_*"))
        phases = [1, 3, 5] if single else [1, 3, 5, 6]
        schedules = ["S1", "S2", "S3"] if single else ["FS1", "FS2", "FS3", "FS4"]
        output = {"path": str(path.relative_to(ROOT)), "stored_runs": len(runs), "phases": {}}
        for phase, schedule in zip(phases, schedules):
            rows = []
            for run in runs:
                location = run / "shift_01" if single else run
                table_name = "table3.json" if single else "factory_table.json"
                table = json.loads((location / table_name).read_text())[schedule]
                events = json.loads((location / "audit_trail.json").read_text())
                gates = [e["details"] for e in events if e["phase"] == phase and "gate_certified" in e.get("details", {})]
                assert len(gates) == 1, (label, run, phase)
                gate = gates[0]
                verifications = [e["details"] for e in events if e["phase"] == phase and e["layer"] == "L3"]
                assert len(verifications) == 1, (label, run, phase)
                verification = verifications[0]
                assert table["feasible"] == verification["feasible"], (label, run, phase)
                if "certified" in verification:
                    assert gate["gate_certified"] == verification["certified"], (label, run, phase)
                pool_empty = not any(score["feasible"] for score in gate["scores"].values())
                if "gate_no_feasible_candidate" in gate:
                    assert pool_empty == gate["gate_no_feasible_candidate"], (label, run, phase)
                rows.append({
                    "throughput": table["throughput_uph" if single else "total_uph"],
                    "nominal_feasible": table["feasible"],
                    "certified": gate["gate_certified"],
                    "no_feasible_candidate": pool_empty,
                })
            output["phases"][schedule] = {
                "phase": phase,
                "selected_throughput_uph": stats([r["throughput"] for r in rows]),
                "nominal_feasible_count": sum(r["nominal_feasible"] for r in rows),
                "certified_count": sum(r["certified"] for r in rows),
                "no_feasible_candidate_count": sum(r["no_feasible_candidate"] for r in rows),
                "certified_throughput_uph": stats([r["throughput"] for r in rows if r["nominal_feasible"] and r["certified"]]),
            }
        result["batches"][label] = output
    return result


if __name__ == "__main__":
    destination = ROOT / "data/baselines/ablation_acceptance_summary.json"
    destination.write_text(json.dumps(summarize(), indent=2) + "\n")
    print(destination)
