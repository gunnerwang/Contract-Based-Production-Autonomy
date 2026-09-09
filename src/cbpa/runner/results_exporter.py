"""Export experiment results as LaTeX, CSV, and JSON."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cbpa.runner.experiment import ExperimentResult, ShiftSummary


class ResultsExporter:
    """Exports Table 3, execution trace, and audit trail."""

    def __init__(self, output_dir: str = "data/results", clean: bool = True):
        self.output_dir = Path(output_dir)
        if clean and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self, result: ExperimentResult) -> None:
        self.export_table3_csv(result)
        self.export_table3_latex(result)
        self.export_table3_json(result)
        self.export_trace(result)
        self.export_mode_transitions(result)
        self.export_llm_contribution(result)
        result.audit.export_json(self.output_dir / "audit_trail.json")

    def export_table3_csv(self, result: ExperimentResult) -> None:
        lines = ["Plan,Throughput(u/h),DefectRate,Noise(dB),FatigueIndex,Energy(kWh),DeadlineGap(%),V/R,Feasible"]
        for name in ["S1", "S2", "S3"]:
            m = result.schedule_metrics[name]
            vr = result.schedule_vr[name]
            f = result.schedule_feasibility[name]
            lines.append(
                f"{name},{m.throughput_uph},{m.defect_rate},{m.noise_db},"
                f"{m.fatigue_index},{m.energy_kwh},{m.deadline_gap_pct},"
                f"{vr.vr_score},{'Yes' if f.is_feasible else 'No'}"
            )
        (self.output_dir / "table3.csv").write_text("\n".join(lines) + "\n")

    def export_table3_latex(self, result: ExperimentResult) -> None:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Numerical comparison of candidate schedules.}",
            r"\label{tab:numerical-sim}",
            r"\small",
            r"\begin{tabular}{@{}lrrrrrrrl@{}}",
            r"\toprule",
            r"Plan & Throughput & Defect & Noise & Fatigue & Energy & Gap & V/R & Feasible \\",
            r"     & (units/h)  & rate   & (dB)  & Index   & (kWh)  & (\%) & score &         \\",
            r"\midrule",
        ]
        for name in ["S1", "S2", "S3"]:
            m = result.schedule_metrics[name]
            vr = result.schedule_vr[name]
            f = result.schedule_feasibility[name]
            feas = "Yes" if f.is_feasible else "No"
            lines.append(
                f"$S_{{{name[1]}}}$ & {m.throughput_uph} & {m.defect_rate} & "
                f"{m.noise_db} & {m.fatigue_index} & {int(m.energy_kwh)} & "
                f"{m.deadline_gap_pct} & {vr.vr_score} & {feas} \\\\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        (self.output_dir / "table3.tex").write_text("\n".join(lines) + "\n")

    def export_table3_json(self, result: ExperimentResult) -> None:
        data = {}
        for name in ["S1", "S2", "S3"]:
            m = result.schedule_metrics[name]
            vr = result.schedule_vr[name]
            f = result.schedule_feasibility[name]
            data[name] = {
                "throughput_uph": m.throughput_uph,
                "defect_rate": m.defect_rate,
                "noise_db": m.noise_db,
                "fatigue_index": m.fatigue_index,
                "energy_kwh": m.energy_kwh,
                "deadline_gap_pct": m.deadline_gap_pct,
                "vr_score": vr.vr_score,
                "feasible": f.is_feasible,
            }
        (self.output_dir / "table3.json").write_text(
            json.dumps(data, indent=2) + "\n"
        )

    def export_trace(self, result: ExperimentResult) -> None:
        """Export execution trace (Table 2)."""
        trace = []
        phase_map = {
            1: ("Shift start", "L1 → L2 → L3 → L4", "C1 verified"),
            2: ("Demand +20%", "L4 (Monitor)", "C1 active; gap"),
            3: ("Auto-replan", "L2 → L3", "C1 active"),
            4: ("Escalation", "Meta → L1", "C1 → C2"),
            5: ("Stabilization", "L2 → L3 → L5 → L4", "C2 verified"),
        }
        for pr in result.phases:
            trigger, layers, state = phase_map.get(
                pr.phase, ("", "", "")
            )
            trace.append({
                "phase": pr.phase,
                "trigger": trigger,
                "active_layers": layers,
                "contract_state": state,
                "outcome": pr.description,
                "working_mode": pr.working_mode.value if hasattr(pr.working_mode, "value") else pr.working_mode,
            })
        (self.output_dir / "execution_trace.json").write_text(
            json.dumps(trace, indent=2) + "\n"
        )

    def export_mode_transitions(self, result: ExperimentResult) -> None:
        """Export the human working-mode lifecycle as JSON.

        Writes ``working_modes.json`` containing:
        - The mode summary produced by ``GovernanceOrchestrator.get_mode_summary()``
          (current mode, transition count, modes used, full transition list)
        - A per-phase breakdown mapping each phase number to its active mode
        """
        phase_modes = {
            pr.phase: (pr.working_mode.value if hasattr(pr.working_mode, "value") else pr.working_mode)
            for pr in result.phases
        }

        data = {
            "description": (
                "Human working-mode lifecycle across the 5-phase CBPA experiment. "
                "Demonstrates CBPA's unique 'human role transitions' capability: "
                "Legislator (sets contracts) → Auditor (observes) → "
                "Partner (co-decides) → Auditor (reviews results)."
            ),
            "phase_modes": phase_modes,
            **result.mode_summary,
        }

        (self.output_dir / "working_modes.json").write_text(
            json.dumps(data, indent=2) + "\n"
        )

    # ------------------------------------------------------------------
    # Multi-shift learning curve export
    # ------------------------------------------------------------------

    def export_multi_shift(self, shift_summaries: list[ShiftSummary]) -> None:
        """Export per-shift learning-curve metrics as CSV and JSON.

        Demonstrates CBPA's *cumulative operational intelligence*: the
        learning store accumulates experience across shifts, enabling
        faster convergence and fewer violations on later shifts.
        """
        self._export_multi_shift_csv(shift_summaries)
        self._export_multi_shift_json(shift_summaries)

    def _export_multi_shift_csv(self, shift_summaries: list[ShiftSummary]) -> None:
        header = (
            "Shift,DemandSpike%,PhasesToResolution,VR_Accepted,"
            "ConstraintViolations,LearningBiasAvailable,"
            "CandidatesSkipped,ExperienceRecordsBefore,ExperienceRecordsAfter"
        )
        lines = [header]
        for s in shift_summaries:
            lines.append(
                f"{s.shift},{s.demand_spike_pct},{s.phases_to_resolution},"
                f"{s.vr_score_accepted},{s.total_constraint_violations},"
                f"{'Yes' if s.learning_bias_available else 'No'},"
                f"{s.candidates_skipped},"
                f"{s.experience_records_before},{s.experience_records_after}"
            )
        (self.output_dir / "multi_shift_learning_curve.csv").write_text(
            "\n".join(lines) + "\n"
        )

    def _export_multi_shift_json(self, shift_summaries: list[ShiftSummary]) -> None:
        data = {
            "description": (
                "Multi-shift learning curve demonstrating CBPA's cumulative "
                "operational intelligence. The learning store accumulates "
                "experience across shifts so that later shifts converge "
                "faster to robust solutions."
            ),
            "shifts": [
                {
                    "shift": s.shift,
                    "demand_spike_pct": s.demand_spike_pct,
                    "phases_to_resolution": s.phases_to_resolution,
                    "vr_score_accepted": s.vr_score_accepted,
                    "total_constraint_violations": s.total_constraint_violations,
                    "learning_bias_available": s.learning_bias_available,
                    "candidates_skipped": s.candidates_skipped,
                    "experience_records_before": s.experience_records_before,
                    "experience_records_after": s.experience_records_after,
                }
                for s in shift_summaries
            ],
        }
        (self.output_dir / "multi_shift_learning_curve.json").write_text(
            json.dumps(data, indent=2) + "\n"
        )

    # ------------------------------------------------------------------
    # LLM contribution provenance export
    # ------------------------------------------------------------------

    def export_llm_contribution(self, result: ExperimentResult) -> None:
        """Export LLM vs deterministic provenance analysis.

        Scans audit trail entries for ``candidate_sources`` and produces
        a per-phase breakdown showing how many candidates came from
        each source, which survived Pareto filtering, and which was
        ultimately selected.
        """
        phases: list[dict] = []
        total_llm = 0
        total_det = 0
        llm_selected = 0
        det_selected = 0
        llm_on_front = 0
        det_on_front = 0

        for entry in result.audit.entries:
            d = entry.details
            sources = d.get("candidate_sources")
            if not sources:
                continue
            phase = entry.phase
            selected = d.get("selected", "")
            selected_src = d.get("selected_source", "deterministic")
            non_dominated = d.get("non_dominated", d.get("pareto_front", []))

            n_llm = sum(1 for s in sources.values() if s == "llm")
            n_det = sum(1 for s in sources.values() if s != "llm")
            nd_llm = sum(1 for name in non_dominated if sources.get(name) == "llm")
            nd_det = sum(1 for name in non_dominated if sources.get(name) != "llm")

            total_llm += n_llm
            total_det += n_det
            if selected_src == "llm":
                llm_selected += 1
            else:
                det_selected += 1
            llm_on_front += nd_llm
            det_on_front += nd_det

            phases.append({
                "phase": phase,
                "action": entry.action,
                "total_candidates": n_llm + n_det,
                "llm_candidates": n_llm,
                "deterministic_candidates": n_det,
                "non_dominated_llm": nd_llm,
                "non_dominated_det": nd_det,
                "selected": selected,
                "selected_source": selected_src,
            })

        summary = {
            "description": (
                "LLM vs deterministic candidate provenance analysis. "
                "Shows per-phase breakdown of candidate origins, Pareto "
                "survival, and final selection source."
            ),
            "totals": {
                "llm_candidates": total_llm,
                "deterministic_candidates": total_det,
                "llm_on_pareto_front": llm_on_front,
                "det_on_pareto_front": det_on_front,
                "llm_selected_count": llm_selected,
                "det_selected_count": det_selected,
            },
            "phases": phases,
        }

        (self.output_dir / "llm_contribution.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
