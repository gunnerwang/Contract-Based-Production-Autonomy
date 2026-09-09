"""Export factory experiment results as CSV, LaTeX, JSON, trace, and mode transitions.

Mirrors ``results_exporter.py`` for the single-cell experiment, adapted to
factory-level data structures (two cells, per-operator fatigue, AGV, variants).

Key schedules exported (FS1 / FS2 / FS3) correspond to the three lifecycle
stages that produce a new factory schedule:
    Phase 1  → FS1   Initial deployment
    Phase 3  → FS2   Autonomous replan attempt (usually infeasible)
    Phase 5  → FS3   Stabilized post-escalation
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cbpa.runner.factory_experiment import FactoryExperimentResult

# Phases that produce a named schedule result
_SCHEDULE_PHASES: dict[int, str] = {1: "FS1", 3: "FS2", 5: "FS3", 6: "FS4"}

# Execution trace metadata — mirrors single-cell phase_map
_PHASE_MAP: dict[int, tuple[str, str, str]] = {
    1: ("Shift start",        "L1 → L2 → L3 → L4",               "C1_factory verified"),
    2: ("Disturbance",        "L4 (Monitor)",                      "C1_factory active; drift"),
    3: ("Cross-cell replan",  "L2 → L3",                           "C1_factory active"),
    4: ("Escalation",         "Meta",                              "C1_factory → C2_factory (pre-authorised)"),
    5: ("Stabilization",      "L5 → L2 → L3 → L4 → L5",           "C2_factory active (pre-authorised)"),
    6: ("Shift-Close Releg.",  "L4 → L5 → L1 → L2 → L3 → L4",    "C2_factory → C3_factory (next shift)"),
}


class FactoryResultsExporter:
    """Exports factory_table.{csv,tex,json}, execution_trace.json,
    working_modes.json, and audit_trail.json for a factory experiment run.
    """

    def __init__(self, output_dir: str, clean: bool = False):
        # clean=False: keep factory_experiment_result.json already written
        self.output_dir = Path(output_dir)
        if clean and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------

    def export_all(self, result: FactoryExperimentResult, audit: Any = None) -> None:
        """Export all factory result artefacts."""
        self.export_table_csv(result)
        self.export_table_latex(result)
        self.export_table_json(result)
        self.export_trace(result)
        self.export_mode_transitions(result)
        if audit is not None:
            try:
                audit.export_json(self.output_dir / "audit_trail.json")
            except Exception:
                pass
            self.export_llm_contribution(audit)

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_rows(self, result: FactoryExperimentResult) -> dict[str, dict]:
        """Return {FS1, FS2, FS3} metric dicts from the matching phases."""
        rows: dict[str, dict] = {}
        for pr in result.phases:
            label = _SCHEDULE_PHASES.get(pr.phase)
            if label is None or pr.factory_metrics is None:
                continue
            fm = pr.factory_metrics
            cellA = fm.cell_metrics.get("A") if fm.cell_metrics else None
            cellB = fm.cell_metrics.get("B") if fm.cell_metrics else None
            feas  = pr.feasibility
            rows[label] = {
                "cellA_uph":              round(cellA.throughput_uph if cellA else 0.0, 1),
                "cellB_uph":              round(cellB.throughput_uph if cellB else 0.0, 1),
                "total_uph":              round(fm.total_throughput_uph, 1),
                "noise_db":               round(fm.factory_noise_db, 1),
                "cell_balance_loss_pct":  round(fm.cell_balance_loss_pct, 1),
                "agv_utilization":        round(fm.agv_utilization, 3),
                "fatigue_h1":             round(fm.operator_fatigue.get("H1", 0.0), 4),
                "fatigue_h2":             round(fm.operator_fatigue.get("H2", 0.0), 4),
                "fatigue_h3":             round(fm.operator_fatigue.get("H3", 0.0), 4),
                "energy_kwh":             round(fm.factory_energy_kwh, 0),
                "defect_rate":            round(fm.factory_defect_rate, 4),
                "feasible":               feas.is_feasible if feas is not None else None,
            }
        return rows

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def export_table_csv(self, result: FactoryExperimentResult) -> None:
        rows = self._extract_rows(result)
        header = (
            "Plan,CellA_uph,CellB_uph,Total_uph,Noise_dB,"
            "CellBalance_Loss%,AGV_Util,H1_Fatigue,H2_Fatigue,H3_Fatigue,"
            "Energy_kWh,DefectRate,Feasible"
        )
        lines = [header]
        for label in ["FS1", "FS2", "FS3", "FS4"]:
            r = rows.get(label)
            if r is None:
                continue
            feas = "Yes" if r["feasible"] else ("No" if r["feasible"] is not None else "N/A")
            lines.append(
                f"{label},{r['cellA_uph']},{r['cellB_uph']},{r['total_uph']},"
                f"{r['noise_db']},{r['cell_balance_loss_pct']},{r['agv_utilization']},"
                f"{r['fatigue_h1']},{r['fatigue_h2']},{r['fatigue_h3']},"
                f"{int(r['energy_kwh'])},{r['defect_rate']},{feas}"
            )
        (self.output_dir / "factory_table.csv").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # LaTeX
    # ------------------------------------------------------------------

    def export_table_latex(self, result: FactoryExperimentResult) -> None:
        rows = self._extract_rows(result)
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Factory schedule comparison: initial deployment (FS1), "
            r"autonomous replan (FS2), and stabilised post-escalation (FS3).}",
            r"\label{tab:factory-schedules}",
            r"\small",
            r"\begin{tabular}{@{}lrrrrrrrrrrl@{}}",
            r"\toprule",
            r"Plan & $\dot{Q}_A$ & $\dot{Q}_B$ & $\dot{Q}$ "
            r"& Noise & Bal.loss & AGV & $F_{H1}$ & $F_{H2}$ & $F_{H3}$ "
            r"& Energy & Feas. \\",
            r"     & (u/h) & (u/h) & (u/h) & (dB) & (\%) & util "
            r"& & & & (kWh) & \\",
            r"\midrule",
        ]
        tex_name = {"FS1": r"$\mathit{FS}_1$",
                    "FS2": r"$\mathit{FS}_2$",
                    "FS3": r"$\mathit{FS}_3$",
                    "FS4": r"$\mathit{FS}_4$"}
        for label in ["FS1", "FS2", "FS3", "FS4"]:
            r = rows.get(label)
            if r is None:
                continue
            if r["feasible"] is True:
                feas = r"\checkmark"
            elif r["feasible"] is False:
                feas = r"\texttimes"
            else:
                feas = "---"
            lines.append(
                f"{tex_name[label]} & {r['cellA_uph']:.1f} & {r['cellB_uph']:.1f} "
                f"& {r['total_uph']:.1f} & {r['noise_db']:.1f} "
                f"& {r['cell_balance_loss_pct']:.1f} & {r['agv_utilization']:.2f} "
                f"& {r['fatigue_h1']:.3f} & {r['fatigue_h2']:.3f} & {r['fatigue_h3']:.3f} "
                f"& {int(r['energy_kwh'])} & {feas} \\\\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        (self.output_dir / "factory_table.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def export_table_json(self, result: FactoryExperimentResult) -> None:
        rows = self._extract_rows(result)
        (self.output_dir / "factory_table.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )

    # ------------------------------------------------------------------
    # Execution trace
    # ------------------------------------------------------------------

    def export_trace(self, result: FactoryExperimentResult) -> None:
        trace = []
        for pr in result.phases:
            trigger, layers, state = _PHASE_MAP.get(pr.phase, ("", "", ""))
            trace.append({
                "phase":            pr.phase,
                "phase_name":       pr.phase_name,
                "trigger":          trigger,
                "active_layers":    layers,
                "contract_state":   state,
                "working_mode":     pr.working_mode,
                "notes":            pr.notes,
                "manager_decision": pr.manager_decision,
            })
        (self.output_dir / "execution_trace.json").write_text(
            json.dumps(trace, indent=2) + "\n"
        )

    # ------------------------------------------------------------------
    # Working-mode transitions
    # ------------------------------------------------------------------

    def export_mode_transitions(self, result: FactoryExperimentResult) -> None:
        phases = result.phases
        phase_modes = {p.phase: p.working_mode for p in phases}
        modes_used: list[str] = list(dict.fromkeys(p.working_mode for p in phases))

        transitions: list[dict] = []
        for i in range(1, len(phases)):
            if phases[i].working_mode != phases[i - 1].working_mode:
                transitions.append({
                    "from_mode": phases[i - 1].working_mode,
                    "to_mode":   phases[i].working_mode,
                    "phase":     phases[i].phase,
                    "trigger":   phases[i].phase_name,
                })

        data = {
            "description": (
                "Human working-mode lifecycle across the 6-phase factory CBPA experiment. "
                "LEGISLATOR (sets cross-cell contract and variant routing) → "
                "AUDITOR (monitors disturbances and operator absence) → "
                "PARTNER (co-decides escalation option) → "
                "AUDITOR (reviews stabilised factory state) → "
                "LEGISLATOR (shift-close: L5 learning feeds L1, rewrites C3_factory "
                "and pre-stages FS4 for next shift)."
            ),
            "phase_modes":      phase_modes,
            "current_mode":     phases[-1].working_mode if phases else "",
            "transition_count": len(transitions),
            "modes_used":       modes_used,
            "transitions":      transitions,
        }
        (self.output_dir / "working_modes.json").write_text(
            json.dumps(data, indent=2) + "\n"
        )

    # ------------------------------------------------------------------
    # LLM contribution provenance
    # ------------------------------------------------------------------

    def export_llm_contribution(self, audit: Any) -> None:
        """Export LLM vs deterministic provenance from the audit trail."""
        entries = getattr(audit, "entries", [])
        if not entries:
            return

        phases: list[dict] = []
        total_llm = 0
        total_det = 0
        llm_selected = 0
        det_selected = 0
        llm_on_front = 0
        det_on_front = 0

        for entry in entries:
            d = entry.details if hasattr(entry, "details") else entry
            sources = d.get("candidate_sources")
            if not sources:
                continue
            phase = entry.phase if hasattr(entry, "phase") else d.get("phase", "?")
            selected = d.get("selected", "")
            selected_src = d.get("selected_source", "deterministic")
            non_dominated = d.get("pareto_front", [])

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
                "action": entry.action if hasattr(entry, "action") else d.get("action", ""),
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
                "LLM vs deterministic candidate provenance for the factory experiment. "
                "Shows per-phase candidate origins, Pareto survival, and selection."
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
