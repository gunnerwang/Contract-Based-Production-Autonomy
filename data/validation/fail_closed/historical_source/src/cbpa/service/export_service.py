"""Export service: in-memory CSV/LaTeX/JSON/ZIP for st.download_button."""

from __future__ import annotations

import io
import json
import zipfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cbpa.runner.experiment import ExperimentResult


class ExportService:
    """Produces in-memory export strings/bytes — no disk I/O required."""

    @staticmethod
    def table3_csv(result: ExperimentResult) -> str:
        lines = [
            "Plan,Throughput(u/h),DefectRate,Noise(dB),FatigueIndex,"
            "Energy(kWh),DeadlineGap(%),V/R,Feasible"
        ]
        for name in ["S1", "S2", "S3"]:
            m = result.schedule_metrics[name]
            vr = result.schedule_vr[name]
            f = result.schedule_feasibility[name]
            lines.append(
                f"{name},{m.throughput_uph},{m.defect_rate},{m.noise_db},"
                f"{m.fatigue_index},{m.energy_kwh},{m.deadline_gap_pct},"
                f"{vr.vr_score},{'Yes' if f.is_feasible else 'No'}"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def table3_latex(result: ExperimentResult) -> str:
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
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines) + "\n"

    @staticmethod
    def table3_json(result: ExperimentResult) -> str:
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
        return json.dumps(data, indent=2) + "\n"

    @staticmethod
    def full_results_json(result: ExperimentResult) -> str:
        """Full experiment dump: phases + audit + metrics."""
        data = {
            "phases": [
                {
                    "phase": pr.phase,
                    "description": pr.description,
                    "schedule": pr.schedule.model_dump() if pr.schedule else None,
                    "metrics": pr.metrics.model_dump() if pr.metrics else None,
                    "vr_score": pr.vr_score.model_dump() if pr.vr_score else None,
                    "feasibility": pr.feasibility.model_dump() if pr.feasibility else None,
                    "decision": pr.decision.model_dump() if pr.decision else None,
                }
                for pr in result.phases
            ],
            "audit": [e.model_dump() for e in result.audit.entries],
            "schedule_metrics": {
                name: m.model_dump()
                for name, m in result.schedule_metrics.items()
            },
        }
        return json.dumps(data, indent=2, default=str) + "\n"

    @staticmethod
    def audit_json(result: ExperimentResult) -> str:
        data = [e.model_dump() for e in result.audit.entries]
        return json.dumps(data, indent=2, default=str) + "\n"

    @staticmethod
    def results_zip(result: ExperimentResult) -> bytes:
        """ZIP containing all export formats."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("table3.csv", ExportService.table3_csv(result))
            zf.writestr("table3.tex", ExportService.table3_latex(result))
            zf.writestr("table3.json", ExportService.table3_json(result))
            zf.writestr("full_results.json", ExportService.full_results_json(result))
            zf.writestr("audit_trail.json", ExportService.audit_json(result))
        return buf.getvalue()
