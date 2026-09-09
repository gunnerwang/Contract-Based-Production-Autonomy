"""Write experiment results to a JSON file after each phase for live dashboard consumption.

Handles both single-cell ``PhaseResult`` and factory ``FactoryPhaseResult``
transparently — the dashboard gets a uniform JSON schema either way.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel file the dashboard polls to detect an active demo.
_DEFAULT_LIVE_FILE = "data/results/live_demo.json"


class LiveResultWriter:
    """Writes incremental experiment results to a JSON file.

    The dashboard reads this file with auto-refresh to display results
    as phases complete during a batch ``--full-demo`` run.

    Accepts both ``PhaseResult`` (single-cell) and ``FactoryPhaseResult``
    (factory) — uses duck typing to extract available fields.
    """

    def __init__(self, path: str = _DEFAULT_LIVE_FILE) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._phases: list[dict] = []
        self._start_time = time.time()
        self._shift: int = 1
        # Write initial state
        self._flush(state="running", current_phase=0)

    def set_shift(self, shift: int) -> None:
        self._shift = shift

    def on_phase_complete(self, pr: Any) -> None:
        """Callback invoked after each phase completes.

        Works with both ``PhaseResult`` and ``FactoryPhaseResult``:
        - ``PhaseResult``: reads ``.metrics``, ``.vr_score``, ``.pareto_front``, etc.
        - ``FactoryPhaseResult``: reads ``.factory_metrics``, ``.notes``, etc.
        """
        # Common fields
        working_mode = getattr(pr, "working_mode", None)
        if hasattr(working_mode, "value"):
            working_mode = working_mode.value

        entry: dict = {
            "shift": self._shift,
            "phase": pr.phase,
            "timestamp": round(time.time() - self._start_time, 2),
            "description": getattr(pr, "description", "")
                or getattr(pr, "phase_name", f"Phase {pr.phase}"),
            "working_mode": working_mode,
        }

        # Single-cell metrics (PhaseResult.metrics: SimulationMetrics)
        metrics = getattr(pr, "metrics", None)
        if metrics is not None:
            entry["metrics"] = {
                "throughput_uph": metrics.throughput_uph,
                "defect_rate": metrics.defect_rate,
                "noise_db": metrics.noise_db,
                "fatigue_index": metrics.fatigue_index,
                "energy_kwh": metrics.energy_kwh,
                "deadline_gap_pct": metrics.deadline_gap_pct,
            }

        # Factory metrics (FactoryPhaseResult.factory_metrics: FactoryMetrics)
        factory_metrics = getattr(pr, "factory_metrics", None)
        if factory_metrics is not None and metrics is None:
            max_fatigue = (
                max(factory_metrics.operator_fatigue.values())
                if factory_metrics.operator_fatigue else 0.0
            )
            entry["metrics"] = {
                "throughput_uph": factory_metrics.total_throughput_uph,
                "defect_rate": factory_metrics.factory_defect_rate,
                "noise_db": factory_metrics.factory_noise_db,
                "fatigue_index": max_fatigue,
                "energy_kwh": factory_metrics.factory_energy_kwh,
                "deadline_gap_pct": 0.0,
            }
            entry["factory"] = {
                "cell_balance_loss_pct": factory_metrics.cell_balance_loss_pct,
                "agv_utilization": factory_metrics.agv_utilization,
                "operator_fatigue": factory_metrics.operator_fatigue,
            }

        # V/R score
        vr_score = getattr(pr, "vr_score", None)
        if vr_score is not None:
            entry["vr_score"] = vr_score.vr_score

        # Feasibility
        feasibility = getattr(pr, "feasibility", None)
        if feasibility is not None:
            entry["feasible"] = feasibility.is_feasible
            entry["violations"] = feasibility.violations

        # Stochastic report (factory adds this)
        stoch = getattr(pr, "stochastic_report", None)
        if stoch is not None:
            entry["p_feasible"] = stoch.p_feasible

        # Pareto front
        pareto_front = getattr(pr, "pareto_front", None)
        if pareto_front is not None:
            entry["pareto"] = {
                "candidates": len(pareto_front.candidates),
                "non_dominated": pareto_front.non_dominated,
                "selected": pareto_front.selected,
            }

        # Manager decision
        decision = getattr(pr, "decision", None)
        if decision is not None:
            entry["decision"] = {
                "selected_option": decision.selected_option,
                "rationale": decision.rationale,
            }
        # Factory escalation (FactoryPhaseResult.manager_decision is a string)
        factory_decision = getattr(pr, "manager_decision", None)
        if factory_decision is not None and decision is None:
            entry["decision"] = {
                "selected_option": factory_decision,
                "rationale": "Prioritizing human well-being over deadline",
            }

        # Guard events
        guard_events = getattr(pr, "guard_events", None)
        if guard_events:
            entry["guard_events"] = [
                {
                    "guard": ev.guard_name,
                    "metric": ev.metric_name,
                    "value": ev.metric_value,
                    "threshold": ev.threshold,
                    "action": ev.action,
                }
                for ev in guard_events
            ]

        # Notes (factory phases use notes instead of description)
        notes = getattr(pr, "notes", None)
        if notes:
            entry["notes"] = notes

        self._phases.append(entry)
        self._flush(
            state="running",
            current_phase=pr.phase,
        )
        logger.debug("LiveResultWriter: phase %d written to %s", pr.phase, self.path)

    def finalize(self, result: Any = None) -> None:
        """Mark the experiment as complete."""
        extra: dict = {}
        if result is not None and hasattr(result, "mode_summary") and result.mode_summary:
            extra["mode_summary"] = result.mode_summary
        self._flush(state="completed", current_phase=0, **extra)
        logger.info("LiveResultWriter: finalized at %s", self.path)

    def _flush(self, **meta: object) -> None:
        payload = {
            "updated_at": time.time(),
            "elapsed_s": round(time.time() - self._start_time, 2),
            **meta,
            "phases": self._phases,
        }
        self.path.write_text(json.dumps(payload, indent=2, default=str))
