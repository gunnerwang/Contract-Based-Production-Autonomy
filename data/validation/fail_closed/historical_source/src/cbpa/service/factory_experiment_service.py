"""Factory experiment service: phase-by-phase execution with escalation handshake.

Mirrors ExperimentService but wraps FactoryExperiment instead of CBPAExperiment.
Phase 4 uses a two-step handshake identical to the single-cell dashboard:
  1. run_phase(4) → generates escalation query, enters AWAITING_ESCALATION
  2. UI renders escalation chooser
  3. resolve_escalation(decision) → completes Phase 4 with the manager's choice
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from cbpa.config.scenario import FactoryScenarioConfig
from cbpa.models.escalation import EscalationQuery, ManagerDecision
from cbpa.runner.factory_experiment import (
    FactoryExperiment,
    FactoryExperimentResult,
    FactoryPhaseResult,
    make_c1_factory,
)
from cbpa.service.experiment_service import ExperimentState

logger = logging.getLogger(__name__)


@dataclass
class FactoryExperimentStatus:
    """Snapshot of factory experiment progress."""

    state: ExperimentState = ExperimentState.IDLE
    current_phase: int = 0
    completed_phases: list[FactoryPhaseResult] = field(default_factory=list)
    escalation_query: EscalationQuery | None = None
    result: FactoryExperimentResult | None = None
    error: str | None = None


class FactoryExperimentService:
    """Wraps FactoryExperiment with phase-by-phase control for the UI.

    Key design: Phase 4 is a two-step handshake:
      1. run_phase(4) returns with state=AWAITING_ESCALATION
      2. UI renders escalation chooser
      3. resolve_escalation(decision) completes Phase 4
    """

    def __init__(
        self,
        config: FactoryScenarioConfig | None = None,
        use_llm: bool = False,
    ):
        self._config = config or FactoryScenarioConfig()
        self._use_llm = use_llm
        self._experiment: FactoryExperiment | None = None
        self._status = FactoryExperimentStatus()

    @property
    def status(self) -> FactoryExperimentStatus:
        return self._status

    def reset(self, config: FactoryScenarioConfig | None = None) -> None:
        if config is not None:
            self._config = config
        self._experiment = None
        self._status = FactoryExperimentStatus()

    def _ensure_experiment(self) -> FactoryExperiment:
        if self._experiment is None:
            self._experiment = FactoryExperiment(
                config=self._config, use_llm=self._use_llm
            )
        return self._experiment

    # ── Run all (auto-resolves escalation) ────────────────────────────

    def run_all(
        self,
        on_phase_complete: Callable[[FactoryPhaseResult], None] | None = None,
    ) -> FactoryExperimentResult:
        """Run all rounds, auto-resolving the Phase 4 escalation."""
        exp = self._ensure_experiment()
        total = exp.total_phases

        for phase_num in range(1, total + 1):
            self.run_phase(phase_num)

            # Auto-resolve escalation in run_all mode
            if self._status.state == ExperimentState.AWAITING_ESCALATION:
                self.resolve_escalation(
                    ManagerDecision(
                        selected_option="Option D",
                        rationale="Prioritise human well-being over deadline (auto)",
                    )
                )

            if on_phase_complete and self._status.completed_phases:
                on_phase_complete(self._status.completed_phases[-1])

        self._status.state = ExperimentState.COMPLETED
        self._status.result = exp.build_result(self._status.completed_phases)
        return self._status.result

    # ── Step-by-step execution ────────────────────────────────────────

    def run_phase(self, phase_num: int) -> FactoryPhaseResult | None:
        """Execute a single phase.

        For Phase 4, enters AWAITING_ESCALATION and returns None.
        Call resolve_escalation() to complete it.
        """
        exp = self._ensure_experiment()
        self._status.state = ExperimentState.RUNNING
        self._status.current_phase = phase_num

        try:
            # Phase 4 special handling: split into query + decision
            if phase_num == 4:
                return self._start_phase4(exp)

            pr = exp.run_single_phase(phase_num, prior=self._status.completed_phases)
            if pr is not None:
                self._status.completed_phases.append(pr)
                total = exp.total_phases
                if pr.phase >= total:
                    self._status.state = ExperimentState.COMPLETED
                    self._status.result = exp.build_result(self._status.completed_phases)
                else:
                    self._status.state = ExperimentState.RUNNING
            return pr

        except Exception as e:
            logger.exception("Factory phase %d failed", phase_num)
            self._status.state = ExperimentState.ERROR
            self._status.error = str(e)
            return None

    # ── Escalation handshake ──────────────────────────────────────────

    def _start_phase4(self, exp: FactoryExperiment) -> None:
        """Generate the escalation query and pause for manager decision."""
        p3 = self._status.completed_phases[2]  # Phase 3 result
        report = p3.feasibility

        query = exp.escalation.generate_factory_query(
            report, demand_increase_pct=exp.config.demand_spike_pct,
        )

        exp.audit.record(
            phase=4, layer="Meta",
            action="conflict_raised",
            contract_state="C1_factory → conflict",
            working_mode="partner",
            conflict=query.conflict_summary,
            options_count=len(query.options),
        )

        self._status.escalation_query = query
        self._status.state = ExperimentState.AWAITING_ESCALATION
        return None

    def resolve_escalation(self, decision: ManagerDecision) -> FactoryPhaseResult | None:
        """Complete Phase 4 after the user has chosen an option."""
        if self._status.state != ExperimentState.AWAITING_ESCALATION:
            raise RuntimeError(
                f"Cannot resolve escalation in state {self._status.state}"
            )
        exp = self._ensure_experiment()
        return self._complete_phase4(exp, decision)

    def _complete_phase4(
        self, exp: FactoryExperiment, decision: ManagerDecision
    ) -> FactoryPhaseResult:
        """Record the manager decision and finalize Phase 4."""
        query = self._status.escalation_query

        exp.audit.record(
            phase=4, layer="Meta",
            action="manager_decision_received",
            contract_state="C1_factory → pending revision",
            working_mode="partner",
            selected=decision.selected_option,
            rationale=decision.rationale,
        )

        # Create C2_factory — the pre-authorised contract for Phase 5
        c2 = make_c1_factory()
        c2 = c2.model_copy(update={"name": "C2_factory"})
        c2.context["manager_decision"] = decision.selected_option
        c2.context["deadline_gap_acknowledged"] = True

        exp.audit.record(
            phase=4, layer="Meta",
            action="contract_preauthorised",
            contract_state=f"C1_factory → {c2.name}",
            working_mode="partner",
            contract=c2.name,
        )

        options_dicts = []
        if query:
            options_dicts = [
                {"label": o.label, "description": o.description,
                 "preserves_human_constraints": o.preserves_human_constraints,
                 "expected_deadline_gap_pct": o.expected_deadline_gap_pct}
                for o in query.options
            ]

        pr = FactoryPhaseResult(
            phase=4,
            phase_name="Human-in-the-Loop Escalation",
            contract=c2,
            escalation_options=options_dicts,
            manager_decision=decision.selected_option,
            notes=[
                f"Conflict: {query.conflict_summary}" if query else "Escalation",
                f"{len(options_dicts)} remediation options presented to manager.",
                f"Manager selected: {decision.selected_option} — {decision.rationale}",
                f"Contract pre-authorised: {c2.name}",
            ],
        )

        self._status.completed_phases.append(pr)
        self._status.state = ExperimentState.RUNNING
        self._status.escalation_query = None
        return pr

    def get_escalation_query(self) -> EscalationQuery | None:
        """Return the pending escalation query, if any."""
        return self._status.escalation_query
