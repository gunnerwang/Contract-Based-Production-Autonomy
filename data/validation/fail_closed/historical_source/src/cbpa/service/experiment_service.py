"""Experiment service: phase-by-phase execution with escalation handshake."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from cbpa.config.scenario import ScenarioConfig
from cbpa.models.escalation import EscalationQuery, ManagerDecision
from cbpa.runner.experiment import (
    CBPAExperiment,
    ExperimentResult,
    PhaseResult,
)

logger = logging.getLogger(__name__)


class ExperimentState(str, Enum):
    """Current state of the experiment service."""

    IDLE = "idle"
    RUNNING = "running"
    AWAITING_ESCALATION = "awaiting_escalation"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ExperimentStatus:
    """Snapshot of experiment progress."""

    state: ExperimentState = ExperimentState.IDLE
    current_phase: int = 0
    completed_phases: list[PhaseResult] = field(default_factory=list)
    escalation_query: EscalationQuery | None = None
    result: ExperimentResult | None = None
    error: str | None = None


class ExperimentService:
    """Wraps CBPAExperiment with phase-by-phase control for the UI.

    Key design: Phase 4 is a two-step handshake:
      1. run_phase(4) returns with state=AWAITING_ESCALATION
      2. UI renders escalation chooser
      3. resolve_escalation(decision) completes Phase 4
    """

    def __init__(self, config: ScenarioConfig | None = None, use_llm: bool = False):
        self._config = config or ScenarioConfig()
        self._use_llm = use_llm
        self._experiment: CBPAExperiment | None = None
        self._status = ExperimentStatus()
        # Internal references for phase-to-phase data passing
        self._p1: PhaseResult | None = None
        self._p3: PhaseResult | None = None

    @property
    def status(self) -> ExperimentStatus:
        return self._status

    def reset(self, config: ScenarioConfig | None = None) -> None:
        """Clear state for a new experiment."""
        if config is not None:
            self._config = config
        self._experiment = None
        self._status = ExperimentStatus()
        self._p1 = None
        self._p3 = None
        self._p4 = None

    def _ensure_experiment(self) -> CBPAExperiment:
        if self._experiment is None:
            self._experiment = CBPAExperiment(
                config=self._config, use_llm=self._use_llm
            )
        return self._experiment

    def run_all(
        self,
        on_phase_complete: Callable[[PhaseResult], None] | None = None,
    ) -> ExperimentResult:
        """Run all 5 phases, using auto-select for Phase 4 escalation."""
        for phase_num in range(1, 6):
            self.run_phase(phase_num)

            # Auto-resolve escalation in run_all mode
            if self._status.state == ExperimentState.AWAITING_ESCALATION:
                self.resolve_escalation(
                    ManagerDecision(
                        selected_option="Option C",
                        rationale="Prioritizing human well-being over deadline",
                    )
                )

            if on_phase_complete and self._status.completed_phases:
                on_phase_complete(self._status.completed_phases[-1])

        self._build_result()
        return self._status.result  # type: ignore[return-value]

    def run_phase(self, phase_num: int, manager_decision: ManagerDecision | None = None) -> PhaseResult | None:
        """Execute a single phase.

        For Phase 4, if no manager_decision is provided, the service enters
        AWAITING_ESCALATION state and returns None.  The UI must then call
        resolve_escalation() with the user's choice.
        """
        exp = self._ensure_experiment()
        self._status.state = ExperimentState.RUNNING
        self._status.current_phase = phase_num

        try:
            if phase_num == 1:
                pr = exp._phase1()
                self._p1 = pr

            elif phase_num == 2:
                assert self._p1 is not None, "Phase 1 must complete before Phase 2"
                pr = exp._phase2(self._p1)

            elif phase_num == 3:
                assert self._p1 is not None and self._p1.contract is not None
                pr = exp._phase3(self._p1.contract)
                self._p3 = pr

            elif phase_num == 4:
                assert self._p3 is not None and self._p3.feasibility is not None

                # Generate escalation query
                query = exp.escalation.generate_query(
                    self._p3.feasibility,
                    exp.config.demand_spike_pct,
                )
                exp.audit.record(
                    phase=4,
                    layer="Meta",
                    action="escalation_generated",
                    contract_state="C1 -> pending revision",
                    conflict=query.conflict_summary,
                    options=[o.label for o in query.options],
                )
                self._status.escalation_query = query

                if manager_decision is None:
                    # Enter handshake mode — wait for UI
                    self._status.state = ExperimentState.AWAITING_ESCALATION
                    return None

                # Decision provided inline
                return self._complete_phase4(exp, manager_decision)

            elif phase_num == 5:
                assert self._p1 is not None and self._p3 is not None
                assert self._p4 is not None
                pr = exp._phase5(self._p1, self._p3, self._p4)

            else:
                raise ValueError(f"Invalid phase number: {phase_num}")

            self._status.completed_phases.append(pr)
            if phase_num == 5:
                self._build_result()
                self._status.state = ExperimentState.COMPLETED
            else:
                self._status.state = ExperimentState.RUNNING
            return pr

        except Exception as e:
            logger.exception(f"Phase {phase_num} failed")
            self._status.state = ExperimentState.ERROR
            self._status.error = str(e)
            return None

    def resolve_escalation(self, decision: ManagerDecision) -> PhaseResult | None:
        """Complete Phase 4 after the user has chosen an escalation option."""
        if self._status.state != ExperimentState.AWAITING_ESCALATION:
            raise RuntimeError(
                f"Cannot resolve escalation in state {self._status.state}"
            )
        exp = self._ensure_experiment()
        return self._complete_phase4(exp, decision)

    def _complete_phase4(
        self, exp: CBPAExperiment, decision: ManagerDecision
    ) -> PhaseResult:
        """Record the manager decision and finalize Phase 4."""
        exp.audit.record(
            phase=4,
            layer="Meta -> L1",
            action="manager_decision",
            contract_state="C1 -> C2",
            selected=decision.selected_option,
            rationale=decision.rationale,
        )

        pr = PhaseResult(
            phase=4,
            decision=decision,
            description=f"Manager: {decision.selected_option} (keep constraints)",
        )
        self._p4 = pr
        self._status.completed_phases.append(pr)
        self._status.state = ExperimentState.RUNNING
        self._status.escalation_query = None
        return pr

    def get_escalation_query(self) -> EscalationQuery | None:
        """Return the pending escalation query, if any."""
        return self._status.escalation_query

    def _build_result(self) -> None:
        """Assemble ExperimentResult from completed phases."""
        if self._experiment is None:
            return

        result = ExperimentResult(audit=self._experiment.audit)
        result.phases = list(self._status.completed_phases)

        for pr in result.phases:
            name_map = {1: "S1", 3: "S2", 5: "S3"}
            sname = name_map.get(pr.phase)
            if sname and pr.metrics is not None:
                result.schedule_metrics[sname] = pr.metrics
            if sname and pr.vr_score is not None:
                result.schedule_vr[sname] = pr.vr_score
            if sname and pr.feasibility is not None:
                result.schedule_feasibility[sname] = pr.feasibility

        self._status.result = result
