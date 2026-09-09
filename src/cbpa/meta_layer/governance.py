"""Governance orchestration for escalation flow."""

from __future__ import annotations

import logging

from cbpa.meta_layer.audit_trail import AuditTrail
from cbpa.meta_layer.escalation_agent import EscalationAgent
from cbpa.models.escalation import (
    EscalationQuery,
    ManagerDecision,
    WorkingMode,
    WorkingModeTransition,
)
from cbpa.models.metrics import FeasibilityReport

logger = logging.getLogger(__name__)


class GovernanceOrchestrator:
    """Orchestrates the escalation flow when constraints conflict.

    Also tracks the human working-mode lifecycle (Legislator → Auditor →
    Partner → Auditor) that is a defining capability of CBPA.
    """

    def __init__(
        self,
        escalation_agent: EscalationAgent,
        audit: AuditTrail,
    ):
        self.escalation_agent = escalation_agent
        self.audit = audit

        # Working-mode state
        self.current_mode: WorkingMode = WorkingMode.LEGISLATOR
        self.mode_history: list[WorkingModeTransition] = []

    # ------------------------------------------------------------------
    # Working-mode tracking
    # ------------------------------------------------------------------

    def transition_mode(
        self,
        new_mode: WorkingMode,
        trigger: str,
        phase: str,
    ) -> WorkingModeTransition:
        """Record a human working-mode transition and emit an audit entry.

        Parameters
        ----------
        new_mode:
            The mode being entered.
        trigger:
            Human-readable description of what caused the transition.
        phase:
            Label for the experiment phase boundary, e.g. ``"1→2"``.
        """
        transition = WorkingModeTransition(
            from_mode=self.current_mode,
            to_mode=new_mode,
            trigger=trigger,
            phase=phase,
        )
        self.mode_history.append(transition)
        self.current_mode = new_mode

        # Derive an integer phase number for the audit trail (use the
        # destination phase where one exists, else parse from the label).
        # Support both Unicode arrow "→" and ASCII arrow "->".
        try:
            import re as _re
            _dest = _re.split(r"[-→>]+", phase)[-1].strip()
            audit_phase = int(_dest)
        except (ValueError, AttributeError):
            audit_phase = 0

        self.audit.record(
            phase=audit_phase,
            layer="Meta (Governance)",
            action="mode_transition",
            contract_state=f"working_mode={new_mode.value}",
            from_mode=transition.from_mode.value,
            to_mode=transition.to_mode.value,
            trigger=trigger,
            transition_phase=phase,
            working_mode=new_mode.value,
        )

        logger.info(
            f"Working mode: {transition.from_mode.value} → {new_mode.value} "
            f"(phase {phase}, trigger: {trigger})"
        )
        return transition

    def get_mode_summary(self) -> dict:
        """Return a summary of mode transitions and current state.

        Returns a dict with:
        - ``current_mode``: the active mode string
        - ``transition_count``: total number of transitions recorded
        - ``transitions``: ordered list of transition dicts
        - ``modes_used``: set of distinct modes that were active
        """
        return {
            "current_mode": self.current_mode.value,
            "transition_count": len(self.mode_history),
            "transitions": [t.model_dump(mode="json") for t in self.mode_history],
            "modes_used": sorted(
                {t.from_mode.value for t in self.mode_history}
                | {t.to_mode.value for t in self.mode_history}
            ),
        }

    # ------------------------------------------------------------------
    # Escalation flow
    # ------------------------------------------------------------------

    def handle_conflict(
        self,
        report: FeasibilityReport,
        demand_increase_pct: float = 20.0,
        phase: int = 4,
        assumption_drifts: list[str] | None = None,
    ) -> tuple[EscalationQuery, ManagerDecision]:
        """Run the full escalation flow.

        Parameters
        ----------
        report:
            Feasibility report from L3 constraint checking.
        demand_increase_pct:
            Percentage demand increase that triggered the conflict.
        phase:
            Experiment phase number for audit trail entries.
        assumption_drifts:
            Optional list of assumption attribution strings produced by
            :class:`~cbpa.layer4_execution.assumption_tracker.AssumptionTracker`.
            When provided, they are attached to the :class:`EscalationQuery`
            so the manager receives a root-cause explanation alongside the
            remediation options.

        Returns
        -------
        tuple[EscalationQuery, ManagerDecision]
        """
        # Step 1: Generate query
        query = self.escalation_agent.generate_query(report, demand_increase_pct)

        # Attach assumption attribution strings so the escalation message
        # to the manager includes which assumptions broke and why.
        if assumption_drifts:
            query.assumption_drifts = list(assumption_drifts)

        self.audit.record(
            phase=phase,
            layer="Meta",
            action="escalation_generated",
            contract_state="C1 → pending revision",
            conflict=query.conflict_summary,
            options=[o.label for o in query.options],
            assumption_drifts=query.assumption_drifts,
            working_mode=self.current_mode.value,
        )

        # Step 2: Manager decision
        decision = self.escalation_agent.simulate_manager_decision(query)

        self.audit.record(
            phase=phase,
            layer="Meta → L1",
            action="manager_decision",
            contract_state="C1 → C2",
            selected=decision.selected_option,
            rationale=decision.rationale,
            working_mode=self.current_mode.value,
        )

        logger.info(
            f"Phase {phase}: Manager selected {decision.selected_option}: "
            f"{decision.rationale}"
        )

        return query, decision
