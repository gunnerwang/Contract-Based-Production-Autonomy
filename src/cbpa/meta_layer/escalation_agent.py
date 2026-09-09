"""Meta-layer: LLM-based escalation explanation + options."""

from __future__ import annotations

import logging

from cbpa.llm.client import ClaudeClient
from cbpa.llm.prompts import (
    ESCALATION_SYSTEM,
    ESCALATION_TOOL_SCHEMA,
    FACTORY_ESCALATION_SYSTEM,
    FACTORY_ESCALATION_TOOL_SCHEMA,
)
from cbpa.models.escalation import (
    EscalationQuery,
    ManagerDecision,
    RemediationOption,
)
from cbpa.models.metrics import FeasibilityReport

logger = logging.getLogger(__name__)


class EscalationAgent:
    """Generates human-readable escalation queries with remediation options.

    LLM mode: Claude composes the explanation.
    Deterministic mode: returns the paper's exact 3 options.
    """

    def __init__(
        self,
        client: ClaudeClient | None = None,
        use_llm: bool = False,
    ):
        self.client = client
        self.use_llm = use_llm and client is not None and client.is_available

    def generate_query(
        self,
        report: FeasibilityReport,
        demand_increase_pct: float = 20.0,
    ) -> EscalationQuery:
        if self.use_llm:
            return self._generate_llm(report, demand_increase_pct)
        return self._generate_deterministic(report, demand_increase_pct)

    def _generate_deterministic(
        self,
        report: FeasibilityReport,
        demand_increase_pct: float,
    ) -> EscalationQuery:
        """Return the paper's exact escalation options."""
        return EscalationQuery(
            conflict_summary=(
                f"Constraint Conflict: Cannot meet new Demand (+{demand_increase_pct:.0f}%) "
                "without violating Human Fatigue and Noise limits."
            ),
            violated_constraints=report.violations,
            demand_increase_pct=demand_increase_pct,
            options=[
                RemediationOption(
                    label="Option A",
                    description="Relax Fatigue Limit to 0.6 (Allows high-speed mode)",
                    relaxes_constraint="FatigueIndex",
                    new_limit=0.6,
                    expected_deadline_gap_pct=0.0,
                    preserves_human_constraints=False,
                ),
                RemediationOption(
                    label="Option B",
                    description="Relax Noise Limit to 90dB (Allows robot acceleration)",
                    relaxes_constraint="Noise",
                    new_limit=90.0,
                    expected_deadline_gap_pct=0.0,
                    preserves_human_constraints=False,
                ),
                RemediationOption(
                    label="Option C",
                    description="Keep Human Constraints (Miss deadline by ~15%)",
                    relaxes_constraint=None,
                    new_limit=None,
                    expected_deadline_gap_pct=15.0,
                    preserves_human_constraints=True,
                ),
            ],
        )

    def _generate_llm(
        self,
        report: FeasibilityReport,
        demand_increase_pct: float,
    ) -> EscalationQuery:
        """Use Claude to generate escalation options."""
        assert self.client is not None

        user_msg = (
            f"A schedule was rejected with violations:\n"
            f"{chr(10).join(report.violations)}\n\n"
            f"Demand increased by {demand_increase_pct}%.\n"
            f"Constraint margins: {report.constraint_margins}\n\n"
            "Generate a conflict summary and 3 ranked remediation options "
            "for the production manager."
        )

        try:
            result = self.client.query_structured(
                system=ESCALATION_SYSTEM,
                user_message=user_msg,
                tool_name="escalation_query",
                tool_schema=ESCALATION_TOOL_SCHEMA,
                tool_description="Generate escalation options for manager",
            )

            options = [
                RemediationOption(
                    label=o.get("label", f"Option {i+1}"),
                    description=o["description"],
                    relaxes_constraint=o.get("relaxes_constraint"),
                    new_limit=o.get("new_limit"),
                    expected_deadline_gap_pct=o.get("expected_deadline_gap_pct", 0),
                    preserves_human_constraints=o.get("preserves_human_constraints", False),
                )
                for i, o in enumerate(result.get("options", []))
            ]

            return EscalationQuery(
                conflict_summary=result.get("conflict_summary", "Constraint conflict"),
                violated_constraints=report.violations,
                demand_increase_pct=demand_increase_pct,
                options=options or self._generate_deterministic(report, demand_increase_pct).options,
            )
        except Exception as e:
            print(f"  [LLM FALLBACK] Escalation: {e} -> using deterministic")
            logger.warning(f"LLM escalation failed: {e}, falling back to deterministic")
            return self._generate_deterministic(report, demand_increase_pct)

    def simulate_manager_decision(
        self,
        query: EscalationQuery,
        auto_select: str = "Option C",
    ) -> ManagerDecision:
        """Simulate the manager selecting an option.

        In the paper, the manager selects Option C (keep constraints).
        """
        return ManagerDecision(
            selected_option=auto_select,
            rationale="Prioritizing human well-being over deadline",
        )

    # ------------------------------------------------------------------
    # Factory-level escalation (multi-cell, 5 options)
    # ------------------------------------------------------------------

    def generate_factory_query(
        self,
        report: FeasibilityReport,
        demand_increase_pct: float = 20.0,
    ) -> EscalationQuery:
        """Generate factory-level escalation with 5 remediation categories."""
        if self.use_llm:
            return self._generate_factory_llm(report, demand_increase_pct)
        return self._generate_factory_deterministic(report, demand_increase_pct)

    def _generate_factory_deterministic(
        self,
        report: FeasibilityReport,
        demand_increase_pct: float,
    ) -> EscalationQuery:
        """Return 5-option factory escalation matching the CBPA paper categories."""
        return EscalationQuery(
            conflict_summary=(
                f"Factory Constraint Conflict: Cannot meet +{demand_increase_pct:.0f}% "
                "demand across both cells without violating per-operator fatigue "
                "limits or combined factory noise constraint (82 dB)."
            ),
            violated_constraints=report.violations,
            demand_increase_pct=demand_increase_pct,
            options=[
                RemediationOption(
                    label="Option A",
                    description=(
                        "Relax H1 fatigue limit from 0.4 to 0.5. Cell A runs "
                        "high-speed mode to compensate for Cell B's reduced capacity."
                    ),
                    relaxes_constraint="FatigueIndex_H1",
                    new_limit=0.5,
                    expected_deadline_gap_pct=5.0,
                    preserves_human_constraints=False,
                ),
                RemediationOption(
                    label="Option B",
                    description=(
                        "Drop V_C from this shift entirely. Cell B handles only "
                        "V_A and V_B at comfortable pace. ~15% revenue loss."
                    ),
                    relaxes_constraint=None,
                    new_limit=None,
                    expected_deadline_gap_pct=15.0,
                    preserves_human_constraints=True,
                ),
                RemediationOption(
                    label="Option C",
                    description=(
                        "Extend shift by 1 hour with H3 as overtime backup in "
                        "Cell B. Union rules permit 1h overtime with consent. "
                        "Cost increase ~12%."
                    ),
                    relaxes_constraint=None,
                    new_limit=None,
                    expected_deadline_gap_pct=3.0,
                    preserves_human_constraints=True,
                ),
                RemediationOption(
                    label="Option D",
                    description=(
                        "Accept 10% lower total factory throughput. Both cells "
                        "run at comfortable pace. All human constraints preserved."
                    ),
                    relaxes_constraint=None,
                    new_limit=None,
                    expected_deadline_gap_pct=10.0,
                    preserves_human_constraints=True,
                ),
                RemediationOption(
                    label="Option E",
                    description=(
                        "Redistribute: Cell A handles all V_A and V_B. Cell B "
                        "handles only V_C with H3 (reduced pace, automated testing). "
                        "Move H1 to cover both cells' inspection during breaks."
                    ),
                    relaxes_constraint=None,
                    new_limit=None,
                    expected_deadline_gap_pct=8.0,
                    preserves_human_constraints=True,
                ),
            ],
        )

    def _generate_factory_llm(
        self,
        report: FeasibilityReport,
        demand_increase_pct: float,
    ) -> EscalationQuery:
        """Use LLM for factory-level escalation options."""
        assert self.client is not None

        user_msg = (
            f"Factory-level schedule was rejected with violations:\n"
            f"{chr(10).join(report.violations)}\n\n"
            f"Demand increased by {demand_increase_pct}%.\n"
            f"Constraint margins: {report.constraint_margins}\n\n"
            "Generate a conflict summary and 5 ranked remediation options "
            "spanning: constraint relaxation, product mix change, overtime/backup, "
            "throughput reduction, and cross-cell redistribution."
        )

        try:
            result = self.client.query_structured(
                system=FACTORY_ESCALATION_SYSTEM,
                user_message=user_msg,
                tool_name="factory_escalation_query",
                tool_schema=FACTORY_ESCALATION_TOOL_SCHEMA,
                tool_description="Generate factory escalation options for manager",
            )

            options = [
                RemediationOption(
                    label=o.get("label", f"Option {i+1}"),
                    description=o["description"],
                    relaxes_constraint=o.get("relaxes_constraint"),
                    new_limit=o.get("new_limit"),
                    expected_deadline_gap_pct=o.get("expected_deadline_gap_pct", 0),
                    preserves_human_constraints=o.get("preserves_human_constraints", False),
                )
                for i, o in enumerate(result.get("options", []))
            ]

            return EscalationQuery(
                conflict_summary=result.get("conflict_summary", "Factory constraint conflict"),
                violated_constraints=report.violations,
                demand_increase_pct=demand_increase_pct,
                options=options or self._generate_factory_deterministic(report, demand_increase_pct).options,
            )
        except Exception as e:
            print(f"  [LLM FALLBACK] Factory escalation: {e} -> using deterministic")
            logger.warning(f"LLM factory escalation failed: {e}, falling back")
            return self._generate_factory_deterministic(report, demand_increase_pct)
