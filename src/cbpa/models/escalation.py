"""Escalation and governance data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class WorkingMode(str, Enum):
    """Human working mode in the CBPA governance loop.

    Represents the three distinct roles the human manager occupies at
    different points in the experiment, corresponding to the paper's
    'human role transitions' capability.
    """

    LEGISLATOR = "legislator"  # Human sets contracts and constraints
    PARTNER = "partner"        # Human and system co-decide (escalation negotiation)
    AUDITOR = "auditor"        # Human reviews after-the-fact; system runs autonomously


class WorkingModeTransition(BaseModel):
    """Records a single human-role transition in the governance log."""

    from_mode: WorkingMode
    to_mode: WorkingMode
    trigger: str        # what caused the transition (human-readable)
    phase: str          # which experiment phase label, e.g. "1→2"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class EscalationQuery(BaseModel):
    """Structured query presented to the human manager."""

    conflict_summary: str
    violated_constraints: list[str]
    demand_increase_pct: float
    options: list[RemediationOption]
    assumption_drifts: list[str] = Field(
        default_factory=list,
        description=(
            "Attribution strings linking constraint violations back to the "
            "specific environmental assumptions that drifted beyond their "
            "contracted tolerance bounds.  Populated by AssumptionTracker "
            "when drift is detected prior to escalation."
        ),
    )


class RemediationOption(BaseModel):
    """One option in the escalation menu."""

    label: str  # e.g., "Option A"
    description: str
    relaxes_constraint: str | None = None
    new_limit: float | None = None
    expected_deadline_gap_pct: float = 0.0
    preserves_human_constraints: bool = True


class ManagerDecision(BaseModel):
    """The manager's choice."""

    selected_option: str  # label
    rationale: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )


class AdaptationLevel(str, Enum):
    MICRO = "micro"  # Parameter retune
    MESO = "meso"  # Policy switch
    MACRO = "macro"  # Full replan


class AuditEntry(BaseModel):
    """Single entry in the append-only audit trail."""

    phase: int
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    layer: str
    action: str
    contract_state: str
    working_mode: str = ""
    details: dict = Field(default_factory=dict)


# Rebuild EscalationQuery now that RemediationOption is defined
EscalationQuery.model_rebuild()
