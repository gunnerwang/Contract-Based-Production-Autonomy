"""Outcome contract data model: C^out = <KPI, K, P, X, A>."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Assumption(BaseModel):
    """A typed environmental assumption with tolerance bounds.

    Assumptions are the world-model assertions the contract was designed
    under.  When a measured value drifts outside ``tolerance_pct`` of
    ``expected_value`` the assumption is flagged as violated and can be
    linked to downstream constraint breaches through ``AssumptionTracker``.
    """

    name: str
    """Programmatic identifier, e.g. ``"demand_base_uph"``."""

    description: str
    """Human-readable description of what this assumption captures."""

    expected_value: float
    """The value assumed when the contract was authored."""

    tolerance_pct: float = 10.0
    """Maximum acceptable percentage drift from ``expected_value``."""

    unit: str = ""
    """Physical unit string, e.g. ``"uph"``, ``"m/s"``, ``"ms"``."""

    current_value: float | None = None
    """Latest measured value; ``None`` until first ``update`` call."""

    violated: bool = False
    """``True`` once drift exceeds ``tolerance_pct``."""

    drift_pct: float | None = None
    """Signed percentage drift ``(current - expected) / expected * 100``
    when last checked; ``None`` until first ``update`` call."""

    def check(self, observed: float) -> bool:
        """Check whether *observed* is within tolerance of the expected value.

        Updates ``current_value``, ``drift_pct``, and ``violated`` in place.

        Returns ``True`` if the assumption still holds (within tolerance),
        ``False`` if violated.
        """
        self.current_value = observed
        if self.expected_value == 0.0:
            self.drift_pct = 0.0
        else:
            self.drift_pct = round(
                (observed - self.expected_value) / abs(self.expected_value) * 100.0,
                1,
            )
        self.violated = abs(self.drift_pct) > self.tolerance_pct
        return not self.violated


class AmbiguityFlag(BaseModel):
    """Records a detected ambiguity and its resolution during intent elicitation.

    An ambiguity arises when a draft contract contains a KPI objective or
    constraint whose meaning is under-specified enough that two reasonable
    interpretations would produce materially different schedules.  The
    propose-and-refine loop surfaces each such flag to the manager (or, in
    deterministic mode, resolves them with pre-set answers) before the
    contract is finalised.
    """

    field: str
    """The contract field or KPI/constraint name that is ambiguous,
    e.g. ``"energy_kwh_cap"`` or ``"shift_hours"``."""

    question: str
    """The trade-off question posed to the manager,
    e.g. ``"Maximising throughput may increase energy by ~40%. Cap at 3.0 kWh?"``"""

    default_resolution: str
    """The assumption the system would apply if the manager gives no answer."""

    resolution: str = ""
    """The manager's actual answer (empty until resolved)."""

    resolved: bool = False
    """``True`` once ``resolution`` has been filled in."""


class KPIDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    UPPER_BOUND = "upper_bound"
    LOWER_BOUND = "lower_bound"


class KPITarget(BaseModel):
    """A single KPI objective."""

    name: str
    direction: KPIDirection
    threshold: float | None = None
    unit: str = ""


class HardConstraint(BaseModel):
    """A non-negotiable constraint in K."""

    name: str
    operator: str = "<="  # "<=", ">=", "==", "<", ">"
    limit: float
    unit: str = ""


class PriorityLevel(str, Enum):
    HUMAN_WELLBEING = "HumanWellbeing"
    SAFETY = "Safety"
    QUALITY = "Quality"
    THROUGHPUT = "Throughput"
    COST = "Cost"
    FLEXIBILITY = "Flexibility"


class OutcomeContract(BaseModel):
    """
    C^out = <KPI, K, P, X, A>

    The formal outcome contract that captures manager intent.
    """

    name: str = "C1"
    kpi_targets: list[KPITarget] = Field(default_factory=list)
    hard_constraints: list[HardConstraint] = Field(
        default_factory=list, description="K: non-negotiable constraints"
    )
    priority_order: list[PriorityLevel] = Field(
        default_factory=list, description="P: stakeholder priority ordering"
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="X: operational context"
    )
    assumptions: dict[str, Any] = Field(
        default_factory=dict, description="A: environmental assumptions (legacy dict)"
    )
    typed_assumptions: list[Assumption] = Field(
        default_factory=list,
        description=(
            "A: typed environmental assumptions with tolerance bounds. "
            "Checked at runtime by AssumptionTracker to detect drift and "
            "attribute constraint breaches."
        ),
    )
    ambiguities: list[AmbiguityFlag] = Field(
        default_factory=list,
        description=(
            "Ambiguity flags surfaced and resolved during the "
            "propose-and-refine elicitation loop."
        ),
    )

    def get_constraint(self, name: str) -> HardConstraint | None:
        for c in self.hard_constraints:
            if c.name == name:
                return c
        return None


def reinsert_typed_assumptions(
    contract: OutcomeContract, baseline: OutcomeContract, logger=None
) -> OutcomeContract:
    """Bind an elicited contract to the plant's typed assumptions (fail-closed).

    The LLM elicitation agent returns assumptions as free text in
    ``contract.assumptions``; the Layer-4 ``AssumptionTracker`` checks only
    ``typed_assumptions``.  Any plant assumption the draft does not carry as a
    typed entry is re-inserted from *baseline* (the plant model), so that
    drift detection and breach attribution cannot silently fall open when the
    contract was written in the LLM's own vocabulary.  Typed assumptions the
    draft already carries are kept unchanged.
    """
    present = {a.name for a in contract.typed_assumptions}
    added = []
    for a in baseline.typed_assumptions:
        if a.name not in present:
            added.append(a.model_copy(deep=True))
    if added and logger is not None:
        logger.warning(
            "LLM contract omitted typed assumption(s) %s; re-inserted from the plant model",
            [a.name for a in added],
        )
    if not added:
        return contract
    return contract.model_copy(
        update={"typed_assumptions": list(contract.typed_assumptions) + added}
    )


def make_c1() -> OutcomeContract:
    """Create the paper's C1 contract."""
    return OutcomeContract(
        name="C1",
        kpi_targets=[
            KPITarget(
                name="Throughput",
                direction=KPIDirection.MAXIMIZE,
                unit="units/h",
            ),
            KPITarget(
                name="DefectRate",
                direction=KPIDirection.UPPER_BOUND,
                threshold=0.01,
            ),
            KPITarget(
                name="ChangeoverTime",
                direction=KPIDirection.MINIMIZE,
                unit="min",
            ),
        ],
        hard_constraints=[
            HardConstraint(name="FatigueIndex", operator="<=", limit=0.4),
            HardConstraint(name="Noise", operator="<=", limit=80.0, unit="dB"),
            HardConstraint(
                name="CyberRiskLevel", operator="<=", limit=2.0
            ),  # Medium=2
        ],
        priority_order=[
            PriorityLevel.HUMAN_WELLBEING,
            PriorityLevel.SAFETY,
            PriorityLevel.QUALITY,
            PriorityLevel.THROUGHPUT,
        ],
        context={
            "shift_hours": 8,
            "mix": ["V_A", "V_B"],
            "demand_spike_risk": "High",
        },
        assumptions={
            "r1_max_speed_mps": 1.5,
            "network_latency_ms": 20,
        },
        typed_assumptions=[
            Assumption(
                name="demand_base_uph",
                description="Baseline throughput demand for this shift",
                expected_value=52.0,
                tolerance_pct=10.0,
                unit="uph",
            ),
            Assumption(
                name="r1_max_speed_mps",
                description="Maximum operational speed of robot R1",
                expected_value=1.5,
                tolerance_pct=5.0,
                unit="m/s",
            ),
            Assumption(
                name="r2_max_speed_mps",
                description="Maximum operational speed of robot R2",
                expected_value=1.2,
                tolerance_pct=5.0,
                unit="m/s",
            ),
            Assumption(
                name="network_latency_ms",
                description="Round-trip network latency for robot command signalling",
                expected_value=20.0,
                tolerance_pct=50.0,
                unit="ms",
            ),
            Assumption(
                name="ambient_temp_c",
                description="Ambient cell temperature affecting equipment calibration",
                expected_value=22.0,
                tolerance_pct=10.0,
                unit="°C",
            ),
        ],
    )


def make_c1_factory() -> OutcomeContract:
    """Create a factory-level C1 contract for multi-cell operation.

    Extends the single-cell C1 with per-operator fatigue constraints,
    combined factory noise, cell balance KPI, and operator assumptions.
    """
    return OutcomeContract(
        name="C1_factory",
        kpi_targets=[
            KPITarget(
                name="FactoryThroughput",
                direction=KPIDirection.MAXIMIZE,
                unit="units/h",
            ),
            KPITarget(
                name="DefectRate",
                direction=KPIDirection.UPPER_BOUND,
                threshold=0.01,
            ),
            KPITarget(
                name="CellBalanceLoss",
                direction=KPIDirection.UPPER_BOUND,
                threshold=25.0,
                unit="%",
            ),
            KPITarget(
                name="ChangeoverTime",
                direction=KPIDirection.MINIMIZE,
                unit="min",
            ),
        ],
        hard_constraints=[
            HardConstraint(
                name="FatigueIndex_H1", operator="<=", limit=0.4
            ),
            HardConstraint(
                name="FatigueIndex_H2", operator="<=", limit=0.35
            ),
            HardConstraint(
                name="FatigueIndex_H3", operator="<=", limit=0.4
            ),
            HardConstraint(
                name="FactoryNoise", operator="<=", limit=82.0, unit="dB"
            ),
            HardConstraint(
                name="CyberRiskLevel", operator="<=", limit=2.0
            ),
            HardConstraint(
                name="AGVTransferTime", operator="<=", limit=60.0, unit="s"
            ),
        ],
        priority_order=[
            PriorityLevel.HUMAN_WELLBEING,
            PriorityLevel.SAFETY,
            PriorityLevel.QUALITY,
            PriorityLevel.THROUGHPUT,
        ],
        context={
            "shift_hours": 8,
            "mix": ["V_A", "V_B", "V_C"],
            "cells": ["A", "B"],
            "demand_spike_risk": "High",
            "operator_pool": ["H1", "H2", "H3"],
        },
        assumptions={
            "r1_max_speed_mps": 1.5,
            "network_latency_ms": 20,
        },
        typed_assumptions=[
            Assumption(
                name="factory_demand_uph",
                description="Combined factory throughput demand",
                expected_value=100.0,
                tolerance_pct=10.0,
                unit="uph",
            ),
            Assumption(
                name="cellA_demand_uph",
                description="Cell A assembly throughput demand",
                expected_value=52.0,
                tolerance_pct=10.0,
                unit="uph",
            ),
            Assumption(
                name="cellB_demand_uph",
                description="Cell B test & pack throughput demand",
                expected_value=48.0,
                tolerance_pct=10.0,
                unit="uph",
            ),
            Assumption(
                name="h2_availability",
                description="Operator H2 available for full shift",
                expected_value=1.0,
                tolerance_pct=5.0,
                unit="boolean",
            ),
            Assumption(
                name="supply_vc_available",
                description="V_C raw materials available on time",
                expected_value=1.0,
                tolerance_pct=5.0,
                unit="boolean",
            ),
            Assumption(
                name="agv_transfer_time_s",
                description="AGV transfer time between cells",
                expected_value=45.0,
                tolerance_pct=30.0,
                unit="s",
            ),
        ],
    )


def make_c2() -> OutcomeContract:
    """Create C2: same human constraints, acknowledged throughput gap."""
    c2 = make_c1()
    c2.name = "C2"
    c2.context["demand_increase_pct"] = 20
    c2.context["deadline_gap_acknowledged"] = True
    return c2


def make_c3_factory() -> OutcomeContract:
    """Create C3_factory: next-shift contract revised from shift-close learning.

    The Legislator rewrites the contract at shift close, incorporating
    lessons learned this shift:
    - H2 absence risk is now a first-class assumption (tolerance tightened)
    - Cell B demand baseline lowered conservatively given observed bottleneck
    - V_B/V_C routing flexibility built in as a contract-level contingency
    - Human constraints are never relaxed

    This contract is pre-staged and ready to deploy at next shift start.
    """
    c3 = make_c1_factory()
    c3 = c3.model_copy(update={"name": "C3_factory"})
    c3.context["origin"] = "shift_close_learning"
    c3.context["h2_absence_risk"] = "elevated"
    c3.context["variant_routing_flexibility"] = True
    # Tighten H2 availability assumption tolerance — absence observed this shift
    for a in c3.typed_assumptions:
        if a.name == "h2_availability":
            a.tolerance_pct = 0.0   # no tolerance: plan for absence
            a.description = "H2 availability: plan conservatively (absence observed)"
        # Conservative Cell B baseline: reflects observed bottleneck under reduced staffing
        if a.name == "cellB_demand_uph":
            a.expected_value = 44.0
            a.description = "Cell B conservative demand baseline (learned from this shift)"
    return c3
