"""Schedule and task allocation models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Schedule(BaseModel):
    """A candidate production schedule parameterizing the cell."""

    name: str = "S1"
    source: str = Field(
        default="deterministic",
        description="Candidate origin: 'llm' or 'deterministic'",
    )
    r1_speed_fraction: float = Field(
        ge=0.0, le=1.0, description="R1 speed as fraction of max (0-1)"
    )
    r2_speed_fraction: float = Field(
        ge=0.0, le=1.0, description="R2 speed as fraction of max (0-1)"
    )
    human_cycle_rate_multiplier: float = Field(
        ge=0.5, le=2.0, description="Human pace multiplier (1.0 = nominal)"
    )
    buffer_time_s: float = Field(
        ge=0.0, description="Buffer time between operations (seconds)"
    )
    demand_target_uph: float = Field(
        ge=0.0, description="Target throughput (units per hour)"
    )

    @property
    def is_aggressive(self) -> bool:
        return (
            self.r2_speed_fraction > 0.9
            or self.human_cycle_rate_multiplier > 1.1
        )


class TaskAllocation(BaseModel):
    """How tasks are distributed across resources."""

    r1_tasks: list[str] = Field(default_factory=lambda: ["pick_and_place"])
    r2_tasks: list[str] = Field(default_factory=lambda: ["screwdriving"])
    human_tasks: list[str] = Field(
        default_factory=lambda: ["insertion", "inspection"]
    )


# ---------------------------------------------------------------------------
# Factory-level schedule models (multi-cell extension)
# ---------------------------------------------------------------------------


class CellSchedule(Schedule):
    """Schedule for one cell in a multi-cell factory."""

    cell_id: str = Field(default="A", description="Cell identifier")
    assigned_operator: str = Field(
        default="H1", description="Operator assigned to this cell"
    )
    assigned_variants: list[str] = Field(
        default_factory=lambda: ["V_A", "V_B"],
        description="Product variants this cell handles",
    )


class FactorySchedule(BaseModel):
    """Multi-cell production plan spanning the whole factory."""

    name: str = "FS1"
    source: str = Field(
        default="deterministic",
        description="Candidate origin: 'llm' or 'deterministic'",
    )
    cell_schedules: dict[str, CellSchedule] = Field(
        default_factory=dict,
        description="cell_id -> CellSchedule",
    )
    variant_routing: dict[str, list[str]] = Field(
        default_factory=dict,
        description="variant -> ordered list of cell_ids (e.g. V_A -> [A, B])",
    )
    operator_assignments: dict[str, str] = Field(
        default_factory=dict,
        description="operator_id -> cell_id",
    )
    agv_priority: str = Field(
        default="balanced",
        description="AGV dispatch priority: balanced | cellA_first | cellB_first",
    )
