"""Simulation metrics and scoring results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StochasticFeasibilityReport(BaseModel):
    """Summary statistics from a Monte Carlo feasibility sweep."""

    p_feasible: float = Field(
        description="Fraction of Monte Carlo samples passing all constraints"
    )
    constraint_violation_probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Per-constraint probability of violation across samples",
    )
    worst_case_margins: dict[str, float] = Field(
        default_factory=dict,
        description="5th-percentile margin (limit - actual) per constraint",
    )
    sensitivity_ranking: list[tuple[str, str, float]] = Field(
        default_factory=list,
        description=(
            "Ranked list of (parameter, constraint, sensitivity_score) "
            "showing which input perturbation most drives each constraint"
        ),
    )
    n_samples: int = Field(description="Number of Monte Carlo samples used")
    seed: int = Field(description="RNG seed used; guarantees reproducibility")
    certified: bool = Field(
        default=False,
        description="True if p_feasible >= confidence_threshold",
    )
    confidence_threshold: float = Field(
        default=0.95,
        description="Minimum P(feasible) required for certification",
    )


class SimulationMetrics(BaseModel):
    """KPI outputs from a simulation run."""

    throughput_uph: float = Field(description="Units per hour")
    defect_rate: float = Field(description="Fraction defective")
    noise_db: float = Field(description="Noise level in dB")
    fatigue_index: float = Field(description="Cumulative fatigue index (0-1)")
    energy_kwh: float = Field(description="Energy per shift in kWh")
    deadline_gap_pct: float = Field(
        description="Percentage shortfall vs demand target"
    )
    units_produced: int = Field(default=0, description="Total units in shift")
    shift_hours: float = Field(default=8.0)


class VRScoreResult(BaseModel):
    """Value/Resource score breakdown."""

    value_numerator: float
    resource_denominator: float
    vr_score: float
    component_breakdown: dict[str, float] = Field(default_factory=dict)


class FeasibilityReport(BaseModel):
    """Result of constraint checking against K."""

    is_feasible: bool
    violations: list[str] = Field(default_factory=list)
    constraint_margins: dict[str, float] = Field(
        default_factory=dict,
        description="margin = limit - actual (positive = OK)",
    )


class FactoryMetrics(BaseModel):
    """Aggregated metrics across a multi-cell factory."""

    cell_metrics: dict[str, SimulationMetrics] = Field(
        default_factory=dict,
        description="Per-cell metrics keyed by cell_id",
    )
    total_throughput_uph: float = Field(
        default=0.0, description="Combined throughput across all cells"
    )
    cell_balance_loss_pct: float = Field(
        default=0.0,
        description="Throughput imbalance: (max - min) / max * 100",
    )
    factory_noise_db: float = Field(
        default=0.0,
        description="Combined noise (logarithmic sum across cells)",
    )
    agv_utilization: float = Field(
        default=0.0,
        description="AGV utilization fraction (0-1)",
    )
    operator_fatigue: dict[str, float] = Field(
        default_factory=dict,
        description="Per-operator fatigue index: operator_id -> fatigue",
    )
    factory_energy_kwh: float = Field(
        default=0.0,
        description="Total energy consumption across all cells",
    )
    factory_defect_rate: float = Field(
        default=0.0,
        description="Weighted average defect rate across cells",
    )


class ParetoFrontResult(BaseModel):
    """Result of multi-objective Pareto filtering over a candidate set.

    Records which schedules form the non-dominated front, which were
    dominated, which was ultimately selected, and the per-candidate
    objective scores used to make that determination.
    """

    candidates: list[str] = Field(
        description="All candidate schedule names evaluated"
    )
    non_dominated: list[str] = Field(
        description="Pareto-optimal (non-dominated) schedule names"
    )
    dominated: list[str] = Field(
        description="Dominated schedule names"
    )
    selected: str = Field(
        description="Name of the schedule chosen for deployment"
    )
    selection_rationale: str = Field(
        description="Human-readable explanation of the selection logic"
    )
    scores: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "Per-schedule objective scores: "
            "schedule_name -> {vr_score, constraint_margin, feasible, ...}"
        ),
    )
    candidate_sources: dict[str, str] = Field(
        default_factory=dict,
        description="schedule_name -> 'llm' or 'deterministic'",
    )
