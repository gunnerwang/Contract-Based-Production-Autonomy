"""Scenario configuration for the manufacturing cell and factory."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CellConfig:
    """Physical parameters of the collaborative assembly cell."""

    # Resource cycle times at nominal speed (seconds per unit)
    r1_cycle_time_s: float = 60.0  # R1 pick-and-place
    r2_cycle_time_s: float = 55.0  # R2 screwdriving
    human_insertion_time_s: float = 70.0  # H insertion (bottleneck)
    human_inspection_time_s: float = 15.0  # H inspection

    # Speed ranges
    r1_max_speed_mps: float = 1.5
    r2_max_speed_mps: float = 1.5

    # Shift parameters
    shift_hours: float = 8.0
    demand_base_uph: float = 52.0  # Base demand (units/h)

    # Multi-cell identifiers (backward-compat: defaults match single-cell)
    cell_id: str = "A"
    cell_type: str = "assembly"  # "assembly" or "test_pack"
    supported_variants: list[str] = field(
        default_factory=lambda: ["V_A", "V_B"]
    )


@dataclass
class OperatorProfile:
    """Profile of a human operator with certifications and fatigue limits."""

    id: str = "H1"
    name: str = "Operator 1"
    certifications: list[str] = field(
        default_factory=lambda: ["assembly", "inspection"]
    )
    max_fatigue_limit: float = 0.4
    fatigue_recovery_rate: float = 0.05  # per break-hour
    current_fatigue: float = 0.0
    shift_hours_worked: float = 0.0


@dataclass
class FactoryConfig:
    """Multi-cell factory configuration."""

    cells: dict[str, CellConfig] = field(default_factory=lambda: {
        "A": CellConfig(
            cell_id="A", cell_type="assembly",
            supported_variants=["V_A", "V_B", "V_C"],
        ),
        "B": CellConfig(
            cell_id="B", cell_type="test_pack",
            r1_cycle_time_s=50.0,  # R3 functional testing (faster)
            r2_cycle_time_s=45.0,  # R4 packaging
            human_insertion_time_s=60.0,  # H2 labeling + QC
            human_inspection_time_s=20.0,  # More thorough inspection
            demand_base_uph=48.0,
            supported_variants=["V_A", "V_B", "V_C"],
        ),
    })
    operators: list[OperatorProfile] = field(default_factory=lambda: [
        OperatorProfile(
            id="H1", name="Operator 1",
            certifications=["assembly", "inspection"],
            max_fatigue_limit=0.4,
        ),
        OperatorProfile(
            id="H2", name="Operator 2",
            certifications=["testing", "inspection", "packaging"],
            max_fatigue_limit=0.35,
        ),
        OperatorProfile(
            id="H3", name="Backup Operator",
            certifications=["inspection"],
            max_fatigue_limit=0.45,
        ),
    ])
    agv_transfer_time_s: float = 45.0
    buffer_capacity: int = 4
    product_variants: list[str] = field(
        default_factory=lambda: ["V_A", "V_B", "V_C"]
    )


@dataclass
class ScheduleParams:
    """Parameters for the three candidate schedules S1, S2, S3."""

    # S1: moderate, comfortable baseline
    s1_r1_speed: float = 0.65
    s1_r2_speed: float = 0.60
    s1_human_rate: float = 1.0
    s1_buffer_s: float = 3.0

    # S2: aggressive, violates constraints
    s2_r1_speed: float = 0.90
    s2_r2_speed: float = 1.0
    s2_human_rate: float = 1.15
    s2_buffer_s: float = 1.0

    # S3: optimized within constraints
    s3_r1_speed: float = 0.72
    s3_r2_speed: float = 0.70
    s3_human_rate: float = 1.05
    s3_buffer_s: float = 2.0


@dataclass
class ScenarioConfig:
    """Full scenario configuration."""

    cell: CellConfig = field(default_factory=CellConfig)
    schedules: ScheduleParams = field(default_factory=ScheduleParams)

    # Ablation: withhold deterministic anchors (LLM-only candidate pool, H2 condition)
    llm_only: bool = False

    # Add a certified constrained-optimiser schedule to the anchor pool of the initial,
    # stabilisation, and next-shift phases (the recommendation of Section 5.5)
    optimiser_anchor: bool = False

    # Demand spike
    demand_spike_pct: float = 20.0
    spike_time_hours: float = 2.0  # 2h into shift (10:00 AM if shift starts 8:00)

    # V/R weights from the paper (Section 6)
    vr_value_weights: list[float] = field(
        default_factory=lambda: [0.50, 0.30, 0.20]  # throughput, quality, flexibility
    )
    vr_resource_weights: list[float] = field(
        default_factory=lambda: [0.35, 0.25, 0.20, 0.20]  # energy, downtime, labor, cost
    )

    # Tolerance for Table 3 reproduction
    table3_tolerance_pct: float = 2.0

    # Phase 6 (macro-adaptation demonstration) — disabled by default
    # so existing 5-phase tests are unaffected.
    enable_macro_phase: bool = False
    macro_r1_degradation_fraction: float = 0.50  # R1 drops to 50% of max speed
    macro_adapt_threshold: float = 0.90  # p_feasible threshold for macro trigger

    # Multi-shift learning curve configuration.
    # n_shifts=1 (default) runs the standard single-shift experiment.
    # When n_shifts>1, the LearningStore persists across shifts so that
    # later shifts benefit from prior experience (cumulative operational
    # intelligence).
    n_shifts: int = 1
    shift_disturbances: list[dict] | None = None

    # When True, deployed schedules are executed through the SimPy
    # discrete-event simulation (8-hour shift) instead of instant
    # analytical evaluation.  Guards fire in real simulated time.
    use_simulation: bool = False

    # When True, deployed schedules are executed through the integrated
    # stack: Isaac Sim (physics), BaSyx AAS (digital twin), and OPC-UA
    # (real-time data).  Mutually exclusive with use_simulation.
    use_integrated: bool = False
    isaac_sim_url: str = "http://localhost:8211"
    basyx_registry_url: str = "http://localhost:9082"
    basyx_aas_server_url: str = "http://localhost:9081"
    opcua_endpoint: str = "opc.tcp://localhost:4840/cbpa/"

    def __post_init__(self) -> None:
        if self.use_simulation and self.use_integrated:
            raise ValueError(
                "use_simulation and use_integrated are mutually exclusive. "
                "Set only one to True."
            )


@dataclass
class FactoryScenarioConfig(ScenarioConfig):
    """Extended scenario config for multi-cell factory experiments."""

    factory: FactoryConfig = field(default_factory=FactoryConfig)

    # Factory-level disturbance types (in addition to demand_spike_pct)
    enable_operator_absence: bool = True
    operator_absence_hour: float = 3.0  # H2 unavailable at hour 3
    absent_operator_id: str = "H2"

    enable_supply_delay: bool = False
    supply_delay_variant: str = "V_C"
    supply_delay_hours: float = 2.0  # V_C materials delayed 2h

    # Factory-level constraint coupling
    factory_noise_limit_db: float = 82.0  # combined noise across cells
    cell_balance_loss_max_pct: float = 25.0  # max throughput imbalance

    # Default variant routing: variant -> [cell_id sequence]
    default_variant_routing: dict[str, list[str]] = field(
        default_factory=lambda: {
            "V_A": ["A", "B"],  # assembly in A, then test & pack in B
            "V_B": ["A", "B"],
            "V_C": ["A", "B"],
        }
    )

    # Default operator assignments: operator_id -> cell_id
    default_operator_assignments: dict[str, str] = field(
        default_factory=lambda: {"H1": "A", "H2": "B"}
    )
