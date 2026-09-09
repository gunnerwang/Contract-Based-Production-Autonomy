"""Pydantic message models for integration bus / ROS / MES."""

from __future__ import annotations

from pydantic import BaseModel, Field


class KPIStreamMsg(BaseModel):
    """Real-time KPI reading pushed every cycle."""

    throughput_uph: float
    defect_rate: float
    noise_db: float
    fatigue_index: float
    energy_kwh: float
    deadline_gap_pct: float


class ConstraintStatusMsg(BaseModel):
    """Constraint compliance snapshot."""

    constraint_name: str
    current_value: float
    limit: float
    margin_pct: float
    status: str = "ok"  # "ok", "warning", "violation"


class ScheduleDeployMsg(BaseModel):
    """Command to deploy a schedule to the cell controller."""

    schedule_name: str
    r1_speed_fraction: float
    r2_speed_fraction: float
    human_cycle_rate_multiplier: float
    buffer_time_s: float
    demand_target_uph: float


class AlertMsg(BaseModel):
    """Alert notification from the monitoring system."""

    constraint_name: str
    severity: str
    message: str
    current_value: float
    limit: float


class PhaseTransitionMsg(BaseModel):
    """Notification that the experiment has entered a new phase."""

    from_phase: int
    to_phase: int
    description: str


class ProductionOrderMsg(BaseModel):
    """Production order from MES."""

    order_id: str
    product_variant: str = "V_A"
    quantity: int = 0
    demand_uph: float = 52.0
    priority: str = "normal"


class ScheduleUploadMsg(BaseModel):
    """Schedule upload acknowledgement from MES."""

    schedule_name: str
    accepted: bool = True
    mes_order_id: str = ""


class KPIReportMsg(BaseModel):
    """Aggregated KPI report sent to MES."""

    schedule_name: str
    throughput_uph: float
    defect_rate: float
    shift_hours: float = 8.0
    units_produced: int = 0


# ── BaSyx / AAS messages ─────────────────────────────────────────


class AASContractSyncMsg(BaseModel):
    """Contract synchronization to AAS submodels."""

    aas_id: str
    contract_name: str
    submodels_synced: list[str] = Field(default_factory=list)


class AASKPISubmodelMsg(BaseModel):
    """Live KPI update pushed to AAS KPI submodel."""

    throughput_uph: float
    defect_rate: float
    noise_db: float
    fatigue_index: float
    energy_kwh: float


# ── OPC-UA messages ───────────────────────────────────────────────


class OPCUANodeUpdateMsg(BaseModel):
    """Batch update of OPC-UA node values."""

    updated_nodes: list[str] = Field(default_factory=list)
    stub: bool = True


# ── Isaac Sim messages ────────────────────────────────────────────


class IsaacSceneCommandMsg(BaseModel):
    """Command sent to Isaac Sim scene."""

    command: str  # "deploy_schedule", "step", "verify_constraints"
    schedule_name: str = ""
    num_steps: int = 0


class IsaacSensorDataMsg(BaseModel):
    """Sensor readings from Isaac Sim."""

    robot_r1_speed_mps: float = 0.0
    robot_r2_speed_mps: float = 0.0
    noise_db: float = 0.0
    fatigue_estimate: float = 0.0


class IsaacVerificationResultMsg(BaseModel):
    """Constraint verification result from Isaac Sim physics."""

    constraint: str
    limit: float
    sim_value: float
    passed: bool
    margin_pct: float = 0.0
