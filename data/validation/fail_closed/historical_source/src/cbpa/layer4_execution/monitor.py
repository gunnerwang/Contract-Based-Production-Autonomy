"""Contract-centric KPI monitoring for runtime execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cbpa.layer3_verification.guard_synthesizer import GuardEnforcer, GuardEvent
from cbpa.layer4_execution.assumption_tracker import AssumptionDrift, AssumptionTracker
from cbpa.models.contract import OutcomeContract
from cbpa.models.metrics import SimulationMetrics

logger = logging.getLogger(__name__)


@dataclass
class MonitorAlert:
    """An alert raised when a KPI is approaching or violating a constraint."""

    constraint_name: str
    current_value: float
    limit: float
    margin_pct: float
    severity: str  # "warning", "violation"
    message: str


class ContractMonitor:
    """Monitors runtime metrics against contract constraints.

    Raises warnings when metrics approach constraint limits and
    violations when limits are breached.  When a :class:`GuardEnforcer`
    is attached, guards from the feasibility certificate are actively
    checked alongside KPI monitoring so that unsafe states are
    overridden at runtime (CBPA Layer 4 enforcement).
    """

    def __init__(
        self,
        contract: OutcomeContract,
        warning_margin_pct: float = 10.0,
        guard_enforcer: GuardEnforcer | None = None,
    ):
        self.contract = contract
        self.warning_margin_pct = warning_margin_pct
        self.alerts: list[MonitorAlert] = []
        self.guard_enforcer = guard_enforcer
        self.guard_events: list[GuardEvent] = []
        self.assumption_tracker: AssumptionTracker | None = (
            AssumptionTracker(contract.typed_assumptions)
            if contract.typed_assumptions
            else None
        )

    def check(self, metrics: SimulationMetrics) -> list[MonitorAlert]:
        """Check current metrics against contract constraints.

        If a :class:`GuardEnforcer` is attached, guards are checked
        first so that clamping actions modify *metrics* before the KPI
        margin calculations run.
        """
        # --- Guard enforcement (runs before KPI margin checks) ---
        if self.guard_enforcer is not None:
            events = self.guard_enforcer.check_and_enforce(metrics)
            self.guard_events.extend(events)
            for ev in events:
                logger.info(
                    "ContractMonitor guard event: %s [%s]",
                    ev.guard_name,
                    ev.action,
                )

        # --- KPI monitoring ---
        alerts: list[MonitorAlert] = []

        _field_map = {
            "FatigueIndex": metrics.fatigue_index,
            "Noise": metrics.noise_db,
        }

        for constraint in self.contract.hard_constraints:
            actual = _field_map.get(constraint.name)
            if actual is None:
                continue

            margin = constraint.limit - actual
            margin_pct = (margin / constraint.limit) * 100 if constraint.limit else 0

            if margin < 0:
                alerts.append(
                    MonitorAlert(
                        constraint_name=constraint.name,
                        current_value=actual,
                        limit=constraint.limit,
                        margin_pct=margin_pct,
                        severity="violation",
                        message=f"{constraint.name} = {actual} VIOLATES limit {constraint.limit}",
                    )
                )
            elif margin_pct < self.warning_margin_pct:
                alerts.append(
                    MonitorAlert(
                        constraint_name=constraint.name,
                        current_value=actual,
                        limit=constraint.limit,
                        margin_pct=margin_pct,
                        severity="warning",
                        message=f"{constraint.name} = {actual} approaching limit {constraint.limit} ({margin_pct:.1f}% margin)",
                    )
                )

        self.alerts.extend(alerts)
        return alerts

    def check_deadline(
        self,
        current_throughput_uph: float,
        demand_uph: float,
    ) -> MonitorAlert | None:
        """Check if current throughput will meet demand."""
        if demand_uph <= 0:
            return None
        gap_pct = (demand_uph - current_throughput_uph) / demand_uph * 100
        if gap_pct > 5.0:
            alert = MonitorAlert(
                constraint_name="DeadlineGap",
                current_value=current_throughput_uph,
                limit=demand_uph,
                margin_pct=-gap_pct,
                severity="warning",
                message=f"Throughput {current_throughput_uph:.1f} uph will miss demand {demand_uph:.1f} uph by {gap_pct:.1f}%",
            )
            self.alerts.append(alert)
            return alert
        return None

    def check_assumptions(
        self,
        observed: dict[str, float],
    ) -> list[AssumptionDrift]:
        """Update assumptions with observed values and return any violations.

        Parameters
        ----------
        observed:
            Mapping of assumption name to its current measured value,
            e.g. ``{"demand_base_uph": 62.4, "network_latency_ms": 22.0}``.

        Returns
        -------
        list[AssumptionDrift]
            One entry per assumption that drifted beyond its tolerance.
        """
        if self.assumption_tracker is None:
            return []

        violations: list[AssumptionDrift] = []
        for name, value in observed.items():
            drift = self.assumption_tracker.update(name, value)
            if drift is not None:
                violations.append(drift)
        return violations
