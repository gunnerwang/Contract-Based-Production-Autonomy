"""Compile every encoded obligation into a fail-closed software guard."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pydantic import BaseModel
from cbpa.models.contract import OutcomeContract
from cbpa.layer3_verification.constraint_checker import ConstraintChecker, _METRIC_MAP, _OPS

logger = logging.getLogger(__name__)


class GuardEvent(BaseModel):
    """Observed guard failure or warning, not proof of a physical intervention."""
    guard_name: str
    metric_name: str
    metric_value: float | None
    threshold: float
    action: str
    enforced: bool = False
    detail: str = ""


class RuntimeGuardViolation(RuntimeError):
    """Abort software execution on a hard violation or unevaluable reading."""
    def __init__(self, events: list[GuardEvent]):
        self.events = events
        super().__init__("; ".join(e.detail for e in events if e.action == "escalate"))


@dataclass
class RuntimeGuard:
    """A threshold assertion with the same semantics as pre-deployment checks."""
    name: str
    metric_field: str
    operator: str
    threshold: float
    action_on_violation: str = "escalate"
    constraint_name: str | None = None
    factory: bool = False

    def check(self, value: float) -> bool:
        return bool(self.operator in _OPS
                    and ConstraintChecker._finite(value)
                    and ConstraintChecker._finite(self.threshold)
                    and _OPS[self.operator](value, self.threshold))

    def __str__(self) -> str:
        return f"GUARD: {self.name} {self.operator} {self.threshold} → {self.action_on_violation}"


class GuardEnforcer:
    """Check supplied observations without modifying them; raise on hard failure.

    The caller must propagate RuntimeGuardViolation to stop further software
    execution. This does not establish that an external machine has stopped.
    ``cyber_risk_level`` is an explicit model input for the illustrative cyber
    proxy, not a sensor reading. Without it that obligation is unevaluable.
    """
    def __init__(self, guards: list[RuntimeGuard], *, cyber_risk_level: float | None = None):
        self.guards = guards
        self.checker = ConstraintChecker(cyber_risk_level=cyber_risk_level)
        self.enforcement_log: list[GuardEvent] = []

    def check_and_enforce(self, metrics) -> list[GuardEvent]:
        events = []
        for guard in self.guards:
            if guard.constraint_name is not None:
                value = self.checker._actual(guard.constraint_name, metrics, guard.factory)
            elif hasattr(metrics, "model_fields_set") and guard.metric_field not in metrics.model_fields_set:
                value = None
            else:
                value = getattr(metrics, guard.metric_field, None)
            if guard.check(value):
                continue
            measurable = ConstraintChecker._finite(value)
            # Missing warning observations also require escalation. Never clamp
            # a reported measurement and present that as an actuator action.
            action = "warn" if measurable and guard.action_on_violation == "warn" else "escalate"
            detail = (f"{guard.metric_field}={value} violates {guard.operator} {guard.threshold}"
                      if measurable else f"{guard.metric_field}: missing or non-finite observation")
            event = GuardEvent(guard_name=guard.name, metric_name=guard.metric_field,
                               metric_value=float(value) if measurable else None,
                               threshold=guard.threshold, action=action,
                               detail=detail + ("; warning only" if action == "warn" else "; software execution aborted"))
            logger.info("Guard fired: %s", event.detail)
            events.append(event)
        self.enforcement_log.extend(events)
        if any(e.action == "escalate" for e in events):
            raise RuntimeGuardViolation(events)
        return events


class GuardSynthesizer:
    """Compile all hard constraints; reject contracts that cannot be monitored."""
    def synthesize(self, contract: OutcomeContract, *, factory: bool = False) -> list[RuntimeGuard]:
        errors = ConstraintChecker.contract_errors(contract, factory=factory)
        if errors:
            raise ValueError("Cannot synthesize runtime guards: " + "; ".join(errors))
        guards = []
        for c in contract.hard_constraints:
            if c.name in {"CyberRiskLevel", "cyber_risk"}:
                field = "cyber_risk_level"
            elif factory and c.name.startswith("FatigueIndex_"):
                field = "operator_fatigue." + c.name.split("_", 1)[1]
            else:
                field = (ConstraintChecker._FACTORY_METRIC_MAP if factory else _METRIC_MAP)[c.name]
                if c.name == "AGVTransferTime":
                    field = "agv_utilization * 60"
            guards.append(RuntimeGuard(c.name, field, c.operator, c.limit,
                                       constraint_name=c.name, factory=factory))
            if field in {"noise_db", "factory_noise_db"} and c.operator in {"<=", "<"} and c.limit > 0:
                guards.append(RuntimeGuard(c.name + "_Warning", field, "<=", c.limit * .9375,
                                           action_on_violation="warn", constraint_name=c.name, factory=factory))
        return guards
