"""Runtime guard synthesis and enforcement from hard constraints K."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import BaseModel

from cbpa.models.contract import OutcomeContract

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guard event model
# ---------------------------------------------------------------------------


class GuardEvent(BaseModel):
    """Record of a single guard enforcement action at runtime."""

    guard_name: str
    metric_name: str
    metric_value: float
    threshold: float
    action: str  # "clamp", "escalate", "warn"
    enforced: bool = True
    detail: str = ""


# ---------------------------------------------------------------------------
# Runtime guard
# ---------------------------------------------------------------------------


@dataclass
class RuntimeGuard:
    """A runtime threshold assertion."""

    name: str
    metric_field: str
    operator: str
    threshold: float
    action_on_violation: str = "escalate"

    def check(self, value: float) -> bool:
        if self.operator == "<=":
            return value <= self.threshold
        elif self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "<":
            return value < self.threshold
        elif self.operator == ">":
            return value > self.threshold
        return True

    def __str__(self) -> str:
        return f"GUARD: {self.name} {self.operator} {self.threshold} → {self.action_on_violation}"


# ---------------------------------------------------------------------------
# Guard enforcer — the runtime enforcement engine
# ---------------------------------------------------------------------------


class GuardEnforcer:
    """Checks all guards against current metrics and enforces actions.

    This is the core of CBPA Layer 4: contract-synthesized guards that
    actually override unsafe states at runtime, not just observe them.
    """

    def __init__(self, guards: list[RuntimeGuard]) -> None:
        self.guards = guards
        self.enforcement_log: list[GuardEvent] = []

    def check_and_enforce(
        self,
        metrics: "SimulationMetrics",  # noqa: F821 — forward ref
    ) -> list[GuardEvent]:
        """Check all guards against *metrics*.

        Returns a list of :class:`GuardEvent` instances for every guard
        that fired (i.e. the constraint was violated).  Events whose
        ``action`` is ``"clamp"`` indicate the metric was capped back to
        the threshold on the metrics object in-place.
        """
        from cbpa.models.metrics import SimulationMetrics  # local to avoid circular

        events: list[GuardEvent] = []
        for guard in self.guards:
            value = getattr(metrics, guard.metric_field, None)
            if value is None:
                continue

            if guard.check(value):
                # Guard satisfied — no action required.
                continue

            # --- Guard violated ---
            action = guard.action_on_violation

            if action == "clamp":
                # Override the metric value back to the safe threshold.
                setattr(metrics, guard.metric_field, guard.threshold)
                event = GuardEvent(
                    guard_name=guard.name,
                    metric_name=guard.metric_field,
                    metric_value=value,
                    threshold=guard.threshold,
                    action="clamp",
                    enforced=True,
                    detail=(
                        f"{guard.metric_field} clamped from {value} "
                        f"to {guard.threshold}"
                    ),
                )
            elif action == "escalate":
                event = GuardEvent(
                    guard_name=guard.name,
                    metric_name=guard.metric_field,
                    metric_value=value,
                    threshold=guard.threshold,
                    action="escalate",
                    enforced=True,
                    detail=(
                        f"{guard.metric_field}={value} violates "
                        f"{guard.operator} {guard.threshold}; escalating"
                    ),
                )
            else:
                # "warn" — log but do not modify
                event = GuardEvent(
                    guard_name=guard.name,
                    metric_name=guard.metric_field,
                    metric_value=value,
                    threshold=guard.threshold,
                    action="warn",
                    enforced=False,
                    detail=(
                        f"{guard.metric_field}={value} approaching/exceeding "
                        f"{guard.threshold}; warning only"
                    ),
                )

            logger.info("Guard fired: %s", event.detail)
            events.append(event)

        self.enforcement_log.extend(events)
        return events


# ---------------------------------------------------------------------------
# Guard synthesis
# ---------------------------------------------------------------------------

# Mapping from contract constraint names to SimulationMetrics field names.
# Includes aliases so LLM-generated constraint names still resolve.
_FIELD_MAP = {
    "FatigueIndex": "fatigue_index",
    "fatigue_index": "fatigue_index",
    "fatigue": "fatigue_index",
    "operator_fatigue": "fatigue_index",
    "Noise": "noise_db",
    "noise": "noise_db",
    "noise_db": "noise_db",
    "noise_level": "noise_db",
}

# Constraint-specific enforcement actions.  Fatigue and defects cannot be
# mechanically clamped (they reflect human/process state), so they must
# escalate.  Noise gets a two-tier treatment: warn first, escalate at the
# hard limit.  Energy can be reduced by clamping robot speeds.
_ACTION_MAP: dict[str, str] = {
    "FatigueIndex": "escalate",
    "fatigue_index": "escalate",
    "fatigue": "escalate",
    "operator_fatigue": "escalate",
    "Noise": "escalate",
    "noise": "escalate",
    "noise_db": "escalate",
    "noise_level": "escalate",
}


class GuardSynthesizer:
    """Compiles hard constraints K into runtime guard assertions."""

    def synthesize(self, contract: OutcomeContract) -> list[RuntimeGuard]:
        guards: list[RuntimeGuard] = []
        for c in contract.hard_constraints:
            field_name = _FIELD_MAP.get(c.name)
            if field_name is None:
                continue

            action = _ACTION_MAP.get(c.name, "escalate")

            # For Noise, add a warning guard at 75 dB in addition to the
            # hard-limit escalation guard at the contract threshold (80 dB).
            if c.name == "Noise":
                guards.append(
                    RuntimeGuard(
                        name=f"{c.name}_Warning",
                        metric_field=field_name,
                        operator="<=",
                        threshold=c.limit * 0.9375,  # 75 dB for 80 dB limit
                        action_on_violation="warn",
                    )
                )

            guards.append(
                RuntimeGuard(
                    name=c.name,
                    metric_field=field_name,
                    operator=c.operator,
                    threshold=c.limit,
                    action_on_violation=action,
                )
            )

        # Energy guard — clamp robot speeds when energy budget is exceeded.
        # 500 kWh is a reasonable cap for an 8-hour shift (S1 ~398, S2 ~462).
        guards.append(
            RuntimeGuard(
                name="Energy_Cap",
                metric_field="energy_kwh",
                operator="<=",
                threshold=500.0,
                action_on_violation="clamp",
            )
        )

        # Defect rate guard — escalate (cannot be mechanically clamped).
        guards.append(
            RuntimeGuard(
                name="DefectRate_Cap",
                metric_field="defect_rate",
                operator="<=",
                threshold=0.01,
                action_on_violation="escalate",
            )
        )

        # Fail-safe speed caps (clampable).
        guards.append(
            RuntimeGuard(
                name="R1_SpeedCap",
                metric_field="r1_speed_fraction",
                operator="<=",
                threshold=1.0,
                action_on_violation="clamp",
            )
        )
        guards.append(
            RuntimeGuard(
                name="R2_SpeedCap",
                metric_field="r2_speed_fraction",
                operator="<=",
                threshold=1.0,
                action_on_violation="clamp",
            )
        )
        return guards
