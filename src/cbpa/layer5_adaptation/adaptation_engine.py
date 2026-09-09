"""Layer 5: Adaptation engine for micro/meso/macro corrections."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from cbpa.models.escalation import AdaptationLevel
from cbpa.models.schedule import Schedule

logger = logging.getLogger(__name__)

# Default threshold below which micro/meso adaptations are deemed
# insufficient and a full macro replan (back to L1-L2) is triggered.
MACRO_ADAPT_THRESHOLD: float = 0.90


@dataclass
class AdaptationAction:
    """A single adaptation action applied to the system."""

    level: AdaptationLevel
    description: str
    parameter_changes: dict[str, float | str]


class AdaptationEngine:
    """Applies micro/meso/macro adaptation to production schedules.

    - micro: parameter retune (speed, buffer adjustments)
    - meso: policy family switch (e.g., high-speed → robust)
    - macro: full replanning from scratch (back to L1-L2)
    """

    # ------------------------------------------------------------------
    # Macro-adaptation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def should_macro_adapt(
        p_feasible: float,
        threshold: float = MACRO_ADAPT_THRESHOLD,
    ) -> bool:
        """Return True when p_feasible is below *threshold*.

        This indicates that lower-tier adaptations (micro parameter retune,
        meso policy family switch) are insufficient and a full replanning
        pass back to L1-L2 contract re-elicitation is required.
        """
        return p_feasible < threshold

    @staticmethod
    def macro_adapt(
        reason: str = "p_feasible below macro-adaptation threshold",
    ) -> AdaptationAction:
        """Create a MACRO adaptation action requesting full replanning.

        The caller is responsible for actually performing the L1-L2 replan;
        this method only produces the action record for the audit trail.
        """
        return AdaptationAction(
            level=AdaptationLevel.MACRO,
            description=(
                f"Full replan from L1-L2 required: {reason}"
            ),
            parameter_changes={"action": "replan_from_scratch"},
        )

    # ------------------------------------------------------------------
    # Standard micro / meso adaptation
    # ------------------------------------------------------------------

    def adapt(
        self,
        current_schedule: Schedule,
        target_schedule: Schedule,
    ) -> list[AdaptationAction]:
        """Determine and apply adaptations to transition between schedules."""
        actions: list[AdaptationAction] = []

        # Meso: detect policy family switch
        if current_schedule.is_aggressive and not target_schedule.is_aggressive:
            actions.append(
                AdaptationAction(
                    level=AdaptationLevel.MESO,
                    description="Switch from high-speed policy family to robust policy family",
                    parameter_changes={
                        "policy_family": "robust",
                        "from_schedule": current_schedule.name,
                        "to_schedule": target_schedule.name,
                    },
                )
            )

        # Micro: parameter retune for specific values
        param_changes: dict[str, float | str] = {}
        if current_schedule.r1_speed_fraction != target_schedule.r1_speed_fraction:
            param_changes["r1_speed_fraction"] = target_schedule.r1_speed_fraction
        if current_schedule.r2_speed_fraction != target_schedule.r2_speed_fraction:
            param_changes["r2_speed_fraction"] = target_schedule.r2_speed_fraction
        if current_schedule.buffer_time_s != target_schedule.buffer_time_s:
            param_changes["buffer_time_s"] = target_schedule.buffer_time_s
        if current_schedule.human_cycle_rate_multiplier != target_schedule.human_cycle_rate_multiplier:
            param_changes["human_cycle_rate_multiplier"] = target_schedule.human_cycle_rate_multiplier

        if param_changes:
            actions.append(
                AdaptationAction(
                    level=AdaptationLevel.MICRO,
                    description="Retune robot velocity and buffer timing parameters",
                    parameter_changes=param_changes,
                )
            )

        for action in actions:
            logger.info(f"Adaptation [{action.level.value}]: {action.description}")

        return actions
