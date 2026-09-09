"""Layer 4: LLM-based attribution for constraint breaches during execution.

When assumption drift causes a constraint breach, this agent explains
the causal chain and assesses urgency, providing richer attribution
than the static assumption-constraint map.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AttributionExplanation:
    """LLM-generated explanation of how assumption drifts caused breaches."""

    drifted_assumption: str
    drift_pct: float
    violated_constraints: list[str]
    mechanism: str
    urgency: str  # "low", "medium", "high"
    recommended_action: str


@dataclass
class AttributionResult:
    """Aggregated attribution explanations for a monitoring event."""

    phase: int
    explanations: list[AttributionExplanation] = field(default_factory=list)
    llm_used: bool = False


class L4AttributionAgent:
    """Uses LLM to explain why constraints are breached during execution.

    Unlike the static _ASSUMPTION_CONSTRAINT_MAP, this agent reasons about
    the specific causal chain for the current execution context.
    """

    def __init__(self, client=None, use_llm: bool = False):
        self.client = client
        self.use_llm = use_llm and client is not None and getattr(client, "is_available", False)

    def explain_breach(
        self,
        phase: int,
        drifts: list[dict],
        current_metrics: dict[str, float],
        breached_constraints: dict[str, float],
        schedule_name: str = "",
    ) -> AttributionResult:
        """Generate explanations for constraint breaches.

        Parameters
        ----------
        phase:
            Current lifecycle phase.
        drifts:
            List of dicts with keys: name, expected, actual, drift_pct.
        current_metrics:
            Current metric values (throughput_uph, fatigue_index, etc.).
        breached_constraints:
            Constraint name -> violation margin (negative = violation).
        schedule_name:
            Name of the active schedule.
        """
        if not breached_constraints:
            return AttributionResult(phase=phase, llm_used=False)

        if not self.use_llm:
            return AttributionResult(
                phase=phase,
                explanations=_deterministic_attribution(drifts, breached_constraints),
                llm_used=False,
            )

        from cbpa.llm.prompts import L4_ATTRIBUTION_SYSTEM, L4_ATTRIBUTION_TOOL_SCHEMA

        drift_summary = "\n".join(
            f"- {d['name']}: {d.get('drift_pct', 0):+.1f}% "
            f"(expected {d.get('expected', '?')}, actual {d.get('actual', '?')})"
            for d in drifts
        )
        metrics_str = "\n".join(f"  {k}: {v}" for k, v in current_metrics.items())
        breach_str = "\n".join(
            f"  {c}: margin {m:.3f}" for c, m in breached_constraints.items()
        )

        user_msg = (
            f"During Phase {phase} execution of schedule '{schedule_name}', "
            f"the following assumptions have drifted:\n{drift_summary}\n\n"
            f"Current metrics:\n{metrics_str}\n\n"
            f"Breached constraints (negative margin = violation):\n{breach_str}\n\n"
            f"For each breached constraint, explain the causal chain from "
            f"assumption drift to breach. Assess urgency (low/medium/high) "
            f"and suggest an immediate action."
        )

        try:
            result = self.client.query_structured(
                system=L4_ATTRIBUTION_SYSTEM,
                user_message=user_msg,
                tool_name="attribution_explanations",
                tool_schema=L4_ATTRIBUTION_TOOL_SCHEMA,
                tool_description="Explain constraint breaches via assumption drift",
            )
            explanations = [
                AttributionExplanation(
                    drifted_assumption=e.get("drifted_assumption", ""),
                    drift_pct=e.get("drift_pct", 0.0),
                    violated_constraints=e.get("violated_constraints", []),
                    mechanism=e.get("mechanism", ""),
                    urgency=e.get("urgency", "medium"),
                    recommended_action=e.get("recommended_action", ""),
                )
                for e in result.get("explanations", [])
            ]
            logger.info(
                f"L4 attribution agent generated {len(explanations)} explanations "
                f"for Phase {phase}"
            )
            return AttributionResult(
                phase=phase,
                explanations=explanations,
                llm_used=True,
            )
        except Exception as e:
            logger.warning(f"L4 attribution agent failed: {e}")
            return AttributionResult(
                phase=phase,
                explanations=_deterministic_attribution(drifts, breached_constraints),
                llm_used=False,
            )


def _deterministic_attribution(
    drifts: list[dict], breached: dict[str, float]
) -> list[AttributionExplanation]:
    """Rule-based fallback using static causal mapping."""
    _CAUSE_MAP = {
        "FatigueIndex": ("demand_base_uph", "Higher demand increases cycle pressure and cumulative fatigue"),
        "Noise": ("demand_base_uph", "Higher throughput requires faster robot speeds, raising noise"),
        "NoiseLimit": ("demand_base_uph", "Higher throughput requires faster robot speeds, raising noise"),
        "Energy": ("demand_base_uph", "Higher production volume directly increases energy consumption"),
    }
    explanations = []
    for constraint, margin in breached.items():
        if margin >= 0:
            continue
        cause, mechanism = _CAUSE_MAP.get(constraint, ("unknown", "Unknown causal chain"))
        drift_pct = 0.0
        for d in drifts:
            if d.get("name") == cause:
                drift_pct = d.get("drift_pct", 0.0)
                break
        explanations.append(AttributionExplanation(
            drifted_assumption=cause,
            drift_pct=drift_pct,
            violated_constraints=[constraint],
            mechanism=mechanism,
            urgency="high" if abs(margin) > 0.05 else "medium",
            recommended_action="Escalate to governance" if abs(margin) > 0.05 else "Monitor closely",
        ))
    return explanations
