"""Layer 3: LLM-based constraint repair mediation.

When a candidate schedule is rejected by the verification firewall,
this agent interprets the rejection reason and suggests targeted
parameter adjustments to bring the schedule back into feasibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RepairSuggestion:
    """A suggested parameter adjustment to fix a constraint violation."""

    parameter: str
    current_value: float
    suggested_value: float
    rationale: str
    expected_improvement: str = ""


@dataclass
class RepairResult:
    """Aggregated repair suggestions for a rejected candidate."""

    schedule_name: str
    violations: list[str]
    suggestions: list[RepairSuggestion] = field(default_factory=list)
    llm_used: bool = False


class L3RepairAgent:
    """Uses LLM to suggest parameter tweaks when constraints are violated.

    The LLM never replaces the verifier — it mediates between the verifier
    and the planner, accelerating convergence without compromising the
    formal safety guarantee.
    """

    def __init__(self, client=None, use_llm: bool = False):
        self.client = client
        self.use_llm = use_llm and client is not None and getattr(client, "is_available", False)

    def suggest_repairs(
        self,
        schedule_name: str,
        schedule_params: dict,
        violations: list[str],
        constraint_margins: dict[str, float],
        metrics: dict[str, float] | None = None,
    ) -> RepairResult:
        """Generate repair suggestions for a rejected schedule.

        Parameters
        ----------
        schedule_name:
            Name of the rejected candidate.
        schedule_params:
            Current parameter values (r1_speed_fraction, etc.).
        violations:
            List of violation strings from the feasibility report.
        constraint_margins:
            Constraint name -> margin (negative = violation).
        metrics:
            Optional measured metrics for context.
        """
        if not self.use_llm or not violations:
            return RepairResult(
                schedule_name=schedule_name,
                violations=violations,
                suggestions=_deterministic_repairs(schedule_params, constraint_margins),
                llm_used=False,
            )

        from cbpa.llm.prompts import L3_REPAIR_SYSTEM, L3_REPAIR_TOOL_SCHEMA

        margin_summary = "; ".join(
            f"{c}: {m:+.3f}" for c, m in constraint_margins.items()
        )
        metrics_str = ""
        if metrics:
            metrics_str = "\n".join(f"  {k}: {v}" for k, v in metrics.items())

        user_msg = (
            f"A manufacturing schedule '{schedule_name}' was rejected.\n\n"
            f"Current parameters:\n"
            + "\n".join(f"  {k}: {v}" for k, v in schedule_params.items())
            + f"\n\nConstraint margins (negative = violation): {margin_summary}\n"
            f"Violations:\n  " + "\n  ".join(violations)
        )
        if metrics_str:
            user_msg += f"\n\nMeasured metrics:\n{metrics_str}"

        user_msg += (
            "\n\nSuggest 2-3 minimal parameter adjustments to fix the violations. "
            "For each, state the parameter, current value, suggested value, and "
            "the causal reasoning. Prioritize smallest changes first."
        )

        try:
            result = self.client.query_structured(
                system=L3_REPAIR_SYSTEM,
                user_message=user_msg,
                tool_name="repair_suggestions",
                tool_schema=L3_REPAIR_TOOL_SCHEMA,
                tool_description="Suggest parameter adjustments to fix constraint violations",
            )
            suggestions = [
                RepairSuggestion(
                    parameter=s.get("parameter", ""),
                    current_value=s.get("current_value", 0.0),
                    suggested_value=s.get("suggested_value", 0.0),
                    rationale=s.get("rationale", ""),
                    expected_improvement=s.get("expected_improvement", ""),
                )
                for s in result.get("suggestions", [])
            ]
            logger.info(
                f"L3 repair agent generated {len(suggestions)} suggestions "
                f"for {schedule_name}"
            )
            return RepairResult(
                schedule_name=schedule_name,
                violations=violations,
                suggestions=suggestions,
                llm_used=True,
            )
        except Exception as e:
            logger.warning(f"L3 repair agent failed: {e}")
            return RepairResult(
                schedule_name=schedule_name,
                violations=violations,
                suggestions=_deterministic_repairs(schedule_params, constraint_margins),
                llm_used=False,
            )


def _deterministic_repairs(
    params: dict, margins: dict[str, float]
) -> list[RepairSuggestion]:
    """Rule-based fallback: reduce the parameter most likely causing the violation."""
    suggestions = []
    for constraint, margin in margins.items():
        if margin >= 0:
            continue
        c_lower = constraint.lower()
        if "fatigue" in c_lower:
            cur = params.get("human_cycle_rate_multiplier", 1.0)
            suggestions.append(RepairSuggestion(
                parameter="human_cycle_rate_multiplier",
                current_value=cur,
                suggested_value=round(max(0.5, cur - 0.15), 2),
                rationale=f"Reduce human cycle rate to lower fatigue (margin {margin:+.3f})",
            ))
        elif "noise" in c_lower:
            cur = params.get("r1_speed_fraction", 0.8)
            suggestions.append(RepairSuggestion(
                parameter="r1_speed_fraction",
                current_value=cur,
                suggested_value=round(max(0.3, cur - 0.1), 2),
                rationale=f"Reduce R1 speed to lower noise (margin {margin:+.3f})",
            ))
        elif "energy" in c_lower:
            cur = params.get("r2_speed_fraction", 0.8)
            suggestions.append(RepairSuggestion(
                parameter="r2_speed_fraction",
                current_value=cur,
                suggested_value=round(max(0.3, cur - 0.1), 2),
                rationale=f"Reduce R2 speed to lower energy (margin {margin:+.3f})",
            ))
    return suggestions
