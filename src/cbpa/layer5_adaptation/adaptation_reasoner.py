"""Layer 5: LLM-based reasoning for adaptation tier selection and experience synthesis.

Instead of threshold-based tier selection (p_feasible < 0.90 -> macro),
this agent reasons about the specific disturbance context, binding
constraints, and prior experience to recommend the most appropriate
adaptation tier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cbpa.models.escalation import AdaptationLevel
from cbpa.layer5_adaptation.learning_store import ExperienceRecord

logger = logging.getLogger(__name__)


@dataclass
class AdaptationReasoning:
    """LLM's analysis of which adaptation tier to use and why."""

    recommended_tier: AdaptationLevel
    confidence: float
    rationale: str
    key_insight: str
    prior_experience_used: bool = False
    experience_summary: str = ""


@dataclass
class ExperienceSynthesis:
    """LLM-synthesized learning from multiple experience records."""

    disturbance_pattern: str
    successful_policies: list[str] = field(default_factory=list)
    failed_policies: list[str] = field(default_factory=list)
    parameter_hints: dict[str, list[float]] = field(default_factory=dict)
    insight: str = ""
    confidence: float = 0.5


class L5AdaptationReasoner:
    """Uses LLM to reason about adaptation tier selection and learning synthesis."""

    def __init__(self, client=None, use_llm: bool = False):
        self.client = client
        self.use_llm = use_llm and client is not None and getattr(client, "is_available", False)

    def reason_tier_selection(
        self,
        p_feasible: float,
        constraint_violations: dict[str, float],
        prior_experiences: list[ExperienceRecord],
        disturbance_context: str,
        schedule_params: dict | None = None,
    ) -> AdaptationReasoning:
        """Recommend which adaptation tier is best for this context.

        Parameters
        ----------
        p_feasible:
            Probability of feasibility from stochastic verification.
        constraint_violations:
            Constraint name -> violation probability.
        prior_experiences:
            Past experiences with similar disturbances.
        disturbance_context:
            Human-readable description of the current disturbance.
        schedule_params:
            Current schedule parameters for context.
        """
        if not self.use_llm:
            tier = AdaptationLevel.MACRO if p_feasible < 0.90 else AdaptationLevel.MESO
            return AdaptationReasoning(
                recommended_tier=tier,
                confidence=0.5,
                rationale=f"Deterministic threshold: p_feasible={p_feasible:.2%}",
                key_insight="Rule-based tier selection (no LLM reasoning)",
            )

        from cbpa.llm.prompts import L5_REASONING_SYSTEM, L5_REASONING_TOOL_SCHEMA

        violation_summary = "; ".join(
            f"{c}: {p:.0%} violation probability"
            for c, p in sorted(constraint_violations.items(), key=lambda x: x[1], reverse=True)
        )

        prior_summary = ""
        if prior_experiences:
            rejected = [r.rejected_policy for r in prior_experiences if r.rejected_policy]
            accepted = [r.accepted_policy for r in prior_experiences if r.accepted_policy]
            prior_summary = (
                f"{len(prior_experiences)} prior similar situations: "
                f"rejected={rejected}, accepted={accepted}"
            )

        params_str = ""
        if schedule_params:
            params_str = "\nCurrent schedule: " + ", ".join(
                f"{k}={v}" for k, v in schedule_params.items()
            )

        user_msg = (
            f"P(feasible) = {p_feasible:.2%}\n"
            f"Constraint violations: {violation_summary}\n"
            f"Disturbance: {disturbance_context}\n"
            f"Prior learning: {prior_summary or 'None (first shift)'}"
            f"{params_str}\n\n"
            f"Which adaptation tier? MICRO (parameter retune), "
            f"MESO (policy family switch), or MACRO (full replan)?\n"
            f"Consider which constraint is binding and whether it can be "
            f"fixed by small adjustments or requires fundamental replanning."
        )

        try:
            result = self.client.query_structured(
                system=L5_REASONING_SYSTEM,
                user_message=user_msg,
                tool_name="adaptation_reasoning",
                tool_schema=L5_REASONING_TOOL_SCHEMA,
                tool_description="Recommend adaptation tier with reasoning",
            )
            tier_map = {
                "MICRO": AdaptationLevel.MICRO,
                "MESO": AdaptationLevel.MESO,
                "MACRO": AdaptationLevel.MACRO,
            }
            tier_str = result.get("recommended_tier", "MACRO")
            reasoning = AdaptationReasoning(
                recommended_tier=tier_map.get(tier_str, AdaptationLevel.MACRO),
                confidence=result.get("confidence", 0.5),
                rationale=result.get("rationale", ""),
                key_insight=result.get("key_insight", ""),
                prior_experience_used=bool(prior_experiences),
                experience_summary=prior_summary,
            )
            logger.info(
                f"L5 reasoner: {reasoning.recommended_tier.value} "
                f"(confidence={reasoning.confidence:.0%}): {reasoning.key_insight}"
            )
            return reasoning
        except Exception as e:
            logger.warning(f"L5 adaptation reasoner failed: {e}")
            tier = AdaptationLevel.MACRO if p_feasible < 0.90 else AdaptationLevel.MESO
            return AdaptationReasoning(
                recommended_tier=tier,
                confidence=0.3,
                rationale=f"LLM failed ({e}), fallback to threshold",
                key_insight="",
            )

    def synthesize_experience(
        self,
        records: list[ExperienceRecord],
        disturbance_type: str,
    ) -> ExperienceSynthesis | None:
        """Synthesize patterns across multiple experience records.

        Instead of just extracting min/max bounds, the LLM identifies
        the common pattern, successful strategies, and parameter sweet spots.
        """
        if not records:
            return None

        if not self.use_llm:
            return _deterministic_synthesis(records, disturbance_type)

        from cbpa.llm.prompts import L5_SYNTHESIS_SYSTEM, L5_SYNTHESIS_TOOL_SCHEMA

        record_summary = "\n".join(
            f"  Attempt {i+1}: rejected '{r.rejected_policy}' "
            f"(reason: {r.rejection_reason}), "
            f"accepted '{r.accepted_policy}', "
            f"satisfied={r.constraint_satisfied}, "
            f"metrics={r.outcome_metrics}"
            for i, r in enumerate(records)
        )

        user_msg = (
            f"Synthesize patterns from {len(records)} past adaptation "
            f"experiences for disturbance type '{disturbance_type}':\n\n"
            f"{record_summary}\n\n"
            f"Identify: (1) successful policies, (2) failed policies, "
            f"(3) parameter ranges that worked, (4) confidence given "
            f"the small sample size."
        )

        try:
            result = self.client.query_structured(
                system=L5_SYNTHESIS_SYSTEM,
                user_message=user_msg,
                tool_name="learning_synthesis",
                tool_schema=L5_SYNTHESIS_TOOL_SCHEMA,
                tool_description="Synthesize learning patterns from experience",
            )
            synthesis = ExperienceSynthesis(
                disturbance_pattern=result.get("disturbance_pattern", disturbance_type),
                successful_policies=result.get("successful_policies", []),
                failed_policies=result.get("failed_policies", []),
                parameter_hints=result.get("parameter_hints", {}),
                insight=result.get("insight", ""),
                confidence=result.get("confidence", 0.5),
            )
            logger.info(
                f"L5 synthesis: {synthesis.disturbance_pattern}, "
                f"confidence={synthesis.confidence:.0%}"
            )
            return synthesis
        except Exception as e:
            logger.warning(f"L5 synthesis failed: {e}")
            return _deterministic_synthesis(records, disturbance_type)


def _deterministic_synthesis(
    records: list[ExperienceRecord], disturbance_type: str
) -> ExperienceSynthesis:
    """Rule-based fallback: extract min/max bounds from successful records."""
    successful = [r for r in records if r.constraint_satisfied]
    failed = [r for r in records if not r.constraint_satisfied]
    return ExperienceSynthesis(
        disturbance_pattern=disturbance_type,
        successful_policies=[r.accepted_policy for r in successful if r.accepted_policy],
        failed_policies=[r.rejected_policy for r in failed if r.rejected_policy],
        insight=f"Deterministic synthesis from {len(records)} records",
        confidence=min(0.3 + 0.1 * len(records), 0.8),
    )
