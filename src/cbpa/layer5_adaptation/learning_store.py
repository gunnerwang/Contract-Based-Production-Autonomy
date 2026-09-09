"""Layer 5: Experience memory for adaptation learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExperienceRecord:
    """Records a disturbance context and the outcome of adaptation."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    disturbance_type: str = ""
    disturbance_magnitude: float = 0.0
    rejected_policy: str = ""
    rejection_reason: str = ""
    accepted_policy: str = ""
    adaptation_level: str = ""
    outcome_metrics: dict[str, float] = field(default_factory=dict)
    constraint_satisfied: bool = True


class LearningStore:
    """Stores adaptation experiences for future policy bias.

    Records are used to:
    1. Bias future policy ranking in Layer 2
    2. Tighten early-warning thresholds in Layer 4
    """

    def __init__(self) -> None:
        self.records: list[ExperienceRecord] = []

    def record(
        self,
        disturbance_type: str,
        disturbance_magnitude: float,
        rejected_policy: str,
        rejection_reason: str,
        accepted_policy: str,
        adaptation_level: str,
        outcome_metrics: dict[str, float],
        constraint_satisfied: bool = True,
    ) -> ExperienceRecord:
        entry = ExperienceRecord(
            disturbance_type=disturbance_type,
            disturbance_magnitude=disturbance_magnitude,
            rejected_policy=rejected_policy,
            rejection_reason=rejection_reason,
            accepted_policy=accepted_policy,
            adaptation_level=adaptation_level,
            outcome_metrics=outcome_metrics,
            constraint_satisfied=constraint_satisfied,
        )
        self.records.append(entry)
        return entry

    def get_similar(self, disturbance_type: str) -> list[ExperienceRecord]:
        """Retrieve past experiences with similar disturbance types."""
        return [r for r in self.records if r.disturbance_type == disturbance_type]

    def get_bias_for_planning(self, disturbance_type: str) -> dict:
        """Query past experiences and return planning bias for Layer 2.

        This closes the learning loop: Layer 5 experience memory biases
        future Layer 2 candidate generation by identifying policies to
        avoid, policy families to prefer, and parameter ranges that
        worked well historically.

        Returns an empty dict when no relevant experience exists (e.g.,
        the very first shift), so the planner operates without bias.
        """
        similar = self.get_similar(disturbance_type)
        if not similar:
            return {}
        return {
            "avoid_policies": [r.rejected_policy for r in similar if r.rejected_policy],
            "prefer_families": [r.accepted_policy for r in similar if r.accepted_policy],
            "parameter_hints": self._extract_parameter_bounds(similar),
        }

    def _extract_parameter_bounds(self, records: list[ExperienceRecord]) -> dict:
        """Derive recommended parameter ranges from accepted policies' outcomes.

        Looks at the ``outcome_metrics`` of records whose constraints
        were satisfied and extracts min/max bounds for each metric key.
        These hints let the planner narrow parameter search toward
        historically successful operating points.
        """
        successful = [r for r in records if r.constraint_satisfied and r.outcome_metrics]
        if not successful:
            return {}

        bounds: dict[str, dict[str, float]] = {}
        for rec in successful:
            for key, value in rec.outcome_metrics.items():
                if key not in bounds:
                    bounds[key] = {"min": value, "max": value}
                else:
                    bounds[key]["min"] = min(bounds[key]["min"], value)
                    bounds[key]["max"] = max(bounds[key]["max"], value)
        return bounds

    @property
    def total_records(self) -> int:
        return len(self.records)
