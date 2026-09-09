"""Assumption drift tracking: detects when world-model assumptions deviate
from their contracted bounds and attributes constraint breaches back to the
specific assumptions that drifted.

This is a core CBPA differentiator: rather than simply flagging that a
constraint was violated, the system explains *why* — which environmental
assumption the contract was written under no longer holds.
"""

from __future__ import annotations

from pydantic import BaseModel

from cbpa.models.contract import Assumption


# ---------------------------------------------------------------------------
# Attribution map: assumption name → constraint names it can cause to breach.
# A demand surge raises throughput pressure → more human overtime → fatigue.
# A robot speed change directly drives fatigue and noise.
# Network latency delays feedback → cyber risk.
# Ambient temperature affects sensor calibration → defect rate.
# ---------------------------------------------------------------------------
_ASSUMPTION_CONSTRAINT_MAP: dict[str, list[str]] = {
    "demand_base_uph": ["FatigueIndex", "Noise", "DeadlineGap"],
    "r1_max_speed_mps": ["FatigueIndex", "Noise"],
    "r2_max_speed_mps": ["FatigueIndex", "Noise"],
    "network_latency_ms": ["CyberRiskLevel"],
    "ambient_temp_c": ["DefectRate"],
}

# Normalize LLM-generated constraint names to canonical forms.
_CANONICAL_CONSTRAINT: dict[str, str] = {
    "fatigueindex": "FatigueIndex",
    "fatigue_index": "FatigueIndex",
    "fatigue": "FatigueIndex",
    "operator_fatigue": "FatigueIndex",
    "noise": "Noise",
    "noise_db": "Noise",
    "noise_level": "Noise",
    "deadlinegap": "DeadlineGap",
    "deadline_gap": "DeadlineGap",
    "deadline_gap_pct": "DeadlineGap",
    "cyberrisklevel": "CyberRiskLevel",
    "cyber_risk": "CyberRiskLevel",
    "defectrate": "DefectRate",
    "defect_rate": "DefectRate",
    "defect": "DefectRate",
}


def _normalize_constraint(name: str) -> str:
    """Map any constraint name variant to its canonical form."""
    return _CANONICAL_CONSTRAINT.get(name.lower().strip(), name)


class AssumptionDrift(BaseModel):
    """Records a single assumption violation with its attribution message."""

    assumption: Assumption
    drift_pct: float | None
    attribution: str
    """Human-readable message linking the drifted assumption to the impact,
    e.g. ``"demand_base_uph drifted from 52.0 to 62.4 (+20.0%),
    exceeding 10.0% tolerance"``."""


class AssumptionTracker:
    """Tracks live values of typed assumptions against their contracted bounds.

    Usage::

        tracker = AssumptionTracker(contract.typed_assumptions)
        drift = tracker.update("demand_base_uph", 62.4)
        # drift.attribution -> "demand_base_uph drifted from 52.0 to 62.4
        #                        (+20.0%), exceeding 10.0% tolerance"

        causes = tracker.attribute_constraint_breach("FatigueIndex", "hard")
        # causes -> ["demand_base_uph drift (+20.0%) likely caused FatigueIndex breach"]
    """

    def __init__(self, assumptions: list[Assumption]) -> None:
        # Work on mutable copies so the originals stay clean.
        self.assumptions: dict[str, Assumption] = {
            a.name: a.model_copy() for a in assumptions
        }
        self.drifts: list[AssumptionDrift] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, name: str, current_value: float) -> AssumptionDrift | None:
        """Update an assumption's current value.

        Returns an :class:`AssumptionDrift` if the new value violates the
        tolerance bound, otherwise ``None``.

        Parameters
        ----------
        name:
            The assumption name as registered in ``typed_assumptions``.
        current_value:
            The freshly measured value.

        Returns
        -------
        AssumptionDrift | None
            Drift record when tolerance is exceeded, ``None`` otherwise.
        """
        if name not in self.assumptions:
            return None

        assumption = self.assumptions[name]
        satisfied = assumption.check(current_value)
        expected = assumption.expected_value
        drift_pct = assumption.drift_pct
        if not satisfied:
            if drift_pct is None:
                attribution = (f"{name} changed from zero to {current_value}; "
                               "relative drift is undefined and the zero-baseline tolerance is violated")
            else:
                attribution = (
                    f"{name} drifted from {expected} to {current_value} "
                    f"({drift_pct:+.1f}%), exceeding {assumption.tolerance_pct}% tolerance"
                )
            drift = AssumptionDrift(
                assumption=assumption.model_copy(),
                drift_pct=drift_pct,
                attribution=attribution,
            )
            # De-duplicate: replace any previous drift record for same assumption.
            self.drifts = [d for d in self.drifts if d.assumption.name != name]
            self.drifts.append(drift)
            return drift

        # Drift is within tolerance; clear any prior violation for this name.
        assumption.violated = False
        self.drifts = [d for d in self.drifts if d.assumption.name != name]
        return None

    def get_violations(self) -> list[AssumptionDrift]:
        """Return all currently violated assumption drifts."""
        return list(self.drifts)

    def attribute_constraint_breach(
        self,
        constraint_name: str,
        constraint_type: str = "hard",
    ) -> list[str]:
        """Map a constraint breach to likely assumption drifts.

        Looks up which violated assumptions are known to affect
        ``constraint_name`` (via the static attribution map) and returns
        human-readable attribution strings.

        Parameters
        ----------
        constraint_name:
            The name of the breached constraint, e.g. ``"FatigueIndex"``.
        constraint_type:
            ``"hard"`` or ``"soft"``; included in the message for clarity.

        Returns
        -------
        list[str]
            One attribution string per contributing assumption drift,
            or an empty list when no violated assumptions are mapped to
            this constraint.

        Example
        -------
        ::

            tracker.attribute_constraint_breach("FatigueIndex", "hard")
            # -> ["demand_base_uph drift (+20.0%) likely caused FatigueIndex
            #      (hard) breach"]
        """
        results: list[str] = []
        canonical = _normalize_constraint(constraint_name)
        for drift in self.drifts:
            mapped_constraints = _ASSUMPTION_CONSTRAINT_MAP.get(
                drift.assumption.name, []
            )
            if canonical in mapped_constraints:
                percentage = "undefined at zero baseline" if drift.drift_pct is None else f"{drift.drift_pct:+.1f}%"
                results.append(
                    f"{drift.assumption.name} drift ({percentage}) "
                    f"may contribute to {constraint_name} ({constraint_type}) breach"
                )
        return results

    def summary(self) -> list[str]:
        """Return attribution strings for all current drifts."""
        return [d.attribution for d in self.drifts]
