"""Defect rate model."""

from __future__ import annotations

from cbpa.config.defaults import (
    DEFECT_BASE_RATE,
    DEFECT_FATIGUE_BETA,
    DEFECT_SPEED_ALPHA,
)


class DefectModel:
    """
    DefectRate = base * (1 + alpha * (avg_speed - 0.5)) * (1 + beta * fatigue)

    Higher speeds and higher fatigue both increase defect rate.
    """

    def __init__(
        self,
        base_rate: float = DEFECT_BASE_RATE,
        speed_alpha: float = DEFECT_SPEED_ALPHA,
        fatigue_beta: float = DEFECT_FATIGUE_BETA,
    ):
        self.base_rate = base_rate
        self.speed_alpha = speed_alpha
        self.fatigue_beta = fatigue_beta

    def compute(
        self,
        avg_speed_fraction: float,
        fatigue_index: float,
    ) -> float:
        speed_factor = 1.0 + self.speed_alpha * (avg_speed_fraction - 0.5)
        fatigue_factor = 1.0 + self.fatigue_beta * fatigue_index
        rate = self.base_rate * speed_factor * fatigue_factor
        return round(rate, 4)
