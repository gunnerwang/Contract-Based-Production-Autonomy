"""Cumulative fatigue index model."""

from __future__ import annotations

from cbpa.config.defaults import FATIGUE_ALPHA, FATIGUE_BETA


class FatigueModel:
    """
    Fatigue = alpha * effective_load^beta

    where effective_load = h_rate * (0.5 + 0.3*(r1_speed + r2_speed))

    Captures how human workload increases with both own pace and robot speeds
    (faster robots mean less buffer time for the human).
    """

    def __init__(
        self,
        alpha: float = FATIGUE_ALPHA,
        beta: float = FATIGUE_BETA,
    ):
        self.alpha = alpha
        self.beta = beta

    def effective_load(
        self,
        r1_speed_fraction: float,
        r2_speed_fraction: float,
        human_rate: float,
    ) -> float:
        return human_rate * (0.5 + 0.3 * (r1_speed_fraction + r2_speed_fraction))

    def compute(
        self,
        r1_speed_fraction: float,
        r2_speed_fraction: float,
        human_rate: float,
        shift_fraction: float = 1.0,
    ) -> float:
        load = self.effective_load(r1_speed_fraction, r2_speed_fraction, human_rate)
        fatigue = self.alpha * (load ** self.beta) * (shift_fraction ** 0.8)
        return round(min(fatigue, 1.0), 4)
