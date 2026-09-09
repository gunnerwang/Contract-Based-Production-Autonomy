"""Energy consumption model."""

from __future__ import annotations

from cbpa.config.defaults import (
    ENERGY_BASE_KWH,
    ENERGY_HUMAN_COEFF,
    ENERGY_ROBOT_COEFF,
)


class EnergyModel:
    """
    Energy (kWh/shift) = base + c_robot*(r1^2 + r2^2)*hours + c_human*h_rate*hours

    Robot energy scales quadratically with speed; human-station energy
    (lighting, tools, ventilation) scales linearly with activity rate.
    """

    def __init__(
        self,
        base_kwh: float = ENERGY_BASE_KWH,
        robot_coeff: float = ENERGY_ROBOT_COEFF,
        human_coeff: float = ENERGY_HUMAN_COEFF,
    ):
        self.base_kwh = base_kwh
        self.robot_coeff = robot_coeff
        self.human_coeff = human_coeff

    def compute(
        self,
        r1_speed_fraction: float,
        r2_speed_fraction: float,
        human_rate: float = 1.0,
        hours: float = 8.0,
    ) -> float:
        robot_term = self.robot_coeff * (
            r1_speed_fraction ** 2 + r2_speed_fraction ** 2
        ) * hours
        human_term = self.human_coeff * human_rate * hours
        energy = self.base_kwh + robot_term + human_term
        return round(energy, 0)
