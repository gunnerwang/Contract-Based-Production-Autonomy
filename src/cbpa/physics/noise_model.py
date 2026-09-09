"""Noise model: noise_db = f(robot speeds)."""

from __future__ import annotations

from cbpa.config.defaults import NOISE_BASE_DB, NOISE_K1, NOISE_K2


class NoiseModel:
    """
    Noise (dB) = base + k1 * r1_speed + k2 * r2_speed

    R2 (screwdriver) dominates noise contribution.
    Analytical model provides approximate fit; deterministic mode
    uses exact Table 3 values.
    """

    def __init__(
        self,
        base_db: float = NOISE_BASE_DB,
        k1: float = NOISE_K1,
        k2: float = NOISE_K2,
    ):
        self.base_db = base_db
        self.k1 = k1
        self.k2 = k2

    def compute(
        self, r1_speed_fraction: float, r2_speed_fraction: float
    ) -> float:
        noise = (
            self.base_db
            + self.k1 * r1_speed_fraction
            + self.k2 * r2_speed_fraction
        )
        return round(noise, 1)

    @staticmethod
    def compute_factory(*cell_noise_db: float) -> float:
        """Combine noise from multiple cells via logarithmic addition.

        Two cells each at 79 dB produce ~82 dB combined.  This models
        the physical reality that acoustic power adds linearly while dB
        is a logarithmic scale.

        Args:
            *cell_noise_db: Noise level of each cell in dB.

        Returns:
            Combined factory noise in dB.
        """
        import math

        if not cell_noise_db:
            return 0.0
        total_power = sum(10 ** (db / 10.0) for db in cell_noise_db)
        return round(10.0 * math.log10(total_power), 1)
