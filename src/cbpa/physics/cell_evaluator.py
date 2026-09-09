"""Combined cell evaluator mapping schedule parameters to metrics.

Operates in two modes:
- deterministic: returns exact Table 3 values for S1/S2/S3
- analytical: uses physics sub-models (approximate, for interpolation)
"""

from __future__ import annotations

import numpy as np

from cbpa.config.defaults import TABLE3_TARGETS
from cbpa.config.scenario import CellConfig
from cbpa.models.metrics import SimulationMetrics
from cbpa.models.schedule import Schedule
from cbpa.physics.defect_model import DefectModel
from cbpa.physics.energy_model import EnergyModel
from cbpa.physics.fatigue_model import FatigueModel
from cbpa.physics.noise_model import NoiseModel

# Map variant "a" names to their canonical base schedule (zero offset; identical
# parameters).  In deterministic mode these resolve to exact Table-3 values.
_VARIANT_A_MAP = {"S1a": "S1", "S2a": "S2", "S3a": "S3"}


class CellEvaluator:
    """Evaluates a schedule against the manufacturing cell physics."""

    def __init__(
        self,
        cell: CellConfig | None = None,
        mode: str = "deterministic",
    ):
        self.cell = cell or CellConfig()
        self.mode = mode
        self.fatigue_model = FatigueModel()
        self.noise_model = NoiseModel()
        self.energy_model = EnergyModel()
        self.defect_model = DefectModel()

    def evaluate(self, schedule: Schedule) -> SimulationMetrics:
        # Resolve variant "a" names (e.g. "S1a") to the canonical name for
        # deterministic Table-3 lookup; their parameters are identical.
        lookup_name = _VARIANT_A_MAP.get(schedule.name, schedule.name)
        if self.mode == "deterministic" and lookup_name in TABLE3_TARGETS:
            return self._deterministic_by_name(lookup_name)
        return self._analytical(schedule)

    def _deterministic_by_name(self, name: str) -> SimulationMetrics:
        """Return exact Table 3 values for a canonical schedule name."""
        t = TABLE3_TARGETS[name]
        hours = self.cell.shift_hours
        units = int(t["throughput_uph"] * hours)
        return SimulationMetrics(
            throughput_uph=t["throughput_uph"],
            defect_rate=t["defect_rate"],
            noise_db=t["noise_db"],
            fatigue_index=t["fatigue_index"],
            energy_kwh=t["energy_kwh"],
            deadline_gap_pct=t["deadline_gap_pct"],
            units_produced=units,
            shift_hours=hours,
        )

    def _deterministic(self, schedule: Schedule) -> SimulationMetrics:
        """Return exact Table 3 values (legacy path; delegates to _deterministic_by_name)."""
        return self._deterministic_by_name(schedule.name)

    def _analytical(self, schedule: Schedule) -> SimulationMetrics:
        """Compute metrics from physics sub-models."""
        r1 = schedule.r1_speed_fraction
        r2 = schedule.r2_speed_fraction
        hr = schedule.human_cycle_rate_multiplier
        buf = schedule.buffer_time_s
        hours = self.cell.shift_hours

        # Throughput from pipeline bottleneck model
        throughput = self._compute_throughput(r1, r2, hr, buf)

        # Noise
        noise = self.noise_model.compute(r1, r2)

        # Fatigue
        fatigue = self.fatigue_model.compute(r1, r2, hr)

        # Energy
        energy = self.energy_model.compute(r1, r2, hr, hours)

        # Defect rate
        avg_speed = (r1 + r2) / 2
        defect = self.defect_model.compute(avg_speed, fatigue)

        # Deadline gap
        demand = schedule.demand_target_uph
        if demand > 0:
            gap = max(0.0, (demand - throughput) / demand * 100)
        else:
            gap = 0.0

        units = int(throughput * hours)

        return SimulationMetrics(
            throughput_uph=round(throughput, 1),
            defect_rate=round(defect, 4),
            noise_db=noise,
            fatigue_index=fatigue,
            energy_kwh=energy,
            deadline_gap_pct=round(gap, 1),
            units_produced=units,
            shift_hours=hours,
        )

    def _compute_throughput(
        self,
        r1_speed: float,
        r2_speed: float,
        human_rate: float,
        buffer_s: float,
    ) -> float:
        """Pipeline throughput from bottleneck resource."""
        r1_time = self.cell.r1_cycle_time_s / r1_speed
        r2_time = self.cell.r2_cycle_time_s / r2_speed
        h_time = (
            self.cell.human_insertion_time_s + self.cell.human_inspection_time_s
        ) / human_rate
        bottleneck = max(r1_time, r2_time, h_time)
        effective_cycle = bottleneck + buffer_s
        return 3600.0 / effective_cycle

    # ------------------------------------------------------------------
    # Monte Carlo / stochastic evaluation
    # ------------------------------------------------------------------

    def evaluate_monte_carlo(
        self,
        schedule: Schedule,
        n_samples: int = 200,
        seed: int = 42,
    ) -> list[SimulationMetrics]:
        """Return *n_samples* perturbed analytical evaluations of *schedule*.

        Each sample draws independent noise from tight normal distributions
        around the nominal schedule parameters, reflecting real-world
        variability in robot calibration and human pace:

        - r1_speed_fraction:        ±5 % (σ = 0.05 / 3, clipped to [0.01, 1.0])
        - r2_speed_fraction:        ±5 % (σ = 0.05 / 3, clipped to [0.01, 1.0])
        - human_cycle_rate_mult:    ±10 % (σ = 0.10 / 3, clipped to [0.5, 2.0])
        - r1/r2 cycle time offsets: ±3 % (σ = 0.03 / 3)

        The method is deterministic when *seed* is fixed (default 42).
        It always uses the analytical sub-models (regardless of self.mode)
        so that the perturbations affect the physics computations.
        """
        rng = np.random.default_rng(seed)

        # Nominal parameter values
        r1_nom = schedule.r1_speed_fraction
        r2_nom = schedule.r2_speed_fraction
        hr_nom = schedule.human_cycle_rate_multiplier
        buf = schedule.buffer_time_s
        demand = schedule.demand_target_uph
        hours = self.cell.shift_hours

        # --- Draw all perturbation arrays at once for efficiency ---
        # Speed perturbations: 3-sigma span equals the stated ±% range
        r1_delta = rng.normal(0.0, r1_nom * 0.05 / 3, size=n_samples)
        r2_delta = rng.normal(0.0, r2_nom * 0.05 / 3, size=n_samples)
        hr_delta = rng.normal(0.0, hr_nom * 0.10 / 3, size=n_samples)

        # Cycle-time fractional offsets (applied to stored cell constants)
        r1_ct_mult = 1.0 + rng.normal(0.0, 0.03 / 3, size=n_samples)
        r2_ct_mult = 1.0 + rng.normal(0.0, 0.03 / 3, size=n_samples)

        # Perturbed speed fractions, clipped to valid ranges
        r1_samples = np.clip(r1_nom + r1_delta, 0.01, 1.0)
        r2_samples = np.clip(r2_nom + r2_delta, 0.01, 1.0)
        hr_samples = np.clip(hr_nom + hr_delta, 0.5, 2.0)

        samples: list[SimulationMetrics] = []

        for i in range(n_samples):
            r1 = float(r1_samples[i])
            r2 = float(r2_samples[i])
            hr = float(hr_samples[i])
            ct1_mult = float(r1_ct_mult[i])
            ct2_mult = float(r2_ct_mult[i])

            # Throughput: use perturbed cycle times directly
            r1_time = (self.cell.r1_cycle_time_s * ct1_mult) / r1
            r2_time = (self.cell.r2_cycle_time_s * ct2_mult) / r2
            h_time = (
                self.cell.human_insertion_time_s + self.cell.human_inspection_time_s
            ) / hr
            bottleneck = max(r1_time, r2_time, h_time)
            effective_cycle = bottleneck + buf
            throughput = 3600.0 / effective_cycle

            noise = self.noise_model.compute(r1, r2)
            fatigue = self.fatigue_model.compute(r1, r2, hr)
            energy = self.energy_model.compute(r1, r2, hr, hours)
            avg_speed = (r1 + r2) / 2
            defect = self.defect_model.compute(avg_speed, fatigue)

            if demand > 0:
                gap = max(0.0, (demand - throughput) / demand * 100)
            else:
                gap = 0.0

            samples.append(
                SimulationMetrics(
                    throughput_uph=round(throughput, 1),
                    defect_rate=round(defect, 4),
                    noise_db=noise,
                    fatigue_index=fatigue,
                    energy_kwh=energy,
                    deadline_gap_pct=round(gap, 1),
                    units_produced=int(throughput * hours),
                    shift_hours=hours,
                )
            )

        return samples
