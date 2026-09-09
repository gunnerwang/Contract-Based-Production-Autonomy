"""SimPy discrete-event simulation of the manufacturing cell.

Models a pipelined assembly: R1 (pick-place) → H (insert) → R2 (screw) → H (inspect).
Resources: R1, R2 are robots (capacity 1 each), H is human (capacity 1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import simpy

from cbpa.config.scenario import CellConfig
from cbpa.layer3_verification.guard_synthesizer import GuardEnforcer, GuardEvent
from cbpa.layer4_execution.kpi_collector import KPICollector
from cbpa.models.metrics import SimulationMetrics
from cbpa.models.schedule import Schedule
from cbpa.physics.defect_model import DefectModel
from cbpa.physics.energy_model import EnergyModel
from cbpa.physics.fatigue_model import FatigueModel
from cbpa.physics.noise_model import NoiseModel


@dataclass
class SimulationResult:
    """Raw result from a simulation run."""

    metrics: SimulationMetrics
    collector: KPICollector
    demand_met: bool = False
    guard_events: list[GuardEvent] = field(default_factory=list)


class ManufacturingCell:
    """SimPy-based manufacturing cell simulation."""

    def __init__(
        self,
        cell: CellConfig | None = None,
        seed: int = 42,
        guard_enforcer: GuardEnforcer | None = None,
    ):
        self.cell = cell or CellConfig()
        self.rng = random.Random(seed)
        self.fatigue_model = FatigueModel()
        self.noise_model = NoiseModel()
        self.energy_model = EnergyModel()
        self.defect_model = DefectModel()
        self.guard_enforcer = guard_enforcer

    def run(
        self,
        schedule: Schedule,
        demand_spike_at_s: float | None = None,
        demand_spike_pct: float = 20.0,
    ) -> SimulationResult:
        """Run an 8-hour shift simulation."""
        shift_s = self.cell.shift_hours * 3600
        env = simpy.Environment()
        collector = KPICollector()

        r1_res = simpy.Resource(env, capacity=1)
        r2_res = simpy.Resource(env, capacity=1)
        h_res = simpy.Resource(env, capacity=1)

        # Stage times
        r1_time = self.cell.r1_cycle_time_s / schedule.r1_speed_fraction
        r2_time = self.cell.r2_cycle_time_s / schedule.r2_speed_fraction
        h_insert_time = self.cell.human_insertion_time_s / schedule.human_cycle_rate_multiplier
        h_inspect_time = self.cell.human_inspection_time_s / schedule.human_cycle_rate_multiplier
        buffer = schedule.buffer_time_s

        # Pre-compute defect probability
        avg_speed = (schedule.r1_speed_fraction + schedule.r2_speed_fraction) / 2
        fatigue_end = self.fatigue_model.compute(
            schedule.r1_speed_fraction,
            schedule.r2_speed_fraction,
            schedule.human_cycle_rate_multiplier,
        )
        defect_rate = self.defect_model.compute(avg_speed, fatigue_end)

        # Noise (constant for given speed profile)
        noise = self.noise_model.compute(
            schedule.r1_speed_fraction, schedule.r2_speed_fraction
        )

        def assembly_process(env: simpy.Environment, unit_id: int) -> None:
            """Process one unit through the pipeline."""
            # Stage 1: R1 pick-and-place
            with r1_res.request() as req:
                yield req
                duration = r1_time * self._jitter()
                yield env.timeout(duration)

            yield env.timeout(buffer * self._jitter())

            # Stage 2: Human insertion
            with h_res.request() as req:
                yield req
                duration = h_insert_time * self._jitter()
                yield env.timeout(duration)
                collector.record_human_work(duration)

            yield env.timeout(buffer * self._jitter())

            # Stage 3: R2 screwdriving
            with r2_res.request() as req:
                yield req
                duration = r2_time * self._jitter()
                yield env.timeout(duration)

            yield env.timeout(buffer * self._jitter())

            # Stage 4: Human inspection
            with h_res.request() as req:
                yield req
                duration = h_inspect_time * self._jitter()
                yield env.timeout(duration)
                collector.record_human_work(duration)

            # Record completion
            # Fatigue increases over the shift, increasing defect probability
            shift_frac = min(env.now / shift_s, 1.0)
            current_fatigue = self.fatigue_model.compute(
                schedule.r1_speed_fraction,
                schedule.r2_speed_fraction,
                schedule.human_cycle_rate_multiplier,
                shift_fraction=shift_frac,
            )
            current_defect = self.defect_model.compute(avg_speed, current_fatigue)
            is_defective = self.rng.random() < current_defect

            collector.record_unit(is_defective)
            collector.record_noise(noise)

        def unit_generator(env: simpy.Environment) -> None:
            """Generate units continuously throughout the shift."""
            unit_id = 0
            while True:
                env.process(assembly_process(env, unit_id))
                unit_id += 1
                # Inter-arrival: slightly less than bottleneck to keep pipeline full
                bottleneck = max(r1_time, h_insert_time + h_inspect_time, r2_time)
                arrival_interval = bottleneck * 0.95 * self._jitter()
                yield env.timeout(arrival_interval)

        env.process(unit_generator(env))
        env.run(until=shift_s)

        collector.total_sim_time = shift_s

        # Compute final metrics
        energy = self.energy_model.compute(
            schedule.r1_speed_fraction,
            schedule.r2_speed_fraction,
            schedule.human_cycle_rate_multiplier,
            self.cell.shift_hours,
        )

        demand = schedule.demand_target_uph
        throughput = collector.throughput_uph
        if demand > 0:
            gap = max(0.0, (demand - throughput) / demand * 100)
        else:
            gap = 0.0

        metrics = SimulationMetrics(
            throughput_uph=round(throughput, 1),
            defect_rate=round(collector.defect_rate, 4),
            noise_db=noise,
            fatigue_index=fatigue_end,
            energy_kwh=energy,
            deadline_gap_pct=round(gap, 1),
            units_produced=collector.units_completed,
            shift_hours=self.cell.shift_hours,
        )

        # --- Guard enforcement ---
        guard_events: list[GuardEvent] = []
        if self.guard_enforcer is not None:
            guard_events = self.guard_enforcer.check_and_enforce(metrics)

        return SimulationResult(
            metrics=metrics,
            collector=collector,
            demand_met=gap <= 0,
            guard_events=guard_events,
        )

    def _jitter(self, spread: float = 0.02) -> float:
        """Small random variation (±spread)."""
        return 1.0 + self.rng.uniform(-spread, spread)
