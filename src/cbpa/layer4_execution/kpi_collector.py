"""Real-time KPI aggregation during simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KPICollector:
    """Collects KPIs during a SimPy simulation run."""

    units_completed: int = 0
    defects: int = 0
    total_human_work_time: float = 0.0
    total_sim_time: float = 0.0
    energy_accumulated: float = 0.0
    noise_samples: list[float] = field(default_factory=list)

    def record_unit(self, is_defective: bool = False) -> None:
        self.units_completed += 1
        if is_defective:
            self.defects += 1

    def record_human_work(self, duration: float) -> None:
        self.total_human_work_time += duration

    def record_noise(self, noise_db: float) -> None:
        self.noise_samples.append(noise_db)

    @property
    def defect_rate(self) -> float:
        if self.units_completed == 0:
            return 0.0
        return self.defects / self.units_completed

    @property
    def throughput_uph(self) -> float:
        if self.total_sim_time <= 0:
            return 0.0
        hours = self.total_sim_time / 3600.0
        return self.units_completed / hours

    @property
    def avg_noise(self) -> float:
        if not self.noise_samples:
            return 0.0
        return sum(self.noise_samples) / len(self.noise_samples)

    @property
    def human_utilization(self) -> float:
        if self.total_sim_time <= 0:
            return 0.0
        return min(self.total_human_work_time / self.total_sim_time, 1.0)
