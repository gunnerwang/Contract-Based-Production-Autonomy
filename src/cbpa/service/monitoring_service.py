"""Monitoring service: KPI streaming and alert management."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from cbpa.layer4_execution.monitor import ContractMonitor, MonitorAlert
from cbpa.models.contract import OutcomeContract
from cbpa.models.metrics import SimulationMetrics


@dataclass
class KPISnapshot:
    """A timestamped KPI reading for live monitoring."""

    timestamp: float
    throughput_uph: float = 0.0
    defect_rate: float = 0.0
    noise_db: float = 0.0
    fatigue_index: float = 0.0
    energy_kwh: float = 0.0
    deadline_gap_pct: float = 0.0
    active_phase: int = 0
    active_layer: str = ""


class MonitoringService:
    """Wraps ContractMonitor for live KPI streaming and alerts."""

    def __init__(self, contract: OutcomeContract | None = None):
        self._contract = contract
        self._monitor: ContractMonitor | None = None
        self._alerts: list[MonitorAlert] = []
        self._snapshots: list[KPISnapshot] = []
        if contract is not None:
            self._monitor = ContractMonitor(contract)

    def set_contract(self, contract: OutcomeContract) -> None:
        self._contract = contract
        self._monitor = ContractMonitor(contract)

    def push_snapshot(self, metrics: SimulationMetrics, phase: int = 0, layer: str = "") -> list[MonitorAlert]:
        """Push a new KPI snapshot and check for alerts."""
        snap = KPISnapshot(
            timestamp=time.time(),
            throughput_uph=metrics.throughput_uph,
            defect_rate=metrics.defect_rate,
            noise_db=metrics.noise_db,
            fatigue_index=metrics.fatigue_index,
            energy_kwh=metrics.energy_kwh,
            deadline_gap_pct=metrics.deadline_gap_pct,
            active_phase=phase,
            active_layer=layer,
        )
        self._snapshots.append(snap)

        alerts: list[MonitorAlert] = []
        if self._monitor is not None:
            alerts = self._monitor.check(metrics)
            self._alerts.extend(alerts)

        return alerts

    def get_alerts(self, severity: str | None = None) -> list[MonitorAlert]:
        if severity is None:
            return list(self._alerts)
        return [a for a in self._alerts if a.severity == severity]

    def get_latest_snapshot(self) -> KPISnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    def get_snapshots(self) -> list[KPISnapshot]:
        return list(self._snapshots)

    def stream_simulation(
        self,
        metrics_sequence: list[SimulationMetrics],
        phase_sequence: list[int] | None = None,
    ) -> Iterator[KPISnapshot]:
        """Yield KPISnapshots from a pre-computed metrics sequence.

        This simulates a real-time stream for the Live Monitor page.
        """
        phases = phase_sequence or [0] * len(metrics_sequence)
        for i, metrics in enumerate(metrics_sequence):
            phase = phases[i] if i < len(phases) else 0
            self.push_snapshot(metrics, phase=phase)
            yield self._snapshots[-1]

    def get_shift_summary(self) -> dict:
        """Aggregated statistics across all snapshots for auditor review."""
        if not self._snapshots:
            return {}
        return {
            "total_snapshots": len(self._snapshots),
            "total_alerts": len(self._alerts),
            "violations": len([a for a in self._alerts if a.severity == "violation"]),
            "warnings": len([a for a in self._alerts if a.severity == "warning"]),
            "peak_fatigue": max(s.fatigue_index for s in self._snapshots),
            "peak_noise": max(s.noise_db for s in self._snapshots),
            "avg_throughput": sum(s.throughput_uph for s in self._snapshots) / len(self._snapshots),
        }

    def clear(self) -> None:
        self._alerts.clear()
        self._snapshots.clear()
        if self._monitor is not None:
            self._monitor.alerts.clear()
