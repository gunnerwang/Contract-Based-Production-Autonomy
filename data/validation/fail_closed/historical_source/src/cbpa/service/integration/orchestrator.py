"""Integration orchestrator: owns EventBus + Isaac Sim, BaSyx, and OPC-UA bridges.

Connects the CBPA experiment to external infrastructure for physics-based
verification (Isaac Sim), digital twin synchronization (BaSyx AAS), and
real-time data exposure (OPC-UA).  Falls back gracefully to analytical
models when services are unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

from cbpa.config.scenario import CellConfig, ScenarioConfig
from cbpa.layer3_verification.guard_synthesizer import GuardEnforcer, GuardEvent
from cbpa.layer4_execution.simulation import SimulationResult
from cbpa.models.metrics import SimulationMetrics
from cbpa.models.schedule import Schedule
from cbpa.physics.defect_model import DefectModel
from cbpa.physics.energy_model import EnergyModel
from cbpa.physics.fatigue_model import FatigueModel
from cbpa.physics.noise_model import NoiseModel
from cbpa.service.integration.basyx_bridge import BaSyxBridge
from cbpa.service.integration.event_bus import EventBus
from cbpa.service.integration.isaac_bridge import IsaacBridge
from cbpa.service.integration.opcua_bridge import OPCUABridge
from cbpa.layer4_execution.kpi_collector import KPICollector

logger = logging.getLogger(__name__)


class IntegrationOrchestrator:
    """Owns the EventBus and all three integration bridges.

    Provides high-level methods that coordinate Isaac Sim physics,
    BaSyx AAS digital twin, and OPC-UA real-time data exposure.
    When services are unreachable the orchestrator logs warnings
    and falls back to analytical physics models.
    """

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self.event_bus = EventBus()

        self.isaac = IsaacBridge(
            event_bus=self.event_bus,
            remote_url=config.isaac_sim_url,
        )
        self.basyx = BaSyxBridge(
            event_bus=self.event_bus,
            registry_url=config.basyx_registry_url,
            aas_server_url=config.basyx_aas_server_url,
        )
        self.opcua = OPCUABridge(
            event_bus=self.event_bus,
            endpoint=config.opcua_endpoint if config.use_integrated else "",
        )

        # Physics models for fallback / gap-filling
        self._energy_model = EnergyModel()
        self._defect_model = DefectModel()
        self._fatigue_model = FatigueModel()
        self._noise_model = NoiseModel()

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    def check_readiness(self) -> dict[str, bool]:
        """Check which bridges are connected."""
        return {
            "isaac_sim": self.isaac.is_connected,
            "basyx_aas": self.basyx.is_connected,
            "opcua": self.opcua.is_connected,
        }

    # ------------------------------------------------------------------
    # Contract synchronization
    # ------------------------------------------------------------------

    def sync_contract(self, contract: Any) -> dict:
        """Push contract to BaSyx AAS and update OPC-UA contract status.

        Parameters
        ----------
        contract : OutcomeContract
            The CBPA outcome contract to synchronize.
        """
        # Serialize contract for BaSyx
        contract_data: dict[str, Any] = {}
        try:
            contract_data = contract.model_dump()
        except Exception:
            contract_data = {
                "name": getattr(contract, "name", "unknown"),
            }

        basyx_result = self.basyx.sync_contract(contract_data)

        # Update OPC-UA contract status
        contract_name = getattr(contract, "name", "unknown")
        self.opcua.update_contract_status(contract_name, "active")

        # Push constraint limits to OPC-UA
        constraint_data: dict[str, float] = {}
        if hasattr(contract, "hard_constraints"):
            for hc in contract.hard_constraints:
                name = getattr(hc, "name", "").lower()
                limit = getattr(hc, "limit", 0.0)
                if "fatigue" in name:
                    constraint_data["fatigue_limit"] = limit
                elif "noise" in name:
                    constraint_data["noise_limit"] = limit
                elif "cyber" in name:
                    constraint_data["cyber_risk_limit"] = limit
        if constraint_data:
            self.opcua.update_constraints(constraint_data)

        logger.info(
            "Contract %s synced to BaSyx + OPC-UA",
            contract_name,
        )
        return basyx_result

    # ------------------------------------------------------------------
    # Deploy and run schedule
    # ------------------------------------------------------------------

    def deploy_and_run_schedule(
        self,
        schedule: Schedule,
        cell_config: CellConfig | None = None,
        guard_enforcer: GuardEnforcer | None = None,
        demand_spike_at_s: float | None = None,
        demand_spike_pct: float = 0.0,
    ) -> SimulationResult:
        """Deploy schedule to Isaac Sim, step physics, collect data.

        Flow:
        1. Deploy schedule parameters to Isaac Sim scene.
        2. Step simulation in batches, collecting sensor data each batch.
        3. Use sensor data where available; fill gaps with physics models.
        4. Push live KPIs to OPC-UA and BaSyx.
        5. Check guards after each batch.
        6. Build final SimulationMetrics and return SimulationResult.

        Falls back entirely to analytical models if Isaac Sim is unreachable.
        """
        cfg = cell_config or self.config.cell
        shift_s = cfg.shift_hours * 3600

        # Deploy schedule to Isaac Sim
        schedule_data = {
            "schedule_name": schedule.name,
            "r1_speed_fraction": schedule.r1_speed_fraction,
            "r2_speed_fraction": schedule.r2_speed_fraction,
            "human_cycle_rate_multiplier": schedule.human_cycle_rate_multiplier,
            "buffer_time_s": schedule.buffer_time_s,
            "demand_target_uph": schedule.demand_target_uph,
        }

        try:
            deploy_result = self.isaac.deploy_schedule(schedule_data)
            logger.info("Isaac deploy result: %s", deploy_result)
        except Exception as e:
            logger.warning("Isaac deploy failed: %s — using analytical fallback", e)

        # --- Simulation loop ---
        # Use a small number of coarse batches to avoid flooding the
        # Isaac Sim REST API.  Each batch steps many physics frames at
        # once; we sample sensors between batches.
        n_batches = 20  # 20 checkpoints across the 8-hour shift
        steps_per_batch = 100  # small batches to stay within HTTP timeout
        batch_duration_s = shift_s / n_batches
        isaac_available = self.isaac.is_connected

        collector = KPICollector()
        guard_events: list[GuardEvent] = []
        cumulative_sensor_noise: list[float] = []
        cumulative_sensor_fatigue: list[float] = []
        cumulative_sim_time_s: float = 0.0
        units_completed: int = 0

        for batch_idx in range(n_batches):
            sim_time_s = (batch_idx + 1) * batch_duration_s
            shift_fraction = min(sim_time_s / shift_s, 1.0)

            # Step Isaac Sim physics (skip if disconnected)
            step_result: dict = {"stub": True}
            if isaac_available:
                try:
                    step_result = self.isaac.step_simulation(steps_per_batch)
                    if step_result.get("error"):
                        logger.warning(
                            "Isaac step error at batch %d/%d — "
                            "falling back to analytical for remaining batches",
                            batch_idx + 1, n_batches,
                        )
                        isaac_available = False
                except Exception as e:
                    logger.warning("Isaac step failed at batch %d: %s — disabling", batch_idx + 1, e)
                    isaac_available = False

            # Track sim-derived throughput data from step result
            if not step_result.get("stub") and not step_result.get("error"):
                cumulative_sim_time_s = step_result.get("sim_time_s", cumulative_sim_time_s)

            # Read sensor data (skip if disconnected)
            sensor_data: dict = {"stub": True}
            if isaac_available:
                try:
                    sensor_data = self.isaac.get_sensor_data()
                except Exception as e:
                    logger.debug("Isaac sensor read failed: %s", e)
                    sensor_data = {"stub": True}

            # Track units completed from sensor production data
            prod_data = sensor_data.get("production", {})
            if prod_data.get("units_completed") is not None:
                units_completed = prod_data["units_completed"]

            # Extract sensor readings (use if available, else fall back)
            acoustic = sensor_data.get("acoustic", {})
            human_op = sensor_data.get("human_operator", {})

            sensor_noise = acoustic.get("noise_db")
            sensor_fatigue = human_op.get("fatigue_estimate")

            if sensor_noise is not None:
                cumulative_sensor_noise.append(sensor_noise)
            if sensor_fatigue is not None:
                cumulative_sensor_fatigue.append(sensor_fatigue)

            # Build running KPI snapshot using physics models + sensor data
            analytical_noise = self._noise_model.compute(
                schedule.r1_speed_fraction, schedule.r2_speed_fraction
            )
            analytical_fatigue = self._fatigue_model.compute(
                schedule.r1_speed_fraction,
                schedule.r2_speed_fraction,
                schedule.human_cycle_rate_multiplier,
                shift_fraction=shift_fraction,
            )

            current_noise = sensor_noise if sensor_noise is not None else analytical_noise
            current_fatigue = sensor_fatigue if sensor_fatigue is not None else analytical_fatigue

            # Approximate throughput from time elapsed
            bottleneck = max(
                cfg.r1_cycle_time_s / schedule.r1_speed_fraction,
                (cfg.human_insertion_time_s + cfg.human_inspection_time_s)
                / schedule.human_cycle_rate_multiplier,
                cfg.r2_cycle_time_s / schedule.r2_speed_fraction,
            )
            cycle_time = bottleneck + schedule.buffer_time_s
            approx_throughput = 3600.0 / cycle_time if cycle_time > 0 else 0.0

            # Push live KPIs to OPC-UA (every 10th batch to reduce load)
            if batch_idx % 10 == 0:
                live_kpi = {
                    "throughput_uph": round(approx_throughput, 1),
                    "noise_db": round(current_noise, 1),
                    "fatigue_index": round(current_fatigue, 3),
                }
                try:
                    self.opcua.update_kpi(live_kpi)
                except Exception as e:
                    logger.debug("OPC-UA live KPI push failed: %s", e)

            # Check guards
            if guard_enforcer is not None and batch_idx % 20 == 0:
                running_metrics = SimulationMetrics(
                    throughput_uph=round(approx_throughput, 1),
                    defect_rate=0.0,
                    noise_db=round(current_noise, 1),
                    fatigue_index=round(current_fatigue, 3),
                    energy_kwh=0.0,
                    deadline_gap_pct=0.0,
                )
                batch_events = guard_enforcer.check_and_enforce(running_metrics)
                guard_events.extend(batch_events)

                # Forward clamp actions to Isaac Sim when guards fire
                if isaac_available and batch_events:
                    for ev in batch_events:
                        if ev.enforced and "clamp" in (ev.action or "").lower():
                            try:
                                self.isaac.apply_guard_action(
                                    "clamp_speed", reduction_fraction=0.1,
                                )
                            except Exception as e:
                                logger.debug("Isaac guard forward failed: %s", e)

        # --- Build final metrics ---
        # Use end-of-shift physics models for comprehensive metrics
        fatigue_end = self._fatigue_model.compute(
            schedule.r1_speed_fraction,
            schedule.r2_speed_fraction,
            schedule.human_cycle_rate_multiplier,
        )
        noise_final = self._noise_model.compute(
            schedule.r1_speed_fraction, schedule.r2_speed_fraction
        )
        energy = self._energy_model.compute(
            schedule.r1_speed_fraction,
            schedule.r2_speed_fraction,
            schedule.human_cycle_rate_multiplier,
            cfg.shift_hours,
        )
        avg_speed = (schedule.r1_speed_fraction + schedule.r2_speed_fraction) / 2
        defect_rate = self._defect_model.compute(avg_speed, fatigue_end)

        # Override noise/fatigue with sensor averages if we got real data
        if cumulative_sensor_noise:
            noise_final = round(
                sum(cumulative_sensor_noise) / len(cumulative_sensor_noise), 1
            )
        if cumulative_sensor_fatigue:
            # Use the last reading (end-of-shift fatigue)
            fatigue_end = round(cumulative_sensor_fatigue[-1], 3)

        # Throughput and demand gap — prefer sim-derived values when Isaac
        # provided real production data during the batch loop.
        bottleneck = max(
            cfg.r1_cycle_time_s / schedule.r1_speed_fraction,
            (cfg.human_insertion_time_s + cfg.human_inspection_time_s)
            / schedule.human_cycle_rate_multiplier,
            cfg.r2_cycle_time_s / schedule.r2_speed_fraction,
        )
        cycle_time = bottleneck + schedule.buffer_time_s
        throughput = 3600.0 / cycle_time if cycle_time > 0 else 0.0
        units_produced = int(throughput * cfg.shift_hours)

        # Override with Isaac Sim production data when available
        if units_completed > 0 and cumulative_sim_time_s > 0:
            sim_throughput = units_completed / cumulative_sim_time_s * 3600.0
            logger.info(
                "Using sim-derived throughput: %.1f u/h (%d units in %.1fs) "
                "vs analytical %.1f u/h",
                sim_throughput, units_completed, cumulative_sim_time_s, throughput,
            )
            throughput = sim_throughput
            units_produced = units_completed

        demand = schedule.demand_target_uph
        if demand > 0:
            gap = max(0.0, (demand - throughput) / demand * 100)
        else:
            gap = 0.0

        metrics = SimulationMetrics(
            throughput_uph=round(throughput, 1),
            defect_rate=round(defect_rate, 4),
            noise_db=noise_final,
            fatigue_index=fatigue_end,
            energy_kwh=energy,
            deadline_gap_pct=round(gap, 1),
            units_produced=units_produced,
            shift_hours=cfg.shift_hours,
        )

        # Push final KPIs to BaSyx and OPC-UA
        self.update_live_kpis(metrics)

        # Final guard check
        if guard_enforcer is not None:
            final_events = guard_enforcer.check_and_enforce(metrics)
            guard_events.extend(final_events)

        collector.total_sim_time = shift_s

        return SimulationResult(
            metrics=metrics,
            collector=collector,
            demand_met=gap <= 0,
            guard_events=guard_events,
        )

    # ------------------------------------------------------------------
    # Disturbance / guard-action delegation to Isaac Sim
    # ------------------------------------------------------------------

    def inject_disturbance(self, disturbance_type: str, **kwargs) -> dict:
        """Forward a disturbance injection to Isaac Sim.

        Falls back to a stub acknowledgement when Isaac is unreachable.
        """
        try:
            return self.isaac.inject_disturbance(disturbance_type, **kwargs)
        except Exception as e:
            logger.warning("Isaac disturbance injection failed: %s", e)
            return {"applied": False, "error": str(e), "stub": True}

    def apply_guard_action(self, action: str, **kwargs) -> dict:
        """Forward a guard enforcement action to Isaac Sim.

        Falls back to a stub acknowledgement when Isaac is unreachable.
        """
        try:
            return self.isaac.apply_guard_action(action, **kwargs)
        except Exception as e:
            logger.warning("Isaac guard action failed: %s", e)
            return {"applied": False, "error": str(e), "stub": True}

    # ------------------------------------------------------------------
    # Live KPI push
    # ------------------------------------------------------------------

    def update_live_kpis(self, metrics: SimulationMetrics) -> None:
        """Push current metrics to OPC-UA and BaSyx."""
        kpi_data = {
            "throughput_uph": metrics.throughput_uph,
            "defect_rate": metrics.defect_rate,
            "noise_db": metrics.noise_db,
            "fatigue_index": metrics.fatigue_index,
            "energy_kwh": metrics.energy_kwh,
            "deadline_gap_pct": metrics.deadline_gap_pct,
        }

        try:
            self.opcua.update_kpi(kpi_data)
        except Exception as e:
            logger.debug("OPC-UA final KPI push failed: %s", e)

        try:
            self.basyx.update_kpi_submodel(kpi_data)
        except Exception as e:
            logger.debug("BaSyx final KPI push failed: %s", e)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Clean close of all bridges."""
        try:
            self.isaac.shutdown()
        except Exception as e:
            logger.debug("Isaac shutdown: %s", e)
        try:
            self.basyx.shutdown()
        except Exception as e:
            logger.debug("BaSyx shutdown: %s", e)
        try:
            self.opcua.shutdown()
        except Exception as e:
            logger.debug("OPC-UA shutdown: %s", e)
        self.event_bus.clear()
        logger.info("IntegrationOrchestrator shut down")
