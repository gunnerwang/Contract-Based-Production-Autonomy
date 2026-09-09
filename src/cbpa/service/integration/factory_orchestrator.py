"""Factory-level integration orchestrator for multi-cell CBPA.

Extends the integration infrastructure with per-cell deployment,
multi-cell sensor aggregation, and factory-level KPI push.  Uses the
existing Isaac Sim multi-cell endpoints (/scene/cellA/deploy,
/scene/cellB/deploy, /scene/factory/sensors) provided by --multi-cell scene.
"""

from __future__ import annotations

import logging
from typing import Any

from cbpa.config.scenario import FactoryScenarioConfig
from cbpa.models.metrics import FactoryMetrics
from cbpa.models.schedule import FactorySchedule
from cbpa.service.integration.basyx_bridge import BaSyxBridge
from cbpa.service.integration.event_bus import (
    EventBus,
    TOPIC_ISAAC_SCENE_COMMAND,
    TOPIC_ISAAC_SENSOR_DATA,
)
from cbpa.service.integration.isaac_bridge import IsaacBridge
from cbpa.service.integration.opcua_bridge import OPCUABridge

logger = logging.getLogger(__name__)


class FactoryOrchestrator:
    """Multi-cell integration orchestrator.

    Uses a single IsaacBridge (shared HTTP client) but hits cell-specific
    REST endpoints for deploy and sensor reads.  BaSyx and OPC-UA are
    shared across the factory.
    """

    def __init__(self, config: FactoryScenarioConfig) -> None:
        self.config = config
        self.event_bus = EventBus()

        # Single Isaac bridge — we reuse its HTTP client for cell-specific calls
        self.isaac = IsaacBridge(
            event_bus=self.event_bus,
            remote_url=config.isaac_sim_url,
            scene_path="/World/Factory",
        )

        self.basyx = BaSyxBridge(
            event_bus=self.event_bus,
            registry_url=config.basyx_registry_url,
            aas_server_url=config.basyx_aas_server_url,
            aas_id="cbpa_factory_001",
        )
        self.opcua = OPCUABridge(
            event_bus=self.event_bus,
            endpoint=config.opcua_endpoint if config.use_integrated else "",
        )

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    def check_readiness(self) -> dict[str, bool]:
        return {
            "isaac_sim": self.isaac.is_connected,
            "basyx_aas": self.basyx.is_connected,
            "opcua": self.opcua.is_connected,
        }

    # ------------------------------------------------------------------
    # Contract sync
    # ------------------------------------------------------------------

    def sync_contract(self, contract: Any) -> dict:
        """Push factory contract to BaSyx AAS and OPC-UA."""
        contract_data: dict[str, Any] = {}
        try:
            contract_data = contract.model_dump()
        except Exception:
            contract_data = {"name": getattr(contract, "name", "unknown")}

        basyx_result = self.basyx.sync_contract(contract_data)

        contract_name = getattr(contract, "name", "unknown")
        self.opcua.update_contract_status(contract_name, "active")

        constraint_data: dict[str, float] = {}
        if hasattr(contract, "hard_constraints"):
            for hc in contract.hard_constraints:
                name = getattr(hc, "name", "").lower()
                limit = getattr(hc, "limit", 0.0)
                if "fatigue" in name:
                    constraint_data.setdefault("fatigue_limit", limit)
                elif "noise" in name:
                    constraint_data["noise_limit"] = limit
                elif "cyber" in name:
                    constraint_data["cyber_risk_limit"] = limit
        if constraint_data:
            self.opcua.update_constraints(constraint_data)

        logger.info("Factory contract %s synced", contract_name)
        return basyx_result

    def sync_factory_schedule(self, factory_schedule: FactorySchedule) -> dict:
        """Push factory schedule details (routing, assignments) to BaSyx."""
        schedule_data = {
            "name": factory_schedule.name,
            "variant_routing": factory_schedule.variant_routing,
            "operator_assignments": factory_schedule.operator_assignments,
            "cell_schedules": {
                cell_id: {
                    "r1_speed_fraction": cs.r1_speed_fraction,
                    "r2_speed_fraction": cs.r2_speed_fraction,
                    "human_cycle_rate_multiplier": cs.human_cycle_rate_multiplier,
                    "buffer_time_s": cs.buffer_time_s,
                    "assigned_operator": cs.assigned_operator,
                    "assigned_variants": cs.assigned_variants,
                }
                for cell_id, cs in factory_schedule.cell_schedules.items()
            },
        }
        try:
            return self.basyx.update_kpi_submodel(schedule_data)
        except Exception as e:
            logger.debug("BaSyx factory schedule sync failed: %s", e)
            return {"stub": True}

    # ------------------------------------------------------------------
    # Isaac Sim: cell-specific HTTP helpers
    # ------------------------------------------------------------------

    def _isaac_post(self, path: str, json_data: dict) -> dict:
        """POST to a specific Isaac Sim endpoint via the bridge's client."""
        self.event_bus.publish(
            TOPIC_ISAAC_SCENE_COMMAND,
            {"endpoint": path, **json_data},
            source="factory_orchestrator",
        )
        client = getattr(self.isaac, "_http_client", None)
        if client is not None:
            try:
                resp = client.post(path, json=json_data)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.warning("Isaac POST %s failed: %s", path, e)
        return {"stub": True, "error": "no client"}

    def _isaac_get(self, path: str) -> dict:
        """GET from a specific Isaac Sim endpoint via the bridge's client."""
        client = getattr(self.isaac, "_http_client", None)
        if client is not None:
            try:
                resp = client.get(path)
                resp.raise_for_status()
                result = resp.json()
                self.event_bus.publish(
                    TOPIC_ISAAC_SENSOR_DATA,
                    {"endpoint": path, **result},
                    source="factory_orchestrator",
                )
                return result
            except Exception as e:
                logger.debug("Isaac GET %s failed: %s", path, e)
        return {"stub": True}

    # ------------------------------------------------------------------
    # Deploy factory schedule (both cells)
    # ------------------------------------------------------------------

    def deploy_factory_schedule(self, factory_schedule: FactorySchedule) -> dict:
        """Deploy per-cell schedules to Isaac Sim Cell A and Cell B."""
        results = {}

        for cell_id, cell_sched in factory_schedule.cell_schedules.items():
            schedule_data = {
                "schedule_name": f"{factory_schedule.name}_{cell_id}",
                "r1_speed_fraction": cell_sched.r1_speed_fraction,
                "r2_speed_fraction": cell_sched.r2_speed_fraction,
                "human_cycle_rate_multiplier": cell_sched.human_cycle_rate_multiplier,
                "buffer_time_s": cell_sched.buffer_time_s,
                "demand_target_uph": cell_sched.demand_target_uph,
            }
            endpoint = f"/scene/cell{cell_id}/deploy"
            result = self._isaac_post(endpoint, schedule_data)
            results[cell_id] = result
            logger.info("Cell %s deploy: %s", cell_id, result)

        return results

    # ------------------------------------------------------------------
    # Read factory sensors
    # ------------------------------------------------------------------

    def get_factory_sensors(self) -> dict:
        """Read aggregated factory sensor data from Isaac Sim."""
        result = self._isaac_get("/scene/factory/sensors")
        if not result.get("stub"):
            return result

        # Fallback: read per-cell
        return {
            "cellA": self._isaac_get("/scene/cellA/sensors"),
            "cellB": self._isaac_get("/scene/cellB/sensors"),
            "stub": True,
        }

    def get_agv_status(self) -> dict:
        """Read AGV status from Isaac Sim."""
        return self._isaac_get("/scene/agv/status")

    # ------------------------------------------------------------------
    # Step simulation
    # ------------------------------------------------------------------

    def step_simulation(self, num_steps: int = 100) -> dict:
        """Step the factory simulation (both cells advance together)."""
        return self.isaac.step_simulation(num_steps)

    # ------------------------------------------------------------------
    # Disturbance injection
    # ------------------------------------------------------------------

    def inject_disturbance(self, disturbance_type: str, **kwargs) -> dict:
        """Inject a disturbance into Isaac Sim."""
        return self.isaac.inject_disturbance(disturbance_type, **kwargs)

    # ------------------------------------------------------------------
    # Live KPI push (factory-level)
    # ------------------------------------------------------------------

    def update_factory_kpis(self, metrics: FactoryMetrics) -> None:
        """Push factory-level KPIs to OPC-UA and BaSyx."""
        kpi_data = {
            # Standard KPIs (compatible with single-cell consumers)
            "throughput_uph": metrics.total_throughput_uph,
            "defect_rate": metrics.factory_defect_rate,
            "noise_db": metrics.factory_noise_db,
            "fatigue_index": max(metrics.operator_fatigue.values()) if metrics.operator_fatigue else 0.0,
            "energy_kwh": metrics.factory_energy_kwh,
            "deadline_gap_pct": 0.0,
            # Factory-specific KPIs (OPC-UA nodes added for factory mode)
            "factory_noise_db": metrics.factory_noise_db,
            "cell_balance_loss_pct": metrics.cell_balance_loss_pct,
            "agv_utilization": metrics.agv_utilization,
            "fatigue_h1": metrics.operator_fatigue.get("H1", 0.0),
            "fatigue_h2": metrics.operator_fatigue.get("H2", 0.0),
            "fatigue_h3": metrics.operator_fatigue.get("H3", 0.0),
        }

        try:
            self.opcua.update_kpi(kpi_data)
        except Exception as e:
            logger.debug("OPC-UA factory KPI push failed: %s", e)
        try:
            self.basyx.update_kpi_submodel(kpi_data)
        except Exception as e:
            logger.debug("BaSyx factory KPI push failed: %s", e)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        for bridge in [self.isaac, self.basyx, self.opcua]:
            try:
                bridge.shutdown()
            except Exception as e:
                logger.debug("Bridge shutdown: %s", e)
        self.event_bus.clear()
        logger.info("FactoryOrchestrator shut down")
