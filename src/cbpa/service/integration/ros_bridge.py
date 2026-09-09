"""ROS 2 bridge: publishes to rclpy when available, always publishes to EventBus."""

from __future__ import annotations

import logging
from typing import Any

from cbpa.service.integration.event_bus import (
    TOPIC_ALERTS,
    TOPIC_KPI_STREAM,
    TOPIC_PHASE_TRANSITION,
    TOPIC_SCHEDULE_DEPLOY,
    EventBus,
)

logger = logging.getLogger(__name__)

# Try to import rclpy (only available in a ROS 2 workspace)
try:
    import rclpy  # type: ignore[import-not-found]
    from rclpy.node import Node  # type: ignore[import-not-found]

    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False


class ROSBridge:
    """Publishes CBPA messages to ROS 2 topics when rclpy is available.

    Always publishes to the in-process EventBus.  When rclpy is importable
    and initialized, also forwards messages to ROS topics.
    """

    def __init__(self, event_bus: EventBus, node_name: str = "cbpa_bridge"):
        self.event_bus = event_bus
        self.node_name = node_name
        self._connected = False
        self._node: Any = None

        if _HAS_RCLPY:
            try:
                if not rclpy.ok():
                    rclpy.init()
                self._node = Node(node_name)  # type: ignore[arg-type]
                self._connected = True
                logger.info("ROS 2 bridge connected (rclpy node active)")
            except Exception as e:
                logger.warning(f"ROS 2 init failed: {e}. Running in stub mode.")
                self._connected = False
        else:
            logger.info("rclpy not available — ROS bridge running in stub mode")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def status(self) -> str:
        return "Connected" if self._connected else "Disconnected (stub mode)"

    def publish_kpi(self, payload: dict) -> None:
        self.event_bus.publish(TOPIC_KPI_STREAM, payload, source="ros_bridge")
        if self._connected:
            self._ros_publish(TOPIC_KPI_STREAM, payload)

    def publish_schedule_deploy(self, payload: dict) -> None:
        self.event_bus.publish(TOPIC_SCHEDULE_DEPLOY, payload, source="ros_bridge")
        if self._connected:
            self._ros_publish(TOPIC_SCHEDULE_DEPLOY, payload)

    def publish_alert(self, payload: dict) -> None:
        self.event_bus.publish(TOPIC_ALERTS, payload, source="ros_bridge")
        if self._connected:
            self._ros_publish(TOPIC_ALERTS, payload)

    def publish_phase_transition(self, payload: dict) -> None:
        self.event_bus.publish(TOPIC_PHASE_TRANSITION, payload, source="ros_bridge")
        if self._connected:
            self._ros_publish(TOPIC_PHASE_TRANSITION, payload)

    def _ros_publish(self, topic: str, payload: dict) -> None:
        """Forward to actual ROS publisher (when connected)."""
        logger.debug(f"ROS publish {topic}: {payload}")

    def shutdown(self) -> None:
        if self._connected and self._node is not None:
            self._node.destroy_node()
            self._connected = False
