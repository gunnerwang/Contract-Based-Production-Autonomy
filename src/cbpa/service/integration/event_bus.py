"""In-process pub/sub event bus.

Topic names map 1:1 to future ROS 2 topic names so the migration path
is: EventBus.publish() -> rclpy publisher.publish().
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Standard topic names (match planned ROS 2 topic layout)
TOPIC_KPI_STREAM = "/cbpa/kpi_stream"
TOPIC_SCHEDULE_DEPLOY = "/cbpa/schedule_deploy"
TOPIC_ALERTS = "/cbpa/alerts"
TOPIC_PHASE_TRANSITION = "/cbpa/phase_transition"
TOPIC_CONSTRAINT_STATUS = "/cbpa/constraint_status"
TOPIC_PRODUCTION_ORDER = "/cbpa/production_order"
TOPIC_SCHEDULE_UPLOAD = "/cbpa/schedule_upload"
TOPIC_KPI_REPORT = "/cbpa/kpi_report"

# AAS / BaSyx topics
TOPIC_AAS_CONTRACT_SYNC = "/cbpa/aas/contract_sync"
TOPIC_AAS_KPI_SUBMODEL = "/cbpa/aas/kpi_submodel"

# OPC-UA topics
TOPIC_OPCUA_KPI_UPDATE = "/cbpa/opcua/kpi_update"
TOPIC_OPCUA_CONSTRAINT_UPDATE = "/cbpa/opcua/constraint_update"

# Isaac Sim topics
TOPIC_ISAAC_SCENE_COMMAND = "/cbpa/isaac/scene_command"
TOPIC_ISAAC_SENSOR_DATA = "/cbpa/isaac/sensor_data"


@dataclass
class BusMessage:
    """Wrapper around any payload published on the bus."""

    topic: str
    payload: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "cbpa"


Callback = Callable[[BusMessage], None]


class EventBus:
    """Thread-safe in-process publish/subscribe bus.

    All integration messages flow through here.  UI subscribes to topics.
    ROS/MES bridges also subscribe and optionally forward to external systems.
    """

    def __init__(self, max_history: int = 200) -> None:
        self._subscribers: dict[str, list[Callback]] = defaultdict(list)
        self._history: list[BusMessage] = []
        self._max_history = max_history
        self._lock = threading.Lock()

    def publish(self, topic: str, payload: Any, source: str = "cbpa") -> BusMessage:
        msg = BusMessage(topic=topic, payload=payload, source=source)
        with self._lock:
            self._history.append(msg)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            subs = list(self._subscribers.get(topic, []))

        for cb in subs:
            try:
                cb(msg)
            except Exception:
                logger.exception(f"Subscriber error on topic {topic}")

        return msg

    def subscribe(self, topic: str, callback: Callback) -> None:
        with self._lock:
            self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callback) -> None:
        with self._lock:
            subs = self._subscribers.get(topic, [])
            if callback in subs:
                subs.remove(callback)

    def get_history(self, topic: str | None = None, limit: int = 50) -> list[BusMessage]:
        with self._lock:
            msgs = list(self._history)
        if topic is not None:
            msgs = [m for m in msgs if m.topic == topic]
        return msgs[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._subscribers.clear()
            self._history.clear()
