"""MES REST bridge: uses httpx for async REST calls to MES endpoints."""

from __future__ import annotations

import logging
from typing import Any

from cbpa.service.integration.event_bus import (
    TOPIC_KPI_REPORT,
    TOPIC_PRODUCTION_ORDER,
    TOPIC_SCHEDULE_UPLOAD,
    EventBus,
)

logger = logging.getLogger(__name__)

# Try httpx — optional dependency
try:
    import httpx

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False


class MESBridge:
    """REST bridge to a Manufacturing Execution System (MES).

    When ``base_url`` is provided and httpx is available, makes real HTTP
    requests.  Otherwise operates in stub mode (messages go to EventBus only).
    """

    def __init__(
        self,
        event_bus: EventBus,
        base_url: str = "",
        timeout: float = 10.0,
    ):
        self.event_bus = event_bus
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.timeout = timeout
        self._connected = False
        self._client: Any = None

        if self.base_url and _HAS_HTTPX:
            self._client = httpx.Client(
                base_url=self.base_url, timeout=timeout
            )
            self._connected = True
            logger.info(f"MES bridge connected to {self.base_url}")
        else:
            reason = "no base_url" if not self.base_url else "httpx not installed"
            logger.info(f"MES bridge stub mode ({reason})")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def status(self) -> str:
        if self._connected:
            return f"Connected ({self.base_url})"
        return "Disconnected (stub mode)"

    def fetch_production_order(self, order_id: str = "default") -> dict:
        """GET /orders/{order_id} from MES."""
        self.event_bus.publish(
            TOPIC_PRODUCTION_ORDER,
            {"order_id": order_id, "action": "fetch"},
            source="mes_bridge",
        )

        if self._connected and self._client is not None:
            try:
                resp = self._client.get(f"/orders/{order_id}")
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.warning(f"MES fetch order failed: {e}")

        # Stub response
        return {
            "order_id": order_id,
            "product_variant": "V_A",
            "quantity": 416,
            "demand_uph": 52.0,
            "priority": "normal",
        }

    def upload_schedule(self, schedule_data: dict) -> dict:
        """POST /schedules to MES."""
        self.event_bus.publish(
            TOPIC_SCHEDULE_UPLOAD,
            schedule_data,
            source="mes_bridge",
        )

        if self._connected and self._client is not None:
            try:
                resp = self._client.post("/schedules", json=schedule_data)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.warning(f"MES schedule upload failed: {e}")

        # Stub acknowledgement
        return {
            "schedule_name": schedule_data.get("schedule_name", "unknown"),
            "accepted": True,
            "mes_order_id": "MES-STUB-001",
        }

    def report_kpi(self, kpi_data: dict) -> dict:
        """POST /kpi-reports to MES."""
        self.event_bus.publish(
            TOPIC_KPI_REPORT,
            kpi_data,
            source="mes_bridge",
        )

        if self._connected and self._client is not None:
            try:
                resp = self._client.post("/kpi-reports", json=kpi_data)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.warning(f"MES KPI report failed: {e}")

        return {"status": "acknowledged", "stub": True}

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._connected = False
