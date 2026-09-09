"""OPC-UA bridge: exposes CBPA KPIs and contract status via OPC-UA server.

When ``asyncua`` is installed and a server endpoint is configured, this
bridge runs an OPC-UA server in a background thread that PLC/SCADA systems
can subscribe to.  Otherwise operates in stub mode (messages go to EventBus only).

Connect any OPC-UA client (e.g. UaExpert, Prosys OPC UA Browser) to:
    opc.tcp://localhost:4840/cbpa/
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import threading
from typing import Any
from urllib.parse import urlparse


def _get_ppid(pid: int) -> int | None:
    """Return the parent PID of *pid*, or None if unreadable."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None

from cbpa.service.integration.event_bus import (
    TOPIC_OPCUA_CONSTRAINT_UPDATE,
    TOPIC_OPCUA_KPI_UPDATE,
    EventBus,
)

logger = logging.getLogger(__name__)

# Try to import asyncua (python-opcua async variant)
try:
    from asyncua import Server as OPCUAServer  # type: ignore[import-not-found]

    _HAS_ASYNCUA = True
except ImportError:
    _HAS_ASYNCUA = False


# OPC-UA namespace and node layout
OPCUA_NAMESPACE = "urn:cbpa:production"
OPCUA_NODES = {
    # Single-cell KPIs
    "throughput_uph": "CBPA.KPI.Throughput",
    "defect_rate": "CBPA.KPI.DefectRate",
    "noise_db": "CBPA.KPI.NoiseDB",
    "fatigue_index": "CBPA.KPI.FatigueIndex",
    "energy_kwh": "CBPA.KPI.EnergyKWh",
    "deadline_gap_pct": "CBPA.KPI.DeadlineGapPct",
    # Factory KPIs (multi-cell extension)
    "cell_balance_loss_pct": "CBPA.KPI.CellBalanceLoss",
    "agv_utilization": "CBPA.KPI.AGVUtilization",
    "fatigue_h1": "CBPA.KPI.FatigueH1",
    "fatigue_h2": "CBPA.KPI.FatigueH2",
    "fatigue_h3": "CBPA.KPI.FatigueH3",
    "factory_noise_db": "CBPA.KPI.FactoryNoise",
    # Constraints
    "fatigue_limit": "CBPA.Constraint.FatigueLimit",
    "noise_limit": "CBPA.Constraint.NoiseLimit",
    "cyber_risk_limit": "CBPA.Constraint.CyberRiskLimit",
    # Contract
    "contract_name": "CBPA.Contract.Name",
    "contract_status": "CBPA.Contract.Status",
}


class OPCUABridge:
    """OPC-UA server bridge exposing CBPA data to industrial systems.

    Runs the server in a background thread so it doesn't block Streamlit.
    Any OPC-UA client can connect and browse/subscribe to the CBPA nodes.
    """

    # Class-level reference to the last active instance so we can shut
    # it down before starting a new one (avoids port conflicts).
    _active_instance = None  # type: OPCUABridge | None

    def _shutdown_class_singleton(self) -> None:
        """Shut down any previously active bridge in this process."""
        prev = OPCUABridge._active_instance
        if prev is not None and prev is not self:
            try:
                prev.shutdown()
            except Exception:
                pass
        OPCUABridge._active_instance = self

    def __init__(
        self,
        event_bus: EventBus,
        endpoint: str = "",
        server_name: str = "CBPA OPC-UA Server",
    ):
        self.event_bus = event_bus
        self.endpoint = endpoint or "opc.tcp://0.0.0.0:4840/cbpa/"
        self.server_name = server_name
        self._connected = False
        self._server: Any = None
        self._nodes: dict[str, Any] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

        if _HAS_ASYNCUA and endpoint:
            # Shut down any in-process server from a previous bridge instance,
            # then kill external stale processes on the port.
            self._shutdown_class_singleton()
            self._kill_stale_port()
            for attempt in range(3):
                try:
                    self._start_server_thread()
                    if self._connected:
                        break
                except Exception as e:
                    logger.warning(f"OPC-UA init attempt {attempt+1} failed: {e}")
                    self._connected = False
                if attempt < 2:
                    self._kill_stale_port()
                    import time
                    time.sleep(1)
            if not self._connected:
                logger.warning("OPC-UA server could not start. Running in stub mode.")
        else:
            reason = "no endpoint" if not endpoint else "asyncua not installed"
            logger.info(f"OPC-UA bridge stub mode ({reason})")

    def _kill_stale_port(self) -> None:
        """Kill only *our own* stale child processes on the OPC-UA port.

        Previous versions would SIGKILL any process on the port, which
        killed the experiment when Streamlit and the experiment both
        initialised an OPCUABridge on port 4840.  Now we only kill
        processes whose parent is *this* process (orphaned server threads
        from a previous bridge instance).
        """
        try:
            parsed = urlparse(self.endpoint)
            port = parsed.port or 4840
            my_pid = os.getpid()
            # Only look for LISTENING servers, not clients
            result = subprocess.run(
                ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip()
            if pids:
                for pid_str in pids.splitlines():
                    try:
                        pid = int(pid_str)
                        if pid == my_pid:
                            continue
                        # Only kill if it's a child of this process
                        ppid = _get_ppid(pid)
                        if ppid == my_pid:
                            logger.info(f"Killing own stale child {pid} on port {port}")
                            os.kill(pid, signal.SIGKILL)
                        else:
                            logger.debug(
                                f"Port {port} held by PID {pid} (ppid={ppid}); "
                                f"not our child — skipping"
                            )
                    except (ValueError, OSError):
                        pass
                import time
                time.sleep(0.5)
        except Exception as e:
            logger.debug(f"Port cleanup check: {e}")

    def _start_server_thread(self) -> None:
        """Start OPC-UA server in a background daemon thread."""
        self._loop = asyncio.new_event_loop()
        ready = threading.Event()
        error_holder: list[Exception] = []

        def _run() -> None:
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._setup_and_start(ready))
                self._loop.run_forever()
            except Exception as e:
                error_holder.append(e)

        self._thread = threading.Thread(target=_run, daemon=True, name="opcua-server")
        self._thread.start()
        # Wait for server to be ready (max 10s)
        if ready.wait(timeout=10.0):
            self._connected = True
            logger.info(f"OPC-UA server running at {self.endpoint}")
        elif error_holder and "address already in use" in str(error_holder[0]):
            # Port occupied — kill stale process and retry once
            logger.warning("Port in use, killing stale process and retrying...")
            self._kill_stale_port()
            self._loop = asyncio.new_event_loop()
            ready = threading.Event()
            self._thread = threading.Thread(target=_run, daemon=True, name="opcua-server")
            self._thread.start()
            if ready.wait(timeout=10.0):
                self._connected = True
                logger.info(f"OPC-UA server running at {self.endpoint} (after retry)")
            else:
                logger.warning("OPC-UA server startup timed out after retry")
        else:
            logger.warning("OPC-UA server startup timed out")

    async def _setup_and_start(self, ready: threading.Event) -> None:
        """Async setup: create server, nodes, and start listening."""
        self._server = OPCUAServer()
        await self._server.init()
        self._server.set_endpoint(self.endpoint)
        self._server.set_server_name(self.server_name)

        # Register namespace
        idx = await self._server.register_namespace(OPCUA_NAMESPACE)

        # Create folder structure: CBPA / {KPI, Constraints, Contract}
        cbpa_folder = await self._server.nodes.objects.add_folder(idx, "CBPA")
        kpi_folder = await cbpa_folder.add_folder(idx, "KPI")
        constraint_folder = await cbpa_folder.add_folder(idx, "Constraints")
        contract_folder = await cbpa_folder.add_folder(idx, "Contract")

        # KPI variables (float) — single-cell + factory
        kpi_keys = [
            "throughput_uph", "defect_rate", "noise_db",
            "fatigue_index", "energy_kwh", "deadline_gap_pct",
            "cell_balance_loss_pct", "agv_utilization",
            "fatigue_h1", "fatigue_h2", "fatigue_h3",
            "factory_noise_db",
        ]
        for key in kpi_keys:
            node = await kpi_folder.add_variable(idx, OPCUA_NODES[key], 0.0)
            await node.set_writable()
            self._nodes[key] = node

        # Constraint variables (float)
        for key in ["fatigue_limit", "noise_limit", "cyber_risk_limit"]:
            node = await constraint_folder.add_variable(idx, OPCUA_NODES[key], 0.0)
            await node.set_writable()
            self._nodes[key] = node

        # Contract variables (string)
        for key in ["contract_name", "contract_status"]:
            node = await contract_folder.add_variable(idx, OPCUA_NODES[key], "")
            await node.set_writable()
            self._nodes[key] = node

        await self._server.start()
        logger.info(f"OPC-UA server started with {len(self._nodes)} nodes")
        ready.set()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def status(self) -> str:
        if self._connected:
            return f"Serving ({self.endpoint})"
        return "Disconnected (stub mode)"

    def update_kpi(self, kpi_data: dict) -> dict:
        """Write current KPI values to OPC-UA variables."""
        self.event_bus.publish(
            TOPIC_OPCUA_KPI_UPDATE,
            kpi_data,
            source="opcua_bridge",
        )

        if self._connected and self._loop is not None:
            try:
                return self._write_nodes(kpi_data)
            except Exception as e:
                logger.warning(f"OPC-UA KPI update failed: {e}")

        return {"updated_nodes": list(kpi_data.keys()), "stub": True}

    def update_constraints(self, constraint_data: dict) -> dict:
        """Write constraint limits to OPC-UA variables."""
        self.event_bus.publish(
            TOPIC_OPCUA_CONSTRAINT_UPDATE,
            constraint_data,
            source="opcua_bridge",
        )

        if self._connected and self._loop is not None:
            try:
                return self._write_nodes(constraint_data)
            except Exception as e:
                logger.warning(f"OPC-UA constraint update failed: {e}")

        return {"updated_nodes": list(constraint_data.keys()), "stub": True}

    def update_contract_status(self, name: str, status_str: str) -> dict:
        """Update contract name and status on OPC-UA server."""
        data = {"contract_name": name, "contract_status": status_str}

        if self._connected and self._loop is not None:
            try:
                return self._write_nodes(data)
            except Exception as e:
                logger.warning(f"OPC-UA contract status update failed: {e}")

        return {"updated_nodes": list(data.keys()), "stub": True}

    def read_all_nodes(self) -> dict:
        """Read current values of all OPC-UA nodes."""
        if self._connected and self._loop is not None:
            try:
                return self._read_nodes()
            except Exception as e:
                logger.warning(f"OPC-UA read failed: {e}")

        return {name: None for name in OPCUA_NODES}

    def _write_nodes(self, data: dict) -> dict:
        """Write values to OPC-UA nodes via the server's event loop."""
        updated: list[str] = []

        async def _do_write() -> None:
            for key, value in data.items():
                if key in self._nodes:
                    await self._nodes[key].write_value(value)
                    updated.append(key)

        future = asyncio.run_coroutine_threadsafe(_do_write(), self._loop)  # type: ignore[arg-type]
        future.result(timeout=5.0)
        logger.debug(f"OPC-UA nodes written: {updated}")
        return {"updated_nodes": updated, "stub": False}

    def _read_nodes(self) -> dict:
        """Read all node values via the server's event loop."""
        result: dict[str, Any] = {}

        async def _do_read() -> None:
            for key, node in self._nodes.items():
                result[key] = await node.read_value()

        future = asyncio.run_coroutine_threadsafe(_do_read(), self._loop)  # type: ignore[arg-type]
        future.result(timeout=5.0)
        return result

    def shutdown(self) -> None:
        """Stop the OPC-UA server and clean up."""
        if self._connected and self._server is not None and self._loop is not None:
            async def _stop() -> None:
                await self._server.stop()

            try:
                future = asyncio.run_coroutine_threadsafe(_stop(), self._loop)
                future.result(timeout=5.0)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._connected = False
        self._server = None
        self._nodes.clear()
