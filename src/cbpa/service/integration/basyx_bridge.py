"""Eclipse BaSyx bridge: maps CBPA contracts to AAS submodels via REST API.

When a BaSyx AAS server URL is configured and reachable (via httpx),
this bridge synchronizes the outcome contract C^out = <KPI, K, P, X, A>
as an Asset Administration Shell with five submodels using the BaSyx v2
REST API.  Otherwise it runs in stub mode (messages go to EventBus only).
"""

from __future__ import annotations

import logging
from typing import Any

from cbpa.service.integration.event_bus import (
    TOPIC_AAS_CONTRACT_SYNC,
    TOPIC_AAS_KPI_SUBMODEL,
    EventBus,
)

logger = logging.getLogger(__name__)

# httpx is an existing optional dependency
try:
    import httpx

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

# AAS API payload helpers

_SUBMODEL_FIELDS = {
    "kpi_targets": "KPITargets",
    "hard_constraints": "HardConstraints",
    "priority_order": "PriorityOrder",
    "context": "OperationalContext",
    "assumptions": "Assumptions",
}


def _make_aas_payload(aas_id: str) -> dict:
    """Build AAS shell JSON for the BaSyx v2 REST API."""
    return {
        "id": aas_id,
        "idShort": "CBPAProductionCell",
        "assetInformation": {
            "assetKind": "Instance",
            "globalAssetId": f"urn:cbpa:{aas_id}",
        },
        "submodels": [
            {"type": "ExternalReference", "keys": [{"type": "Submodel", "value": f"{aas_id}/{sm}"}]}
            for sm in _SUBMODEL_FIELDS.values()
        ],
        "modelType": "AssetAdministrationShell",
    }


def _make_submodel_payload(aas_id: str, sm_name: str, data: Any) -> dict:
    """Build submodel JSON with SubmodelElements from contract data."""
    elements = []

    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # KPI targets / hard constraints — use SubmodelElementCollection
                props = [
                    {"idShort": k, "valueType": "xs:string", "value": str(v), "modelType": "Property"}
                    for k, v in item.items()
                ]
                elements.append({
                    "idShort": f"{sm_name}_{i}",
                    "value": props,
                    "modelType": "SubmodelElementCollection",
                })
            else:
                # Priority order — simple string properties
                elements.append({
                    "idShort": f"{sm_name}_{i}",
                    "valueType": "xs:string",
                    "value": str(item),
                    "modelType": "Property",
                })
    elif isinstance(data, dict):
        for k, v in data.items():
            elements.append({
                "idShort": k,
                "valueType": "xs:string",
                "value": str(v),
                "modelType": "Property",
            })

    return {
        "id": f"{aas_id}/{sm_name}",
        "idShort": sm_name,
        "submodelElements": elements,
        "modelType": "Submodel",
    }


class BaSyxBridge:
    """Maps CBPA outcome contracts to AAS submodels via Eclipse BaSyx REST API.

    The contract tuple C^out = <KPI, K, P, X, A> maps to five AAS submodels:
      - ``KPITargets``       — KPI objectives (direction, threshold, unit)
      - ``HardConstraints``  — non-negotiable limits K
      - ``PriorityOrder``    — stakeholder priority P
      - ``OperationalContext``— shift context X
      - ``Assumptions``      — environmental assumptions A

    Connects via httpx to the BaSyx v2 REST API (no Python SDK needed).
    """

    def __init__(
        self,
        event_bus: EventBus,
        registry_url: str = "",
        aas_server_url: str = "",
        aas_id: str = "cbpa_production_cell_001",
    ):
        self.event_bus = event_bus
        self.registry_url = registry_url.rstrip("/") if registry_url else ""
        # Default: derive aas_server_url from registry_url pattern
        if not aas_server_url and self.registry_url:
            # http://localhost:3000/registry -> http://localhost:3000/aas-server
            base = self.registry_url.rsplit("/", 1)[0]
            self.aas_server_url = f"{base}/aas-server"
        else:
            self.aas_server_url = aas_server_url.rstrip("/") if aas_server_url else ""
        self.aas_id = aas_id
        self._connected = False
        self._client: httpx.Client | None = None
        self._can_connect = bool(self.registry_url and _HAS_HTTPX)

        if self._can_connect:
            self._try_connect()
        else:
            reason = "no registry_url" if not self.registry_url else "httpx not installed"
            logger.info(f"BaSyx bridge stub mode ({reason})")

    def _try_connect(self) -> bool:
        """Attempt to connect to BaSyx registry. Returns True on success."""
        if self._connected:
            return True
        if not self._can_connect:
            return False
        try:
            if self._client is None:
                self._client = httpx.Client(timeout=10.0)
            resp = self._client.get(f"{self.registry_url}/shell-descriptors")
            resp.raise_for_status()
            self._connected = True
            logger.info(f"BaSyx bridge connected to {self.registry_url}")
            return True
        except Exception as e:
            logger.warning(f"BaSyx connect failed: {e}. Will retry on next call.")
            return False

    def _ensure_connected(self) -> bool:
        """Lazy reconnect: retry connection if not yet connected."""
        if self._connected:
            return True
        return self._try_connect()

    @property
    def is_connected(self) -> bool:
        self._ensure_connected()
        return self._connected

    @property
    def status(self) -> str:
        if self._connected or self._ensure_connected():
            return f"Connected ({self.registry_url})"
        if self._can_connect:
            return "Connecting... (will retry)"
        return "Disconnected (stub mode)"

    def sync_contract(self, contract_data: dict) -> dict:
        """Push the full outcome contract C^out to AAS submodels.

        Parameters
        ----------
        contract_data : dict
            Serialized OutcomeContract with keys: name, kpi_targets,
            hard_constraints, priority_order, context, assumptions.

        Returns
        -------
        dict
            Sync result with submodel IDs.
        """
        self.event_bus.publish(
            TOPIC_AAS_CONTRACT_SYNC,
            contract_data,
            source="basyx_bridge",
        )

        if self._ensure_connected() and self._client is not None:
            try:
                return self._push_to_server(contract_data)
            except Exception as e:
                logger.warning(f"BaSyx contract sync failed: {e}")

        # Stub response
        return {
            "aas_id": self.aas_id,
            "contract_name": contract_data.get("name", "unknown"),
            "submodels_synced": list(_SUBMODEL_FIELDS.values()),
            "stub": True,
        }

    def update_kpi_submodel(self, kpi_data: dict) -> dict:
        """Update the live KPI submodel with current readings."""
        self.event_bus.publish(
            TOPIC_AAS_KPI_SUBMODEL,
            kpi_data,
            source="basyx_bridge",
        )

        if self._ensure_connected() and self._client is not None:
            try:
                sm_payload = _make_submodel_payload(self.aas_id, "KPITargets", kpi_data)
                sm_url = f"{self.aas_server_url}/submodels/{_b64(self.aas_id + '/KPITargets')}"
                resp = self._client.put(sm_url, json=sm_payload)
                if resp.status_code == 404:
                    # Submodel doesn't exist yet — create it
                    resp = self._client.post(
                        f"{self.aas_server_url}/submodels", json=sm_payload
                    )
                resp.raise_for_status()
                return {"status": "updated", "fields": list(kpi_data.keys())}
            except Exception as e:
                logger.debug(f"BaSyx KPI update failed: {e}")

        return {"status": "acknowledged", "stub": True}

    def get_contract_submodels(self) -> dict:
        """Retrieve current AAS submodel state (for auditor review)."""
        if self._ensure_connected() and self._client is not None:
            try:
                resp = self._client.get(f"{self.aas_server_url}/submodels")
                resp.raise_for_status()
                return {"aas_id": self.aas_id, "submodels": resp.json(), "stub": False}
            except Exception as e:
                logger.warning(f"BaSyx read failed: {e}")

        return {
            "aas_id": self.aas_id,
            "submodels": {sm: [] for sm in _SUBMODEL_FIELDS.values()},
            "stub": True,
        }

    def _push_to_server(self, contract_data: dict) -> dict:
        """Create/update AAS shell and submodels via REST API."""
        assert self._client is not None

        # 0. Clear stale registry entries (BaSyx registry may have
        #    descriptors from a previous run whose data was wiped)
        try:
            self._client.delete(
                f"{self.registry_url}/shell-descriptors/{_b64(self.aas_id)}"
            )
            # Submodel registry: same host, port 9083
            from urllib.parse import urlparse
            parsed = urlparse(self.registry_url)
            sm_reg_base = f"{parsed.scheme}://{parsed.hostname}:9083"
            for sm_name in _SUBMODEL_FIELDS.values():
                self._client.delete(
                    f"{sm_reg_base}/submodel-descriptors/{_b64(self.aas_id + '/' + sm_name)}"
                )
        except Exception:
            pass  # Registry cleanup is best-effort

        # 1. Create or update the AAS shell
        aas_payload = _make_aas_payload(self.aas_id)
        # Check if shell already exists
        shell_url = f"{self.aas_server_url}/shells/{_b64(self.aas_id)}"
        check = self._client.get(shell_url)
        if check.status_code == 200:
            # Already exists — update
            resp = self._client.put(shell_url, json=aas_payload)
            if not resp.is_success:
                logger.debug("BaSyx shell update returned %s", resp.status_code)
        else:
            # Create new shell (without submodel references to avoid
            # registry integration errors on first creation)
            simple_payload = {
                "id": self.aas_id,
                "idShort": "CBPAProductionCell",
                "assetInformation": {
                    "assetKind": "Instance",
                    "globalAssetId": f"urn:cbpa:{self.aas_id}",
                },
                "modelType": "AssetAdministrationShell",
            }
            resp = self._client.post(f"{self.aas_server_url}/shells", json=simple_payload)
            if not resp.is_success:
                logger.warning("BaSyx shell creation returned %s — continuing with submodels", resp.status_code)

        # 2. Create or update each submodel
        synced = []
        for field_name, sm_name in _SUBMODEL_FIELDS.items():
            data = contract_data.get(field_name, {})
            sm_payload = _make_submodel_payload(self.aas_id, sm_name, data)

            sm_id = _b64(self.aas_id + '/' + sm_name)
            sm_url = f"{self.aas_server_url}/submodels/{sm_id}"
            check = self._client.get(sm_url)
            if check.status_code == 200:
                resp = self._client.put(sm_url, json=sm_payload)
            else:
                resp = self._client.post(
                    f"{self.aas_server_url}/submodels", json=sm_payload
                )
                if resp.status_code in (409, 500):
                    resp = self._client.put(sm_url, json=sm_payload)
            if not resp.is_success:
                logger.debug("BaSyx submodel %s sync returned %s", sm_name, resp.status_code)
                continue
            synced.append(sm_name)

        # 3. Link submodel references to the AAS shell so the UI can find them
        refs_url = f"{self.aas_server_url}/shells/{_b64(self.aas_id)}/submodel-refs"
        for sm_name in synced:
            sm_full_id = f"{self.aas_id}/{sm_name}"
            ref_payload = {
                "type": "ModelReference",
                "keys": [{"type": "Submodel", "value": sm_full_id}],
            }
            ref_resp = self._client.post(refs_url, json=ref_payload)
            # 409 = already linked (idempotent)
            if ref_resp.status_code not in (201, 409):
                logger.debug("BaSyx submodel-ref %s returned %s", sm_name, ref_resp.status_code)

        logger.info(f"BaSyx: synced {len(synced)} submodels for {contract_data.get('name', '?')}")
        return {
            "aas_id": self.aas_id,
            "contract_name": contract_data.get("name", "unknown"),
            "submodels_synced": synced,
            "stub": False,
        }

    def shutdown(self) -> None:
        """Clean up HTTP client."""
        if self._client is not None:
            self._client.close()
        self._connected = False
        self._client = None


def _b64(value: str) -> str:
    """Base64url-encode a string (BaSyx v2 API uses this for IDs in URLs)."""
    import base64
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")
