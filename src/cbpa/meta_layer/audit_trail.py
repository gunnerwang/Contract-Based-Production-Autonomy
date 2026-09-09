"""Append-only audit trail for decision provenance."""

from __future__ import annotations

import json
from pathlib import Path

from cbpa.models.escalation import AuditEntry


class AuditTrail:
    """Append-only log of all CBPA decisions and transitions."""

    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    def record(
        self,
        phase: int,
        layer: str,
        action: str,
        contract_state: str,
        working_mode: str = "",
        **details: object,
    ) -> AuditEntry:
        entry = AuditEntry(
            phase=phase,
            layer=layer,
            action=action,
            contract_state=contract_state,
            working_mode=working_mode,
            details=dict(details),
        )
        self.entries.append(entry)
        return entry

    def get_phase_entries(self, phase: int) -> list[AuditEntry]:
        return [e for e in self.entries if e.phase == phase]

    def export_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.model_dump() for e in self.entries]
        path.write_text(json.dumps(data, indent=2))

    def summary(self) -> list[dict]:
        return [
            {
                "phase": e.phase,
                "layer": e.layer,
                "action": e.action,
                "contract": e.contract_state,
                "working_mode": e.working_mode,
            }
            for e in self.entries
        ]
