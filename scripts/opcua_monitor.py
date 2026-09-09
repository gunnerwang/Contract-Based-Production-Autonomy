#!/usr/bin/env python3
"""Live OPC-UA node monitor — connects to the CBPA OPC-UA server and
prints node values every 2 seconds.

Usage:
    python scripts/opcua_monitor.py
    python scripts/opcua_monitor.py opc.tcp://localhost:4840/cbpa/

Requires: pip install asyncua
"""

from __future__ import annotations

import asyncio
import sys

from asyncua import Client


ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "opc.tcp://localhost:4840/cbpa/"

# Expected node browse paths under Objects > CBPA
NODE_PATHS = {
    "KPI": [
        "CBPA.KPI.Throughput",
        "CBPA.KPI.DefectRate",
        "CBPA.KPI.NoiseDB",
        "CBPA.KPI.FatigueIndex",
        "CBPA.KPI.EnergyKWh",
        "CBPA.KPI.DeadlineGapPct",
    ],
    "Constraints": [
        "CBPA.Constraint.FatigueLimit",
        "CBPA.Constraint.NoiseLimit",
        "CBPA.Constraint.CyberRiskLimit",
    ],
    "Contract": [
        "CBPA.Contract.Name",
        "CBPA.Contract.Status",
    ],
}

UNITS = {
    "CBPA.KPI.Throughput": "u/h",
    "CBPA.KPI.DefectRate": "",
    "CBPA.KPI.NoiseDB": "dB",
    "CBPA.KPI.FatigueIndex": "",
    "CBPA.KPI.EnergyKWh": "kWh",
    "CBPA.KPI.DeadlineGapPct": "%",
    "CBPA.Constraint.FatigueLimit": "",
    "CBPA.Constraint.NoiseLimit": "dB",
    "CBPA.Constraint.CyberRiskLimit": "",
}


async def browse_and_read(client: Client) -> dict[str, dict]:
    """Browse the CBPA folder and read all variable values."""
    ns_idx = await client.get_namespace_index("urn:cbpa:production")
    results: dict[str, dict] = {}

    objects = client.nodes.objects
    cbpa_folder = await objects.get_child(f"{ns_idx}:CBPA")

    for folder_name, node_names in NODE_PATHS.items():
        folder = await cbpa_folder.get_child(f"{ns_idx}:{folder_name}")
        children = await folder.get_children()

        for child in children:
            display_name = (await child.read_display_name()).Text
            value = await child.read_value()
            unit = UNITS.get(display_name, "")
            results[display_name] = {
                "value": value,
                "unit": unit,
                "folder": folder_name,
            }

    return results


def format_value(name: str, info: dict) -> str:
    """Format a node value for display."""
    v = info["value"]
    u = info["unit"]
    if isinstance(v, float):
        if "Rate" in name:
            return f"{v:.4f} {u}".strip()
        return f"{v:.2f} {u}".strip()
    return f"{v} {u}".strip()


async def monitor() -> None:
    print(f"Connecting to {ENDPOINT} ...")
    async with Client(url=ENDPOINT) as client:
        print(f"Connected. Monitoring nodes (Ctrl+C to stop):\n")

        cycle = 0
        while True:
            try:
                data = await browse_and_read(client)
            except Exception as e:
                print(f"  [read error: {e}]")
                await asyncio.sleep(2)
                continue

            cycle += 1
            # Clear screen and print
            print(f"\033[2J\033[H", end="")  # ANSI clear
            print(f"CBPA OPC-UA Monitor — {ENDPOINT}")
            print(f"Cycle {cycle}")
            print("=" * 55)

            for section in ["KPI", "Constraints", "Contract"]:
                print(f"\n  {section}:")
                for name, info in data.items():
                    if info["folder"] == section:
                        label = name.split(".")[-1]
                        val = format_value(name, info)
                        # Highlight non-zero/non-empty values
                        marker = "*" if (info["value"] and info["value"] != 0.0) else " "
                        print(f"  {marker} {label:.<30s} {val}")

            print(f"\n{'=' * 55}")
            print("* = has value  |  Refresh every 2s  |  Ctrl+C to stop")
            await asyncio.sleep(2)


if __name__ == "__main__":
    try:
        asyncio.run(monitor())
    except KeyboardInterrupt:
        print("\nStopped.")
