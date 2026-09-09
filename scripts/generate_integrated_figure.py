#!/usr/bin/env python3
"""Generate publication-quality integrated system stack figure.

Creates a multi-panel figure showing:
  (a) Phase-level schedule evaluations with constraint limits and phase annotations
  (b) OPC-UA node tree exposing CBPA variables to industrial clients
  (c) Integration data-flow architecture (CBPA ↔ Isaac Sim ↔ BaSyx ↔ OPC-UA)

Uses actual experiment data from the integrated 3-shift run.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

# Paths
BASE = Path(__file__).resolve().parent.parent
import argparse as _argparse  # noqa: E402
_ap = _argparse.ArgumentParser(description="Integrated execution-stack figure (Figure 4)")
_ap.add_argument("--data", default=str(BASE / "data" / "results_full_integrated_3shifts"),
                 help="single-cell run directory with table3.json, execution_trace.json, working_modes.json (e.g. the medoid run)")
_ap.add_argument("--factory-data", default=str(BASE / "data" / "results_factory_integrated"),
                 help="factory run directory with factory_table.json")
_ap.add_argument("--out", default=None, help="output figure directory (default: <data>/figures)")
_args = _ap.parse_args()
DATA = Path(_args.data)
FACTORY_DATA = Path(_args.factory_data)
OUT = Path(_args.out) if _args.out else DATA / "figures"
MANUSCRIPT = BASE.parent / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)

# Load data — build live-demo-compatible structure from table3 + execution_trace
table3 = json.loads((DATA / "table3.json").read_text())
exec_trace = json.loads((DATA / "execution_trace.json").read_text())
# Map phases to schedules: Phase 1→S1 (deployed), Phase 3→S2 (rejected), Phase 5→S3 (deployed)
_PHASE_SCHEDULE = {1: "S1", 3: "S2", 5: "S3"}
live = {"phases": []}
for entry in exec_trace:
    phase_num = entry.get("phase", 0)
    phase_data = {
        "phase": phase_num,
        "working_mode": entry.get("working_mode", "auditor"),
        "shift": 1,
    }
    sched_name = _PHASE_SCHEDULE.get(phase_num, "")
    if sched_name and sched_name in table3:
        m = table3[sched_name]
        phase_data["metrics"] = {
            "throughput_uph": m.get("throughput_uph", 0),
            "fatigue_index": m.get("fatigue_index", 0),
            "noise_db": m.get("noise_db", 0),
            "energy_kwh": m.get("energy_kwh", 0),
        }
        phase_data["feasible"] = m.get("feasible", True)
        if sched_name == "S2":
            phase_data["feasible"] = False  # S2 is always rejected
    live["phases"].append(phase_data)
factory_table = json.loads((FACTORY_DATA / "factory_table.json").read_text())
modes = json.loads((DATA / "working_modes.json").read_text())

# Style — Times New Roman, larger sizes
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MODE_COLORS = {
    "legislator": "#2196F3",
    "auditor": "#4CAF50",
    "partner": "#FF9800",
}


def panel_a_dashboard(ax_kpi, ax_timeline):
    """Panel (a): Phase-level schedule evaluations with limits (twin axis layout)."""
    # Extract shift 3 (last shift) phase data
    shift3_phases = [p for p in live["phases"] if p.get("shift", 1) == 3]
    if not shift3_phases:
        shift3_phases = live["phases"][-6:]

    phases_with_metrics = [p for p in shift3_phases if p.get("metrics")]
    throughputs = [p["metrics"]["throughput_uph"] for p in phases_with_metrics]
    fatigues = [p["metrics"]["fatigue_index"] for p in phases_with_metrics]
    noises = [p["metrics"]["noise_db"] for p in phases_with_metrics]
    feasibles = [p.get("feasible", True) for p in phases_with_metrics]

    x = np.arange(len(phases_with_metrics))
    w = 0.27

    # Left axis: throughput + noise (similar magnitudes)
    ax_kpi.bar(x - w / 2, throughputs, w, label="Throughput (uph)",
               color="#FFA726", edgecolor="black", linewidth=0.6, alpha=0.9)
    ax_kpi.bar(x + w / 2, noises, w, label="Noise (dB)",
               color="#42A5F5", edgecolor="black", linewidth=0.6, alpha=0.9)
    ax_kpi.axhline(y=80, color="#1565C0", linestyle="--", linewidth=1.4,
                   label="Noise limit (80 dB)")
    ax_kpi.set_ylim(0, 115)
    ax_kpi.set_ylabel("Throughput (uph) / Noise (dB)", fontsize=11)

    # Right axis: fatigue (0–1)
    ax_fat = ax_kpi.twinx()
    ax_fat.plot(x, fatigues, marker="D", markersize=10, linewidth=2.2,
                color="#2E7D32", label="Fatigue index", zorder=5,
                markerfacecolor="#66BB6A", markeredgecolor="#1B5E20", markeredgewidth=1.2)
    ax_fat.axhline(y=0.40, color="#d32f2f", linestyle="--", linewidth=1.5,
                   label="Fatigue limit (0.40)")
    ax_fat.set_ylim(0, 0.7)
    ax_fat.set_ylabel("Fatigue index (0\u20131)", fontsize=11, color="#2E7D32")
    ax_fat.tick_params(axis="y", labelcolor="#2E7D32", labelsize=10)

    ax_kpi.set_xticks(x)
    phase_labels = [f"Ph{p['phase']}\n{'Accepted' if p.get('feasible', True) else 'Rejected'}"
                    for p in phases_with_metrics]
    ax_kpi.set_xticklabels(phase_labels, fontsize=10)
    for tick, p in zip(ax_kpi.get_xticklabels(), phases_with_metrics):
        if not p.get("feasible", True):
            tick.set_color("#C62828")
            tick.set_fontweight("bold")
    ax_kpi.set_title("(a) Phase-level schedule metrics and constraint limits",
                     fontweight="bold", loc="left", fontsize=12, pad=10)

    # Combined legend
    h1, l1 = ax_kpi.get_legend_handles_labels()
    h2, l2 = ax_fat.get_legend_handles_labels()
    ax_kpi.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9,
                  ncol=2, framealpha=0.9)

    # Timeline with working modes
    all_phases = shift3_phases if shift3_phases else live["phases"][-6:]
    for i, p in enumerate(all_phases):
        mode = p.get("working_mode", "auditor")
        color = MODE_COLORS.get(mode, "#999")
        ax_timeline.barh(0, 1, left=i, height=0.6, color=color, alpha=0.85,
                         edgecolor="white", linewidth=0.6)
        ax_timeline.text(i + 0.5, 0, f"P{p['phase']}", ha="center", va="center",
                         fontsize=11, fontweight="bold", color="white")

    ax_timeline.set_xlim(0, len(all_phases))
    ax_timeline.set_ylim(-0.5, 0.7)
    ax_timeline.set_yticks([])
    ax_timeline.set_xlabel("Phase sequence", fontsize=11)
    ax_timeline.tick_params(axis="x", labelsize=10)
    patches = [mpatches.Patch(color=c, label=m.capitalize()) for m, c in MODE_COLORS.items()]
    ax_timeline.legend(handles=patches, loc="upper center", ncol=3, fontsize=9,
                       framealpha=0.9, bbox_to_anchor=(0.5, 1.0))
    ax_timeline.set_title("(b) Working-mode lifecycle", fontsize=12, loc="left",
                          fontweight="bold")


def panel_b_opcua(ax):
    """Panel (b): OPC-UA node tree, two-column layout."""
    cell = table3["S3"]
    factory = factory_table["FS3"]
    tree_left = {
        "KPI (Single-cell)": [
            ("Throughput_uph", f"{cell['throughput_uph']:.1f}"),
            ("DefectRate", f"{cell['defect_rate']:.4f}"),
            ("Noise_dB", f"{cell['noise_db']:.1f}"),
            ("FatigueIndex", f"{cell['fatigue_index']:.3f}"),
            ("Energy_kWh", f"{cell['energy_kwh']:.0f}"),
            ("DeadlineGap_%", f"{cell['deadline_gap_pct']:.1f}"),
        ],
        "Constraints": [
            ("FatigueLimit", "0.40"),
            ("NoiseLimit", "80.0 dB"),
            ("CyberRiskLimit", "Medium"),
        ],
        "Contract": [
            ("Name", "C2"),
            ("Status", "ACTIVE"),
        ],
    }
    tree_right = {
        "KPI (Factory)": [
            ("CellBalance_%", f"{factory['cell_balance_loss_pct']:.1f}"),
            ("AGV_Util", f"{factory['agv_utilization']:.2f}"),
            ("Fatigue_H1", f"{factory['fatigue_h1']:.3f}"),
            ("Fatigue_H2", f"{factory['fatigue_h2']:.3f}"),
            ("Fatigue_H3", f"{factory['fatigue_h3']:.3f}"),
            ("FactoryNoise_dB", f"{factory['noise_db']:.1f}"),
        ],
    }

    folder_colors = {
        "KPI (Single-cell)": "#1565C0",
        "KPI (Factory)": "#6A1B9A",
        "Constraints": "#d32f2f",
        "Contract": "#2E7D32",
    }

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("(c) OPC-UA node layout with reference stabilisation values",
                 fontweight="bold", loc="left", fontsize=12, pad=12)

    # Header
    ax.text(0.2, 9.4, "CBPA OPC-UA Server", fontsize=11, fontweight="bold",
            fontfamily="monospace", va="center")

    def draw_column(col_x, col_w, tree, start_y=8.4):
        y = start_y
        for folder, items in tree.items():
            color = folder_colors.get(folder, "#666")
            # Folder header bar
            ax.add_patch(mpatches.FancyBboxPatch(
                (col_x, y - 0.22), col_w, 0.6,
                boxstyle="round,pad=0.02", facecolor=color, alpha=0.18,
                edgecolor=color, linewidth=1.2))
            ax.text(col_x + 0.18, y + 0.08, folder, fontsize=10.5,
                    fontweight="bold", color=color, fontfamily="monospace",
                    va="center")
            y -= 0.85
            for key, val in items:
                ax.text(col_x + 0.35, y, key, fontsize=9.5,
                        fontfamily="monospace", va="center", color="#333")
                ax.text(col_x + col_w - 0.15, y, val, fontsize=9.5,
                        fontfamily="monospace", va="center", color=color,
                        fontweight="bold", ha="right")
                y -= 0.62
            y -= 0.35

    draw_column(0.2, 4.6, tree_left)
    draw_column(5.2, 4.6, tree_right)


def panel_c_architecture(ax):
    """Panel (c): Integration data-flow architecture."""
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.4, 4.6)
    ax.axis("off")
    ax.set_title("(d) Integrated execution stack data flow",
                 fontweight="bold", loc="left", fontsize=12, pad=12)

    # Boxes: (x, y, w, h, label, color)
    boxes = [
        (0.2, 1.5, 2.0, 1.3, "CBPA\nLayers 1\u20135\n+ Meta",   "#1565C0"),
        (2.9, 2.7, 2.0, 0.95, "Isaac Sim\nDigital Twin",        "#2E7D32"),
        (2.9, 0.55, 2.0, 0.95, "Eclipse BaSyx\nAAS Registry",   "#6A1B9A"),
        (5.6, 1.5, 2.0, 1.3, "OPC-UA\nServer",                  "#d32f2f"),
        (8.3, 2.7, 2.0, 0.95, "PLC / SCADA\nClients",           "#455A64"),
        (8.3, 0.55, 2.0, 0.95, "Streamlit\nDashboard",          "#FF6F00"),
    ]

    for x, y, w, h, label, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                        facecolor=color, alpha=0.18,
                                        edgecolor=color, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)

    arrow_uni = dict(arrowstyle="->", color="#333", lw=1.5)
    arrow_bi  = dict(arrowstyle="<->", color="#333", lw=1.5)

    # No bbox needed — labels positioned in clear empty space
    # CBPA <-> Isaac (deploy + metrics)
    ax.annotate("", xy=(2.9, 3.15), xytext=(2.2, 2.6), arrowprops=arrow_bi)
    ax.text(2.55, 4.20, "deploy /\nmetrics", fontsize=10, color="#333",
            ha="center", va="center", style="italic")

    # CBPA <-> BaSyx (AAS sync)
    ax.annotate("", xy=(2.9, 1.05), xytext=(2.2, 1.6), arrowprops=arrow_bi)
    ax.text(2.55, -0.05, "AAS sync", fontsize=10, color="#333",
            ha="center", va="center", style="italic")

    # CBPA -> OPC-UA (publish KPIs + contract) — single line, above the arrow
    ax.annotate("", xy=(5.6, 2.05), xytext=(2.2, 2.05), arrowprops=arrow_uni)
    ax.text(3.9, 2.30, "publish KPI + contract variables", fontsize=10,
            color="#333", ha="center", va="center", style="italic")

    # OPC-UA -> PLC (subscribe)
    ax.annotate("", xy=(8.3, 3.15), xytext=(7.6, 2.6), arrowprops=arrow_uni)
    ax.text(7.95, 4.20, "subscribe", fontsize=10, color="#333",
            ha="center", va="center", style="italic")

    # OPC-UA -> Dashboard (read)
    ax.annotate("", xy=(8.3, 1.05), xytext=(7.6, 1.6), arrowprops=arrow_uni)
    ax.text(7.95, -0.05, "read", fontsize=10, color="#333",
            ha="center", va="center", style="italic")


# === Create figure ===
fig = plt.figure(figsize=(13, 11))
gs = GridSpec(3, 2, figure=fig,
              height_ratios=[1.0, 1.15, 0.85],
              width_ratios=[1.5, 1.0],
              hspace=0.45, wspace=0.22)

# Panel (a): top-left — KPI dashboard
ax_kpi = fig.add_subplot(gs[0, 0])
ax_timeline = fig.add_subplot(gs[0, 1])
panel_a_dashboard(ax_kpi, ax_timeline)

# Panel (b): middle — OPC-UA node tree
ax_opcua = fig.add_subplot(gs[1, :])
panel_b_opcua(ax_opcua)

# Panel (c): bottom — architecture
ax_arch = fig.add_subplot(gs[2, :])
panel_c_architecture(ax_arch)

# Save
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"fig_integrated_stack.{ext}", bbox_inches="tight", dpi=300)
    if MANUSCRIPT.is_dir():
        fig.savefig(MANUSCRIPT / f"fig_integrated_stack.{ext}", bbox_inches="tight", dpi=300)

print(f"Saved to {OUT / 'fig_integrated_stack.pdf'}")
if MANUSCRIPT.is_dir():
    print(f"Saved to {MANUSCRIPT / 'fig_integrated_stack.pdf'}")
plt.close()
