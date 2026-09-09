#!/usr/bin/env python3
"""Generate publication-quality visualisations for factory CBPA results.

Produces four figures in <data_dir>/figures/:
  fig1_factory_kpi_comparison   — FS1/FS2/FS3 comparison across key KPIs
  fig2_factory_cell_metrics     — Cell A vs Cell B per-schedule breakdown
  fig3_factory_operator_fatigue — Per-operator fatigue (H1/H2/H3) per schedule
  fig4_factory_execution_swimlane — Working-mode + layer timeline

Usage:
    python scripts/visualize_factory_results.py
    python scripts/visualize_factory_results.py --data-dir data/results_factory_nollm_analytical
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# ── Styling (mirrors visualize_results.py) ────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":        13,
    "axes.titlesize":   15,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "figure.dpi":       150,
})

COLORS = {
    "FS1": "#2196F3",   # blue   — initial deployment
    "FS2": "#F44336",   # red    — rejected replan
    "FS3": "#4CAF50",   # green  — stabilised
    "FS4": "#9C27B0",   # purple — macro-adapted (R3 degradation)
}
HATCH = {"FS1": "", "FS2": "///", "FS3": "", "FS4": "..."}

# Data holders (populated by _load_data)
table  = None   # factory_table.json
trace  = None   # execution_trace.json
modes  = None   # working_modes.json

DATA_DIR = ""
FIG_DIR  = ""


def _load_data(data_dir: str) -> None:
    global table, trace, modes, DATA_DIR, FIG_DIR
    DATA_DIR = data_dir
    FIG_DIR  = os.path.join(DATA_DIR, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    def _load(name):
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    table = _load("factory_table.json")
    trace = _load("execution_trace.json")
    modes = _load("working_modes.json")


def _savefig(name: str) -> None:
    for ext in ("png", "pdf"):
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {name}")


# ════════════════════════════════════════════════════════════════════════
#  Fig 1 — Factory KPI Comparison (6-panel bar chart)
# ════════════════════════════════════════════════════════════════════════
def fig1_factory_kpi_comparison() -> None:
    if table is None:
        return
    schedules = [s for s in ["FS1", "FS2", "FS3", "FS4"] if s in table]
    metrics = [
        ("total_uph",              "Total Throughput (u/h)", None,  None),
        ("defect_rate",            "Defect Rate",             None,  None),
        ("noise_db",               "Factory Noise (dB)",      82.0,  "Noise limit (82 dB)"),
        ("cell_balance_loss_pct",  "Cell Balance Loss (%)",   None,  None),
        ("energy_kwh",             "Energy (kWh)",            None,  None),
        ("agv_utilization",        "AGV Utilisation",         None,  None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for idx, (key, label, threshold, thresh_label) in enumerate(metrics):
        ax = axes[idx]
        vals = [table[s][key] for s in schedules]
        bars = ax.bar(
            schedules, vals,
            color=[COLORS[s] for s in schedules],
            hatch=[HATCH[s] for s in schedules],
            edgecolor="white", linewidth=0.8,
        )
        if threshold is not None:
            ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
                       label=thresh_label)
            ax.legend(fontsize=9)
        ax.set_title(label)
        ax.set_ylabel(label)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.2g}", ha="center", va="bottom", fontsize=9)

    legend_patches = [
        mpatches.Patch(color=COLORS[s], label=_schedule_label(s))
        for s in schedules
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=len(schedules),
               bbox_to_anchor=(0.5, 1.0), fontsize=12, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig("fig1_factory_kpi_comparison")


# ════════════════════════════════════════════════════════════════════════
#  Fig 2 — Cell A vs Cell B throughput comparison
# ════════════════════════════════════════════════════════════════════════
def fig2_factory_cell_metrics() -> None:
    if table is None:
        return
    schedules = [s for s in ["FS1", "FS2", "FS3", "FS4"] if s in table]
    x = np.arange(len(schedules))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Throughput per cell
    ax = axes[0]
    cellA = [table[s]["cellA_uph"] for s in schedules]
    cellB = [table[s]["cellB_uph"] for s in schedules]
    bars_a = ax.bar(x - width / 2, cellA, width, label="Cell A (Assembly)",
                    color="#1565C0", edgecolor="white")
    bars_b = ax.bar(x + width / 2, cellB, width, label="Cell B (Test & Pack)",
                    color="#2E7D32", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([_schedule_label(s) for s in schedules])
    ax.set_ylabel("Throughput (u/h)")
    ax.set_title("Per-Cell Throughput")
    ax.legend()
    for bar, v in list(zip(bars_a, cellA)) + list(zip(bars_b, cellB)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # Energy per cell (derived as 53% / 47% split of factory total, approximate)
    ax = axes[1]
    total_e = [table[s]["energy_kwh"] for s in schedules]
    cellA_e = [e * 0.53 for e in total_e]
    cellB_e = [e * 0.47 for e in total_e]
    bars_a2 = ax.bar(x - width / 2, cellA_e, width, label="Cell A",
                     color="#1565C0", edgecolor="white")
    bars_b2 = ax.bar(x + width / 2, cellB_e, width, label="Cell B",
                     color="#2E7D32", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([_schedule_label(s) for s in schedules])
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("Per-Cell Energy (estimated)")
    ax.legend()

    plt.tight_layout()
    _savefig("fig2_factory_cell_metrics")


# ════════════════════════════════════════════════════════════════════════
#  Fig 3 — Per-operator fatigue (grouped bar)
# ════════════════════════════════════════════════════════════════════════
def fig3_factory_operator_fatigue() -> None:
    if table is None:
        return
    schedules = [s for s in ["FS1", "FS2", "FS3", "FS4"] if s in table]
    operators = ["H1", "H2", "H3"]
    op_colors = {"H1": "#E65100", "H2": "#6A1B9A", "H3": "#00838F"}
    limits = {"H1": 0.40, "H2": 0.35, "H3": 0.40}

    x = np.arange(len(schedules))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11, 5))
    offsets = [-1, 0, 1]
    for i, op in enumerate(operators):
        key = f"fatigue_{op.lower()}"
        vals = [table[s][key] for s in schedules]
        bars = ax.bar(x + offsets[i] * width, vals, width,
                      label=f"{op} (lim {limits[op]:.2f})",
                      color=op_colors[op], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=11)

    # Per-operator limit lines (H2 at 0.35; H1 & H3 share 0.40)
    n = len(schedules)
    ax.axhline(0.35, color="#6A1B9A", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.axhline(0.40, color="#555555", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.annotate("H2 limit (0.35)", xy=(n - 0.5, 0.35), xytext=(6, 0),
                textcoords="offset points", fontsize=11, color="#6A1B9A",
                va="center", ha="left", annotation_clip=False)
    ax.annotate("H1/H3 limit (0.40)", xy=(n - 0.5, 0.40), xytext=(6, 0),
                textcoords="offset points", fontsize=11, color="#555555",
                va="center", ha="left", annotation_clip=False)

    ax.set_xticks(x)
    ax.set_xticklabels([_schedule_label(s) for s in schedules])
    ax.set_ylabel("Fatigue Index")
    # No title — provided by LaTeX caption
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(0.55, ax.get_ylim()[1]))
    plt.tight_layout()
    _savefig("fig3_factory_operator_fatigue")


# ════════════════════════════════════════════════════════════════════════
#  Fig 4 — Execution swimlane (working mode + layer timeline)
# ════════════════════════════════════════════════════════════════════════
def fig4_factory_execution_swimlane() -> None:
    if trace is None:
        return

    mode_colors = {
        "legislator": "#1565C0",
        "auditor":    "#2E7D32",
        "partner":    "#E65100",
    }
    mode_labels = {
        "legislator": "Legislator",
        "auditor":    "Auditor",
        "partner":    "Partner",
    }

    # Short note labels per phase (manually curated for readability)
    note_short = {
        1: "Initial contract\n(9 constraints)",
        2: "Demand surge\n+20%",
        3: "Replan attempt\n(H2 absent)",
        4: "Escalation\n(human co-decision)",
        5: "Adaptation\n(contract C2)",
        6: "Shift-close\n(macro-adapt)",
    }

    n = len(trace)
    fig, ax = plt.subplots(figsize=(15, 5))

    for entry in trace:
        ph = entry["phase"] - 1   # 0-indexed x position
        mode = entry["working_mode"]
        color = mode_colors.get(mode, "#607D8B")

        # Coloured block for this phase
        ax.barh(0, 1, left=ph, height=0.7, color=color, edgecolor="white",
                linewidth=2, align="center")

        # Phase number and mode label inside block
        ax.text(ph + 0.5, 0.05, f"Phase {entry['phase']}\n{mode_labels.get(mode, mode)}",
                ha="center", va="center", fontsize=18, color="white", fontweight="bold")

        # Layer annotation below bar
        layers = entry.get("active_layers", "")
        # Break long chains into two lines at the midpoint arrow
        if layers.count("→") >= 4:
            parts = layers.split(" → ")
            mid = len(parts) // 2
            layers = " → ".join(parts[:mid]) + " →\n" + " → ".join(parts[mid:])
        ax.text(ph + 0.5, -0.50, layers, ha="center", va="top", fontsize=15,
                color="black", fontweight="medium")

        # Short note annotation above bar
        short = note_short.get(entry["phase"], "")
        if short:
            ax.text(ph + 0.5, 0.55, short, ha="center", va="bottom", fontsize=15,
                    color="black", style="italic")

    ax.set_xlim(-0.15, n + 0.15)
    ax.set_ylim(-1.0, 0.85)
    ax.axis("off")
    # No title — provided by LaTeX caption

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=mode_labels[m])
        for m, c in mode_colors.items()
    ]
    ax.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches),
              fontsize=15, bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    plt.tight_layout()
    _savefig("fig4_factory_execution_swimlane")


# ════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════
def _schedule_label(s: str) -> str:
    labels = {
        "FS1": "FS1\n(initial)",
        "FS2": "FS2\n(replan)",
        "FS3": "FS3\n(stabilised)",
        "FS4": "FS4\n(next-shift)",
    }
    return labels.get(s, s)


# ════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate factory CBPA result figures"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing factory result JSON files (default: auto-detect)",
    )
    args = parser.parse_args()

    # Default: first matching results_factory_* directory
    data_dir = args.data_dir
    if data_dir is None:
        import glob
        candidates = sorted(glob.glob(
            os.path.join(PROJECT_DIR, "data", "results_factory*")
        ), reverse=True)
        if candidates:
            data_dir = candidates[0]
        else:
            data_dir = os.path.join(PROJECT_DIR, "data", "results_factory_analytical")

    print(f"Loading factory results from: {data_dir}")
    _load_data(data_dir)

    if table is None:
        print("ERROR: factory_table.json not found. Run the factory experiment first.")
        sys.exit(1)

    print("Generating figures...")
    fig1_factory_kpi_comparison()
    fig2_factory_cell_metrics()
    fig3_factory_operator_fatigue()
    fig4_factory_execution_swimlane()

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
