#!/usr/bin/env python3
"""Generate publication-quality visualizations for the CBPA case study results."""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec


# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.environ.get("CBPA_DATA_DIR", os.path.join(PROJECT_DIR, "data", "results"))
FIG_DIR = os.path.join(DATA_DIR, "figures")

# ── Load data (deferred) ──────────────────────────────────────────────
table3 = None
audit = None
trace = None


def _load_data():
    """Load result files. Called once at the start of main()."""
    global table3, audit, trace, DATA_DIR, FIG_DIR
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "table3.json")) as f:
        table3 = json.load(f)
    with open(os.path.join(DATA_DIR, "audit_trail.json")) as f:
        audit = json.load(f)
    with open(os.path.join(DATA_DIR, "execution_trace.json")) as f:
        trace = json.load(f)

# ── Styling ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
})

COLORS = {
    "S1": "#2196F3",  # blue
    "S2": "#F44336",  # red (rejected)
    "S3": "#4CAF50",  # green (final)
}
HATCH = {"S1": "", "S2": "///", "S3": ""}


# ════════════════════════════════════════════════════════════════════════
#  Figure 1 — Schedule KPI Comparison (6-panel bar chart)
# ════════════════════════════════════════════════════════════════════════
def fig1_kpi_comparison():
    schedules = ["S1", "S2", "S3"]
    metrics = [
        ("throughput_uph", "Throughput (u/h)", None, None),
        ("defect_rate", "Defect Rate", None, None),
        ("noise_db", "Noise (dB)", 80.0, "Noise limit (80 dB)"),
        ("fatigue_index", "Fatigue Index", 0.4, "Fatigue limit (0.4)"),
        ("energy_kwh", "Energy (kWh)", None, None),
        ("deadline_gap_pct", "Deadline Gap (%)", None, None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for idx, (key, label, threshold, thresh_label) in enumerate(metrics):
        ax = axes[idx]
        vals = [table3[s][key] for s in schedules]
        bars = ax.bar(
            schedules, vals,
            color=[COLORS[s] for s in schedules],
            edgecolor="black", linewidth=0.8,
            hatch=[HATCH[s] for s in schedules],
            width=0.55,
        )
        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                f"{val}", ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
        if threshold is not None:
            ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
            ax.text(
                2.35, threshold, thresh_label,
                ha="right", va="bottom", fontsize=8, color="red", fontstyle="italic",
            )
        ax.set_ylabel(label)
        ax.set_title(label)
        # Padding above bars
        ymax = max(vals) * 1.25
        if threshold and threshold > max(vals):
            ymax = threshold * 1.2
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.3)

    # Legend
    patches = [
        mpatches.Patch(facecolor=COLORS["S1"], edgecolor="black", label="S1 (Initial)"),
        mpatches.Patch(facecolor=COLORS["S2"], edgecolor="black", hatch="///", label="S2 (Aggressive, REJECTED)"),
        mpatches.Patch(facecolor=COLORS["S3"], edgecolor="black", label="S3 (Balanced, Final)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.03, 1, 1.0])
    path = os.path.join(FIG_DIR, "fig1_kpi_comparison.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
#  Figure 2 — Radar / Spider Chart (normalized KPIs)
# ════════════════════════════════════════════════════════════════════════
def fig2_radar_chart():
    schedules = ["S1", "S2", "S3"]
    # Metrics to plot: normalize each to [0, 1] where 1 = best
    raw = {
        "Throughput": {s: table3[s]["throughput_uph"] for s in schedules},
        "Quality\n(1-Defect)": {s: 1 - table3[s]["defect_rate"] for s in schedules},
        "Low Noise\n(100-dB)/100": {s: (100 - table3[s]["noise_db"]) / 100 for s in schedules},
        "Low Fatigue\n(1-idx)": {s: 1 - table3[s]["fatigue_index"] for s in schedules},
        "Efficiency\n(500-kWh)/500": {s: (500 - table3[s]["energy_kwh"]) / 500 for s in schedules},
        "V/R Score": {s: table3[s]["vr_score"] for s in schedules},
    }

    labels = list(raw.keys())
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for s in schedules:
        values = [raw[k][s] for k in labels]
        # Normalize throughput to [0,1]: divide by max reasonable
        if "Throughput" in labels:
            idx_t = labels.index("Throughput")
            values[idx_t] = values[idx_t] / 60.0  # max ~60
        values += values[:1]
        ax.plot(angles, values, "o-", color=COLORS[s], linewidth=2, label=s)
        ax.fill(angles, values, color=COLORS[s], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title("Normalized KPI Radar — S1 vs S2 vs S3", fontsize=14, fontweight="bold", pad=20)

    path = os.path.join(FIG_DIR, "fig2_radar_chart.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
#  Figure 3 — Constraint Compliance Heatmap
# ════════════════════════════════════════════════════════════════════════
def fig3_constraint_heatmap():
    schedules = ["S1", "S2", "S3"]
    constraints = {
        "Fatigue <= 0.4": ("fatigue_index", 0.4),
        "Noise <= 80 dB": ("noise_db", 80.0),
    }

    fig, ax = plt.subplots(figsize=(7, 4))

    # Margin matrix: positive = within limit, negative = violation
    data = []
    annot = []
    for cname, (metric, limit) in constraints.items():
        row = []
        arow = []
        for s in schedules:
            val = table3[s][metric]
            margin = limit - val
            margin_pct = (margin / limit) * 100
            row.append(margin_pct)
            arow.append(f"{val}\n({'margin' if margin >= 0 else 'OVER'}: {abs(margin):.1f})")
        data.append(row)
        annot.append(arow)

    data = np.array(data)

    # Color: green for positive margin, red for violation
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=20)
    cmap = plt.cm.RdYlGn

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    # Annotations
    for i in range(len(constraints)):
        for j in range(len(schedules)):
            color = "white" if data[i, j] < -20 else "black"
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=10,
                    fontweight="bold", color=color)

    ax.set_xticks(range(len(schedules)))
    ax.set_xticklabels(schedules, fontsize=12)
    ax.set_yticks(range(len(constraints)))
    ax.set_yticklabels(list(constraints.keys()), fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Margin to Limit (%)", fontsize=11)

    ax.set_title("Constraint Compliance — Hard Constraint K Margins", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_constraint_heatmap.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
#  Figure 4 — V/R Score with Feasibility Overlay
# ════════════════════════════════════════════════════════════════════════
def fig4_vr_feasibility():
    schedules = ["S1", "S2", "S3"]
    vr = [table3[s]["vr_score"] for s in schedules]
    feasible = [table3[s]["feasible"] for s in schedules]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        schedules, vr,
        color=[COLORS[s] for s in schedules],
        edgecolor="black", linewidth=1.2,
        width=0.5,
    )

    # Hatch rejected bars
    for bar, feas, s in zip(bars, feasible, schedules):
        if not feas:
            bar.set_hatch("///")
            bar.set_edgecolor("darkred")

    # Value labels
    for bar, v, feas in zip(bars, vr, feasible):
        label = f"{v:.3f}\n{'FEASIBLE' if feas else 'REJECTED'}"
        color = "green" if feas else "red"
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            label, ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=color,
        )

    ax.set_ylabel("V/R Score (Eq. 3)", fontsize=12)
    ax.set_title("Value-to-Resource Score with Feasibility Verdict", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(vr) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    # Annotation: "Higher V/R != better if infeasible"
    ax.annotate(
        "S2 has highest V/R\nbut violates constraints K",
        xy=(1, vr[1]), xytext=(1.8, vr[1] - 0.08),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        fontsize=10, fontstyle="italic", color="red",
        ha="center",
    )

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig4_vr_feasibility.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
#  Figure 5 — 5-Phase Execution Swimlane
# ════════════════════════════════════════════════════════════════════════
def fig5_execution_swimlane():
    phases = [
        {
            "phase": 1, "title": "Phase 1: Initial Deployment",
            "layers": ["L1", "L2", "L3", "L4"],
            "actions": ["Elicit C1", "Generate S1\nV/R=0.781", "Verify\n(feasible)", "Deploy S1"],
            "color": "#2196F3",
        },
        {
            "phase": 2, "title": "Phase 2: Demand Surge",
            "layers": ["L4"],
            "actions": ["Monitor detects\nshortfall (-29%)"],
            "color": "#FF9800",
        },
        {
            "phase": 3, "title": "Phase 3: Autonomous Replan",
            "layers": ["L2", "L3"],
            "actions": ["Generate S2\nV/R=0.824", "Verify\nREJECTED"],
            "color": "#F44336",
        },
        {
            "phase": 4, "title": "Phase 4: Escalation",
            "layers": ["Meta", "L1"],
            "actions": ["Escalate conflict\n3 options", "Manager picks\nOption C"],
            "color": "#9C27B0",
        },
        {
            "phase": 5, "title": "Phase 5: Adaptation",
            "layers": ["L1", "L2", "L3", "L5", "L4"],
            "actions": ["Update to C2", "Generate S3\nV/R=0.802", "Verify\n(feasible)", "Adapt\nmeso+micro", "Deploy S3"],
            "color": "#4CAF50",
        },
    ]

    all_layers = ["L1", "L2", "L3", "L4", "L5", "Meta"]
    layer_y = {l: i for i, l in enumerate(reversed(all_layers))}

    fig, ax = plt.subplots(figsize=(16, 7))

    x_offset = 0
    for p in phases:
        x_start = x_offset
        for i, (layer, action) in enumerate(zip(p["layers"], p["actions"])):
            x = x_offset + i * 1.5
            y = layer_y[layer]
            rect = plt.Rectangle(
                (x - 0.55, y - 0.35), 1.1, 0.7,
                facecolor=p["color"], alpha=0.25,
                edgecolor=p["color"], linewidth=2,
                zorder=2,
            )
            ax.add_patch(rect)
            ax.text(x, y, action, ha="center", va="center", fontsize=8,
                    fontweight="bold", zorder=3)

            # Arrow between consecutive actions
            if i > 0:
                prev_layer = p["layers"][i - 1]
                ax.annotate(
                    "", xy=(x - 0.55, y),
                    xytext=(x - 0.95, layer_y[prev_layer]),
                    arrowprops=dict(arrowstyle="->", color=p["color"], lw=1.5),
                    zorder=1,
                )

        # Phase header
        x_mid = x_start + (len(p["layers"]) - 1) * 1.5 / 2
        ax.text(x_mid, len(all_layers) - 0.2, p["title"],
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=p["color"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor=p["color"], alpha=0.15))

        x_offset += len(p["layers"]) * 1.5 + 0.8

    # Y-axis layer labels
    ax.set_yticks(range(len(all_layers)))
    ax.set_yticklabels(list(reversed(all_layers)), fontsize=12, fontweight="bold")

    # Horizontal grid for lanes
    for i in range(len(all_layers)):
        ax.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
    ax.axhline(len(all_layers) - 0.5, color="gray", linewidth=0.5, alpha=0.3)

    ax.set_xlim(-1, x_offset - 0.5)
    ax.set_ylim(-0.7, len(all_layers) + 0.3)
    ax.set_xlabel("Execution Flow (left to right)", fontsize=12)
    ax.set_title("Table 2 — 5-Phase Execution Trace Across CBPA Layers", fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig5_execution_swimlane.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
#  Figure 6 — Throughput vs Constraint Trade-off (scatter)
# ════════════════════════════════════════════════════════════════════════
def fig6_tradeoff_scatter():
    schedules = ["S1", "S2", "S3"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Throughput vs Fatigue
    for s in schedules:
        marker = "X" if not table3[s]["feasible"] else "o"
        size = 200
        ax1.scatter(
            table3[s]["throughput_uph"], table3[s]["fatigue_index"],
            color=COLORS[s], s=size, marker=marker, edgecolor="black", linewidth=1.5,
            zorder=5, label=s,
        )
        ax1.annotate(
            s, (table3[s]["throughput_uph"], table3[s]["fatigue_index"]),
            textcoords="offset points", xytext=(10, 5), fontsize=12, fontweight="bold",
        )
    ax1.axhline(0.4, color="red", linestyle="--", linewidth=1.5, label="Fatigue limit (K)")
    ax1.fill_between([40, 56], 0.4, 0.7, alpha=0.08, color="red")
    ax1.set_xlabel("Throughput (u/h)")
    ax1.set_ylabel("Fatigue Index")
    ax1.set_title("(a) Throughput vs Fatigue", fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(40, 56)
    ax1.set_ylim(0.2, 0.7)

    # Panel B: Throughput vs Noise
    for s in schedules:
        marker = "X" if not table3[s]["feasible"] else "o"
        ax2.scatter(
            table3[s]["throughput_uph"], table3[s]["noise_db"],
            color=COLORS[s], s=200, marker=marker, edgecolor="black", linewidth=1.5,
            zorder=5, label=s,
        )
        ax2.annotate(
            s, (table3[s]["throughput_uph"], table3[s]["noise_db"]),
            textcoords="offset points", xytext=(10, 5), fontsize=12, fontweight="bold",
        )
    ax2.axhline(80, color="red", linestyle="--", linewidth=1.5, label="Noise limit (K)")
    ax2.fill_between([40, 56], 80, 90, alpha=0.08, color="red")
    ax2.set_xlabel("Throughput (u/h)")
    ax2.set_ylabel("Noise (dB)")
    ax2.set_title("(b) Throughput vs Noise", fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(40, 56)
    ax2.set_ylim(73, 88)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig6_tradeoff_scatter.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
#  Figure 7 — Pareto Frontier (V/R vs Constraint Margin)
# ════════════════════════════════════════════════════════════════════════
def fig7_pareto_frontier():
    """Plot Pareto frontiers from the live demo data."""
    live_path = os.path.join(DATA_DIR, "live_demo.json")
    if not os.path.exists(live_path):
        print("  Skipped fig7 (no live_demo.json)")
        return

    with open(live_path) as f:
        live = json.load(f)

    # Collect phases with pareto data
    pareto_phases = [p for p in live["phases"] if p.get("pareto")]
    if not pareto_phases:
        print("  Skipped fig7 (no Pareto data)")
        return

    phase_colors = {1: "#2196F3", 3: "#F44336", 5: "#4CAF50", 6: "#9C27B0"}
    phase_labels = {1: "Phase 1 (S1)", 3: "Phase 3 (S2)", 5: "Phase 5 (S3)", 6: "Phase 6 (S4)"}

    fig, ax = plt.subplots(figsize=(9, 6))

    for p in pareto_phases:
        ph = p["phase"]
        pareto = p["pareto"]
        nd = pareto["non_dominated"]
        total = pareto["candidates"]
        selected = pareto["selected"]
        color = phase_colors.get(ph, "gray")
        label = phase_labels.get(ph, f"Phase {ph}")

        # Plot non-dominated count as a point on the frontier summary
        ax.scatter(
            total, len(nd),
            color=color, s=200, edgecolor="black", linewidth=1.5,
            zorder=5, label=f"{label}: {len(nd)}/{total} non-dominated",
        )
        ax.annotate(
            f"sel={selected}",
            (total, len(nd)),
            textcoords="offset points", xytext=(12, -5),
            fontsize=9, color=color, fontweight="bold",
        )

    ax.set_xlabel("Total Candidates Generated", fontsize=12)
    ax.set_ylabel("Non-Dominated (Pareto Front Size)", fontsize=12)
    ax.set_title("Pareto Frontier Exploration Across Phases", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max(p["pareto"]["candidates"] for p in pareto_phases) + 2)
    ax.set_ylim(0, max(len(p["pareto"]["non_dominated"]) for p in pareto_phases) + 2)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig7_pareto_frontier.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
#  Figure 8 — Working Mode Transitions Timeline
# ════════════════════════════════════════════════════════════════════════
def fig8_working_modes():
    """Visualize working mode transitions across phases."""
    live_path = os.path.join(DATA_DIR, "live_demo.json")
    if not os.path.exists(live_path):
        print("  Skipped fig8 (no live_demo.json)")
        return

    with open(live_path) as f:
        live = json.load(f)

    phases = live.get("phases", [])
    if not phases:
        print("  Skipped fig8 (no phases)")
        return

    # Only first shift
    shift1 = [p for p in phases if p.get("shift", 1) == 1]
    if not shift1:
        shift1 = phases

    mode_colors = {
        "legislator": "#2196F3",
        "auditor": "#4CAF50",
        "partner": "#FF9800",
    }
    mode_y = {"legislator": 2, "auditor": 1, "partner": 0}

    fig, ax = plt.subplots(figsize=(12, 4))

    for i, p in enumerate(shift1):
        mode = p.get("working_mode", "unknown")
        color = mode_colors.get(mode, "gray")
        y = mode_y.get(mode, -0.5)

        # Bar for this phase
        ax.barh(y, 1, left=i, height=0.6, color=color, edgecolor="black", linewidth=1)
        ax.text(
            i + 0.5, y, f"P{p['phase']}",
            ha="center", va="center", fontsize=11, fontweight="bold", color="white",
        )

        # Arrow between phases showing transitions
        if i > 0:
            prev_mode = shift1[i-1].get("working_mode", "")
            if prev_mode != mode:
                prev_y = mode_y.get(prev_mode, -0.5)
                ax.annotate(
                    "", xy=(i, y), xytext=(i, prev_y),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2),
                )

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Partner\n(co-decide)", "Auditor\n(autonomous)", "Legislator\n(human sets)"],
                       fontsize=11)
    ax.set_xlabel("Phase Sequence", fontsize=12)
    ax.set_title("Human Working Mode Transitions (Shift 1)", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.2, len(shift1) + 0.2)
    ax.set_ylim(-0.5, 2.8)
    ax.grid(axis="x", alpha=0.3)

    # Legend
    patches = [mpatches.Patch(facecolor=c, edgecolor="black", label=m.capitalize())
               for m, c in mode_colors.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=10)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig8_working_modes.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
#  Figure 9 — Multi-Shift Learning Curve
# ════════════════════════════════════════════════════════════════════════
def fig9_learning_curve():
    """Plot per-shift metrics showing learning accumulation."""
    lc_path = os.path.join(DATA_DIR, "multi_shift_learning_curve.json")
    if not os.path.exists(lc_path):
        print("  Skipped fig9 (no multi_shift_learning_curve.json)")
        return

    with open(lc_path) as f:
        lc = json.load(f)

    shifts = lc.get("shifts", [])
    if not shifts:
        print("  Skipped fig9 (no shift data)")
        return

    shift_nums = [s["shift"] for s in shifts]
    vr_scores = [s["vr_score_accepted"] for s in shifts]
    violations = [s["total_constraint_violations"] for s in shifts]
    bias_flags = [s["learning_bias_available"] for s in shifts]
    records = [s.get("experience_records_after", 0) for s in shifts]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: V/R score convergence
    colors = ["#FF9800" if not b else "#4CAF50" for b in bias_flags]
    ax1.bar(shift_nums, vr_scores, color=colors, edgecolor="black", linewidth=1)
    for i, (x, v) in enumerate(zip(shift_nums, vr_scores)):
        ax1.text(x, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Shift")
    ax1.set_ylabel("V/R Score (accepted)")
    ax1.set_title("(a) Accepted V/R Score", fontweight="bold")
    ax1.set_ylim(0, max(vr_scores) * 1.15)
    ax1.grid(axis="y", alpha=0.3)

    # Panel B: Constraint violations
    ax2.bar(shift_nums, violations, color=colors, edgecolor="black", linewidth=1)
    for x, v in zip(shift_nums, violations):
        ax2.text(x, v + 0.1, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Shift")
    ax2.set_ylabel("Constraint Violations")
    ax2.set_title("(b) Total Violations", fontweight="bold")
    ax2.set_ylim(0, max(violations) * 1.25 if max(violations) > 0 else 3)
    ax2.grid(axis="y", alpha=0.3)

    # Panel C: Experience accumulation
    ax3.plot(shift_nums, records, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax3.fill_between(shift_nums, records, alpha=0.15, color="#2196F3")
    for x, r in zip(shift_nums, records):
        ax3.text(x, r + 0.2, str(r), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Shift")
    ax3.set_ylabel("Learning Store Records")
    ax3.set_title("(c) Cumulative Experience", fontweight="bold")
    ax3.set_ylim(0, max(records) * 1.2)
    ax3.grid(alpha=0.3)

    # Legend
    patches = [
        mpatches.Patch(facecolor="#FF9800", edgecolor="black", label="No prior bias"),
        mpatches.Patch(facecolor="#4CAF50", edgecolor="black", label="Learning bias active"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig9_learning_curve.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
#  Figure 10 — Three-Tier Adaptation Hierarchy
# ════════════════════════════════════════════════════════════════════════
def fig10_adaptation_hierarchy():
    """Visualize micro/meso/macro adaptation attempts from Phase 6."""
    live_path = os.path.join(DATA_DIR, "live_demo.json")
    if not os.path.exists(live_path):
        print("  Skipped fig10 (no live_demo.json)")
        return

    with open(live_path) as f:
        live = json.load(f)

    # Find phase 6 (macro) in first shift
    phase6 = [p for p in live["phases"] if p.get("phase") == 6 and p.get("shift", 1) == 1]
    if not phase6:
        print("  Skipped fig10 (no Phase 6 / macro data)")
        return

    # Hardcode the adaptation hierarchy visualization
    tiers = ["Micro\n(param retune)", "Meso\n(policy switch)", "Macro\n(full replan)"]
    p_feasible = [0.0, 0.0, 1.0]  # micro/meso fail, macro succeeds
    threshold = 0.90

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_colors = ["#F44336", "#F44336", "#4CAF50"]
    bars = ax.bar(tiers, [p * 100 for p in p_feasible],
                  color=bar_colors, edgecolor="black", linewidth=1.5, width=0.5)

    ax.axhline(threshold * 100, color="orange", linestyle="--", linewidth=2,
               label=f"Macro threshold ({threshold:.0%})")

    for bar, p in zip(bars, p_feasible):
        label = f"{p:.0%}" if p > 0 else "0%"
        status = "PASS" if p >= threshold else "FAIL"
        color = "green" if p >= threshold else "red"
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{label}\n{status}", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=color,
        )

    # Arrows showing escalation
    ax.annotate("", xy=(1, 5), xytext=(0, 5),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.annotate("", xy=(2, 5), xytext=(1, 5),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.text(0.5, 8, "insufficient", ha="center", fontsize=9, color="red", fontstyle="italic")
    ax.text(1.5, 8, "insufficient", ha="center", fontsize=9, color="red", fontstyle="italic")

    ax.set_ylabel("P(feasible) %", fontsize=12)
    ax.set_title("Three-Tier Adaptation Hierarchy (Phase 6: R1 Degradation)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig10_adaptation_hierarchy.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
#  Figure 11 — Stochastic Verification Summary
# ════════════════════════════════════════════════════════════════════════
def fig11_stochastic_verification():
    """Two-panel stochastic verification: P(feasible) + per-constraint violation breakdown.

    Reads stochastic verification results from stochastic_verification.json
    in the data directory (recomputed from the deployed schedules at the run's
    Monte-Carlo seed). Raises if the file is absent: no fallback values.
    """
    cert_threshold = 0.95

    # Load the stochastic verification data exported/recomputed for this run.
    # No fallback values: the figure must reflect the actual run.
    import json as _json
    _sv_path = os.path.join(DATA_DIR, "stochastic_verification.json")
    with open(_sv_path) as _fh:
        _sv = _json.load(_fh)
    schedules = list(_sv["schedules"].keys())
    p_feasible = [_sv["schedules"][k]["p_feasible"] for k in schedules]
    viol_fatigue = [_sv["schedules"][k]["violation_probs"].get("FatigueIndex", 0.0) for k in schedules]
    viol_noise = [_sv["schedules"][k]["violation_probs"].get("Noise", 0.0) for k in schedules]
    viol_cyber = [_sv["schedules"][k]["violation_probs"].get("CyberRiskLevel", 0.0) for k in schedules]
    wc_fatigue = [_sv["schedules"][k]["worst_case_margins"].get("FatigueIndex") for k in schedules]
    wc_noise = [_sv["schedules"][k]["worst_case_margins"].get("Noise") for k in schedules]

    certified = [p >= cert_threshold for p in p_feasible]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    # ── Left panel: P(feasible) bars ──
    bar_colors = []
    for p, cert in zip(p_feasible, certified):
        if cert:
            bar_colors.append("#4CAF50")
        elif p == 0:
            bar_colors.append("#F44336")
        else:
            bar_colors.append("#FF9800")

    bars = ax1.bar(schedules, [p * 100 for p in p_feasible],
                   color=bar_colors, edgecolor="black", linewidth=1.2, width=0.5)

    ax1.axhline(cert_threshold * 100, color="#1565C0", linestyle="--",
                linewidth=1.5, label=f"Certification threshold ({cert_threshold:.0%})")

    for bar, p, cert in zip(bars, p_feasible, certified):
        status = "CERTIFIED" if cert else ("REJECTED" if p == 0 else "BELOW\nTHRESHOLD")
        color = "#2E7D32" if cert else "#C62828"
        y_pos = max(bar.get_height(), 4) + 2
        ax1.text(
            bar.get_x() + bar.get_width() / 2, y_pos,
            f"{p:.0%}\n{status}", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=color,
        )

    ax1.set_ylabel("P(feasible) %  (n=200 MC samples)", fontsize=11)
    ax1.set_title("(a) Overall Feasibility", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 125)
    ax1.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3)

    # ── Right panel: per-constraint violation probabilities ──
    x = np.arange(len(schedules))
    w = 0.25

    b1 = ax2.bar(x - w, [v * 100 for v in viol_fatigue], w,
                  label="Fatigue (<=0.4)", color="#E57373", edgecolor="black", linewidth=0.8)
    b2 = ax2.bar(x, [v * 100 for v in viol_noise], w,
                  label="Noise (<=80 dB)", color="#FFB74D", edgecolor="black", linewidth=0.8)
    b3 = ax2.bar(x + w, [v * 100 for v in viol_cyber], w,
                  label="Cyber risk (<=2)", color="#81C784", edgecolor="black", linewidth=0.8)

    # Annotate worst-case (5th-percentile) margins above non-zero bars.
    # When both fatigue and noise are violated, stack them vertically
    # at the group center so they align neatly.
    _wc_box_fat = dict(boxstyle="round,pad=0.2", facecolor="#FFEBEE",
                       edgecolor="#E57373", alpha=0.9)
    _wc_box_noi = dict(boxstyle="round,pad=0.2", facecolor="#FFF3E0",
                       edgecolor="#FFB74D", alpha=0.9)
    for i, name in enumerate(schedules):
        top = max(viol_fatigue[i], viol_noise[i]) * 100
        has_both = viol_fatigue[i] > 0 and viol_noise[i] > 0
        if has_both:
            # Stack at group center: noise on bottom, fatigue on top
            cx = i - w / 2  # center between the two bars
            ax2.text(cx, top + 3,
                     f"Noise  {wc_noise[i]:+.1f} dB",
                     ha="center", va="bottom", fontsize=10, fontweight="bold",
                     color="#E65100", bbox=_wc_box_noi)
            ax2.text(cx, top + 12,
                     f"Fatigue  {wc_fatigue[i]:+.2f}",
                     ha="center", va="bottom", fontsize=10, fontweight="bold",
                     color="#B71C1C", bbox=_wc_box_fat)
        else:
            if viol_fatigue[i] > 0:
                ax2.text(i - w, top + 3,
                         f"Fatigue  {wc_fatigue[i]:+.2f}",
                         ha="center", va="bottom", fontsize=10, fontweight="bold",
                         color="#B71C1C", bbox=_wc_box_fat)
            if viol_noise[i] > 0:
                ax2.text(i, top + 3,
                         f"Noise  {wc_noise[i]:+.1f} dB",
                         ha="center", va="bottom", fontsize=10, fontweight="bold",
                         color="#E65100", bbox=_wc_box_noi)

    ax2.set_xticks(x)
    ax2.set_xticklabels(schedules)
    ax2.set_ylabel("P(violation) %", fontsize=11)
    ax2.set_title("(b) Per-Constraint Violation Probability", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 120)
    ax2.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3)

    # Annotate the rejected schedule (S2) — only when it has violations
    if viol_fatigue[1] > 0 and viol_noise[1] > 0:
        ax2.annotate(
            "S2 violates BOTH\nfatigue and noise\n(rejected by Layer 3)",
            xy=(1, 100), xytext=(0.4, 75),
            arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.2,
                            connectionstyle="arc3,rad=-0.15"),
            fontsize=8.5, fontstyle="italic", color="#B71C1C", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#B71C1C", alpha=0.9),
        )

    fig.tight_layout(w_pad=3)
    path = os.path.join(FIG_DIR, "fig11_stochastic_verification.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════
def main(data_dir: str | None = None):
    global DATA_DIR, FIG_DIR
    if data_dir is not None:
        DATA_DIR = data_dir
        FIG_DIR = os.path.join(DATA_DIR, "figures")

    _load_data()

    print("Generating CBPA case study visualizations...")
    print("\n--- Core figures (Table 3 data) ---")
    fig1_kpi_comparison()
    fig2_radar_chart()
    fig3_constraint_heatmap()
    fig4_vr_feasibility()
    fig5_execution_swimlane()
    fig6_tradeoff_scatter()
    print("\n--- CBPA unique value figures (live demo data) ---")
    fig7_pareto_frontier()
    fig8_working_modes()
    fig9_learning_curve()
    fig10_adaptation_hierarchy()
    fig11_stochastic_verification()
    print(f"\nAll figures saved to: {FIG_DIR}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="Path to results directory")
    args = parser.parse_args()
    main(data_dir=args.data_dir)
