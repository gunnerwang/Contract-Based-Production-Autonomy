"""Page 2: Results Explorer — Table 3 + figures + V/R breakdown + export."""

from __future__ import annotations

import streamlit as st

from cbpa.service.export_service import ExportService
from cbpa.ui.state import is_dev_mode
from cbpa.ui.theme import SCHEDULE_COLORS, SCHEDULE_DISPLAY_NAMES


def _table3_section(result) -> None:
    """Display Table 3 as a dataframe with download buttons."""
    import pandas as pd

    rows = []
    for name in ["S1", "S2", "S3"]:
        m = result.schedule_metrics[name]
        vr = result.schedule_vr[name]
        f = result.schedule_feasibility[name]
        rows.append({
            "Plan": name,
            "Throughput (u/h)": m.throughput_uph,
            "Defect Rate": m.defect_rate,
            "Noise (dB)": m.noise_db,
            "Fatigue Index": m.fatigue_index,
            "Energy (kWh)": m.energy_kwh,
            "Deadline Gap (%)": m.deadline_gap_pct,
            "V/R Score": vr.vr_score,
            "Feasible": "Yes" if f.is_feasible else "No",
        })

    df = pd.DataFrame(rows)

    def _highlight_s2(row):
        if row["Plan"] == "S2":
            return ["background-color: #FFEBEE"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(_highlight_s2, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # Download buttons
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.download_button(
            "Download CSV",
            ExportService.table3_csv(result),
            file_name="table3.csv",
            mime="text/csv",
        )
    with dc2:
        st.download_button(
            "Download LaTeX",
            ExportService.table3_latex(result),
            file_name="table3.tex",
            mime="text/plain",
        )
    with dc3:
        st.download_button(
            "Download JSON",
            ExportService.table3_json(result),
            file_name="table3.json",
            mime="application/json",
        )


def _figures_section(result) -> None:
    """Render the 6 publication figures using matplotlib + optional Plotly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    table3 = {}
    for name in ["S1", "S2", "S3"]:
        m = result.schedule_metrics[name]
        vr = result.schedule_vr[name]
        f = result.schedule_feasibility[name]
        table3[name] = {
            "throughput_uph": m.throughput_uph,
            "defect_rate": m.defect_rate,
            "noise_db": m.noise_db,
            "fatigue_index": m.fatigue_index,
            "energy_kwh": m.energy_kwh,
            "deadline_gap_pct": m.deadline_gap_pct,
            "vr_score": vr.vr_score,
            "feasible": f.is_feasible,
        }

    COLORS = SCHEDULE_COLORS
    HATCH = {"S1": "", "S2": "///", "S3": ""}
    schedules = ["S1", "S2", "S3"]

    # Fig 1: KPI Comparison
    st.subheader("Fig 1: Schedule KPI Comparison")
    metrics_list = [
        ("throughput_uph", "Throughput (u/h)", None),
        ("defect_rate", "Defect Rate", None),
        ("noise_db", "Noise (dB)", 80.0),
        ("fatigue_index", "Fatigue Index", 0.4),
        ("energy_kwh", "Energy (kWh)", None),
        ("deadline_gap_pct", "Deadline Gap (%)", None),
    ]
    fig1, axes = plt.subplots(2, 3, figsize=(14, 8))
    for idx, (key, label, thresh) in enumerate(metrics_list):
        ax = axes.ravel()[idx]
        vals = [table3[s][key] for s in schedules]
        bars = ax.bar(schedules, vals, color=[COLORS[s] for s in schedules],
                       edgecolor="black", hatch=[HATCH[s] for s in schedules], width=0.55)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    f"{val}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        if thresh is not None:
            ax.axhline(thresh, color="red", linestyle="--", linewidth=1.5)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # Fig 2: Radar chart
    st.subheader("Fig 2: Normalized KPI Radar")
    raw = {
        "Throughput": {s: table3[s]["throughput_uph"] / 60.0 for s in schedules},
        "Quality": {s: 1 - table3[s]["defect_rate"] for s in schedules},
        "Low Noise": {s: (100 - table3[s]["noise_db"]) / 100 for s in schedules},
        "Low Fatigue": {s: 1 - table3[s]["fatigue_index"] for s in schedules},
        "Efficiency": {s: (500 - table3[s]["energy_kwh"]) / 500 for s in schedules},
        "V/R Score": {s: table3[s]["vr_score"] for s in schedules},
    }
    labels = list(raw.keys())
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig2, ax2 = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for s in schedules:
        values = [raw[k][s] for k in labels] + [raw[labels[0]][s]]
        ax2.plot(angles, values, "o-", color=COLORS[s], linewidth=2, label=s)
        ax2.fill(angles, values, color=COLORS[s], alpha=0.1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax2.set_title("Normalized KPI Radar", fontweight="bold", pad=20)
    st.pyplot(fig2)
    plt.close(fig2)

    # Fig 3: Constraint heatmap
    st.subheader("Fig 3: Constraint Compliance Heatmap")
    constraints = {"Fatigue <= 0.4": ("fatigue_index", 0.4), "Noise <= 80 dB": ("noise_db", 80.0)}
    data = []
    annot = []
    for cname, (metric, limit) in constraints.items():
        row_data = []
        row_annot = []
        for s in schedules:
            val = table3[s][metric]
            margin = limit - val
            margin_pct = (margin / limit) * 100 if limit else 0
            row_data.append(margin_pct)
            row_annot.append(f"{val}\n({'margin' if margin >= 0 else 'OVER'}: {abs(margin):.1f})")
        data.append(row_data)
        annot.append(row_annot)

    data_arr = np.array(data)
    from matplotlib.colors import TwoSlopeNorm
    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=20)
    im = ax3.imshow(data_arr, cmap=plt.cm.RdYlGn, norm=norm, aspect="auto")
    for i in range(len(constraints)):
        for j in range(len(schedules)):
            color = "white" if data_arr[i, j] < -20 else "black"
            ax3.text(j, i, annot[i][j], ha="center", va="center", fontsize=10, fontweight="bold", color=color)
    ax3.set_xticks(range(len(schedules)))
    ax3.set_xticklabels(schedules)
    ax3.set_yticks(range(len(constraints)))
    ax3.set_yticklabels(list(constraints.keys()))
    fig3.colorbar(im, ax=ax3, shrink=0.8, label="Margin to Limit (%)")
    ax3.set_title("Constraint Compliance Heatmap", fontweight="bold")
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # Fig 4: V/R with feasibility
    st.subheader("Fig 4: V/R Score with Feasibility")
    vr_vals = [table3[s]["vr_score"] for s in schedules]
    feas_vals = [table3[s]["feasible"] for s in schedules]
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    bars4 = ax4.bar(schedules, vr_vals, color=[COLORS[s] for s in schedules], edgecolor="black", width=0.5)
    for bar, v, feas, s in zip(bars4, vr_vals, feas_vals, schedules):
        if not feas:
            bar.set_hatch("///")
        label = f"{v:.3f}\n{'FEASIBLE' if feas else 'REJECTED'}"
        color = "green" if feas else "red"
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 label, ha="center", va="bottom", fontsize=11, fontweight="bold", color=color)
    ax4.set_ylabel("V/R Score")
    ax4.set_title("V/R Score with Feasibility", fontweight="bold")
    ax4.set_ylim(0, max(vr_vals) * 1.25)
    ax4.grid(axis="y", alpha=0.3)
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    # Fig 5: Execution swimlane
    st.subheader("Fig 5: CBPA Lifecycle Swimlane")
    phases_data = [
        {"phase": 1, "title": "Round 1: Contract & Deploy", "layers": ["L1", "L2", "L3", "L4"],
         "actions": ["Elicit C1", f"Generate S1\nV/R={table3['S1']['vr_score']:.3f}", "Verify\n(feasible)", "Deploy S1"],
         "color": "#2196F3"},
        {"phase": 2, "title": "Round 2: Monitor", "layers": ["L4"], "actions": ["Monitor\nshortfall"], "color": "#FF9800"},
        {"phase": 3, "title": "Round 3: Replan", "layers": ["L2", "L3"],
         "actions": [f"Generate S2\nV/R={table3['S2']['vr_score']:.3f}", "Verify\nREJECTED"], "color": "#F44336"},
        {"phase": 4, "title": "Round 4: Escalate", "layers": ["Meta", "L1"],
         "actions": ["Escalate\n3 options", "Manager\nOption C"], "color": "#9C27B0"},
        {"phase": 5, "title": "Round 5: Adapt", "layers": ["L1", "L2", "L3", "L5", "L4"],
         "actions": ["Update C2", f"Generate S3\nV/R={table3['S3']['vr_score']:.3f}", "Verify\n(feasible)", "Adapt", "Deploy S3"],
         "color": "#4CAF50"},
    ]
    all_layers = ["L1", "L2", "L3", "L4", "L5", "Meta"]
    layer_y = {l: i for i, l in enumerate(reversed(all_layers))}
    fig5, ax5 = plt.subplots(figsize=(16, 7))
    x_offset = 0
    for p in phases_data:
        for i, (layer, action) in enumerate(zip(p["layers"], p["actions"])):
            x = x_offset + i * 1.5
            y = layer_y[layer]
            rect = plt.Rectangle((x-0.55, y-0.35), 1.1, 0.7, facecolor=p["color"], alpha=0.25,
                                  edgecolor=p["color"], linewidth=2, zorder=2)
            ax5.add_patch(rect)
            ax5.text(x, y, action, ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)
            if i > 0:
                prev_layer = p["layers"][i-1]
                ax5.annotate("", xy=(x-0.55, y), xytext=(x-0.95, layer_y[prev_layer]),
                             arrowprops=dict(arrowstyle="->", color=p["color"], lw=1.5))
        x_mid = x_offset + (len(p["layers"])-1) * 1.5 / 2
        ax5.text(x_mid, len(all_layers)-0.2, p["title"], ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color=p["color"],
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=p["color"], alpha=0.15))
        x_offset += len(p["layers"]) * 1.5 + 0.8
    ax5.set_yticks(range(len(all_layers)))
    ax5.set_yticklabels(list(reversed(all_layers)), fontsize=12, fontweight="bold")
    for i in range(len(all_layers)):
        ax5.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
    ax5.set_xlim(-1, x_offset - 0.5)
    ax5.set_ylim(-0.7, len(all_layers) + 0.3)
    ax5.set_title("5-Phase Execution Trace", fontsize=15, fontweight="bold")
    ax5.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    fig5.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    # Fig 6: Tradeoff scatter
    st.subheader("Fig 6: Throughput-Constraint Trade-off")
    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(13, 5.5))
    for s in schedules:
        marker = "X" if not table3[s]["feasible"] else "o"
        ax6a.scatter(table3[s]["throughput_uph"], table3[s]["fatigue_index"],
                     color=COLORS[s], s=200, marker=marker, edgecolor="black", linewidth=1.5, label=s)
        ax6a.annotate(s, (table3[s]["throughput_uph"], table3[s]["fatigue_index"]),
                      textcoords="offset points", xytext=(10, 5), fontsize=12, fontweight="bold")
        ax6b.scatter(table3[s]["throughput_uph"], table3[s]["noise_db"],
                     color=COLORS[s], s=200, marker=marker, edgecolor="black", linewidth=1.5, label=s)
        ax6b.annotate(s, (table3[s]["throughput_uph"], table3[s]["noise_db"]),
                      textcoords="offset points", xytext=(10, 5), fontsize=12, fontweight="bold")
    ax6a.axhline(0.4, color="red", linestyle="--", linewidth=1.5, label="Fatigue limit")
    ax6a.set_xlabel("Throughput (u/h)")
    ax6a.set_ylabel("Fatigue Index")
    ax6a.set_title("(a) Throughput vs Fatigue", fontweight="bold")
    ax6a.legend()
    ax6a.grid(alpha=0.3)
    ax6b.axhline(80, color="red", linestyle="--", linewidth=1.5, label="Noise limit")
    ax6b.set_xlabel("Throughput (u/h)")
    ax6b.set_ylabel("Noise (dB)")
    ax6b.set_title("(b) Throughput vs Noise", fontweight="bold")
    ax6b.legend()
    ax6b.grid(alpha=0.3)
    fig6.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)


def _vr_breakdown_section(result) -> None:
    """V/R component breakdown with weight sensitivity slider."""
    st.subheader("V/R Component Breakdown")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    schedules = ["S1", "S2", "S3"]
    COLORS = SCHEDULE_COLORS

    # Stacked bar of V/R components
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Value components
    for s in schedules:
        vr = result.schedule_vr[s]
        breakdown = vr.component_breakdown
        val_keys = [k for k in breakdown if k.startswith("v_")]
        if not val_keys:
            val_keys = ["value"]

    # Simple V vs R bar
    for i, s in enumerate(schedules):
        vr = result.schedule_vr[s]
        ax1.bar(i, vr.value_numerator, color=COLORS[s], alpha=0.7, label=f"{s} Value" if i == 0 else "")
        ax1.bar(i, -vr.resource_denominator, color=COLORS[s], alpha=0.4)
    ax1.set_xticks(range(len(schedules)))
    ax1.set_xticklabels(schedules)
    ax1.set_ylabel("Value (+) / Resource (-)")
    ax1.set_title("V/R Component Stacks", fontweight="bold")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.grid(axis="y", alpha=0.3)

    vr_scores = [result.schedule_vr[s].vr_score for s in schedules]
    bars = ax2.bar(schedules, vr_scores, color=[COLORS[s] for s in schedules], edgecolor="black")
    for bar, v in zip(bars, vr_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
    ax2.set_ylabel("V/R Score")
    ax2.set_title("V/R Score Comparison", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _export_section(result) -> None:
    """Full export section with ZIP download."""
    st.subheader("Export")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Full Results JSON",
            ExportService.full_results_json(result),
            file_name="cbpa_full_results.json",
            mime="application/json",
        )
    with col2:
        st.download_button(
            "Download All (ZIP)",
            ExportService.results_zip(result),
            file_name="cbpa_results.zip",
            mime="application/zip",
        )


# ── Operator-mode views ──────────────────────────────────────────────

def _operator_summary(result) -> None:
    """Plain-language summary with Plan A/B/C side-by-side cards."""
    # Determine winner
    best = None
    for name in ["S3", "S1"]:
        f = result.schedule_feasibility.get(name)
        if f and f.is_feasible:
            best = name
            break

    if best:
        st.success(f"A safe and effective plan was found: **{SCHEDULE_DISPLAY_NAMES.get(best, best)}**")
    else:
        st.warning("No plan met all safety limits. Manager review may be needed.")

    # Side-by-side plan cards
    cols = st.columns(3)
    for col, name in zip(cols, ["S1", "S2", "S3"]):
        m = result.schedule_metrics[name]
        f = result.schedule_feasibility[name]
        display_name = SCHEDULE_DISPLAY_NAMES.get(name, name)
        with col:
            safe = f.is_feasible
            badge = "SAFE" if safe else "NOT SAFE"
            if safe:
                st.success(f"**{display_name}**")
            else:
                st.error(f"**{display_name}**")

            st.metric("Production Rate", f"{m.throughput_uph:.1f} u/h")
            st.metric("Worker Fatigue", f"{m.fatigue_index:.2f}")
            st.metric("Noise Level", f"{m.noise_db:.1f} dB")
            st.metric("Energy Usage", f"{m.energy_kwh:.0f} kWh")
            st.metric("Defect Rate", f"{m.defect_rate:.4f}")
            st.metric("Deadline Gap", f"{m.deadline_gap_pct:.1f} %")
            vr = result.schedule_vr[name]
            st.metric("Efficiency Score", f"{vr.vr_score:.3f}")
            if safe:
                st.markdown(f"**{badge}**")
            else:
                st.markdown(f"**:red[{badge}]**")

    # Simple CSV download
    st.markdown("---")
    st.download_button(
        "Download Results (CSV)",
        ExportService.table3_csv(result),
        file_name="results.csv",
        mime="text/csv",
    )


def _operator_charts(result) -> None:
    """4-panel bar chart with operator-friendly labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schedules = ["S1", "S2", "S3"]
    display_labels = [SCHEDULE_DISPLAY_NAMES.get(s, s) for s in schedules]
    COLORS = [SCHEDULE_COLORS[s] for s in schedules]

    metrics_info = [
        ("throughput_uph", "Production Rate (u/h)", None),
        ("fatigue_index", "Worker Fatigue", 0.4),
        ("noise_db", "Noise Level (dB)", 80.0),
        ("energy_kwh", "Energy Usage (kWh)", None),
        ("defect_rate", "Defect Rate", 0.01),
        ("deadline_gap_pct", "Deadline Gap (%)", None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for idx, (key, label, limit) in enumerate(metrics_info):
        ax = axes.ravel()[idx]
        vals = [getattr(result.schedule_metrics[s], key) for s in schedules]
        bars = ax.bar(display_labels, vals, color=COLORS, edgecolor="black", width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        if limit is not None:
            ax.axhline(limit, color="red", linestyle="--", linewidth=1.5, label=f"Limit ({limit})")
            ax.legend(fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(0, max(vals) * 1.3)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Plan Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render() -> None:
    """Main render function for the Results Explorer page."""
    st.header("Results Explorer" if is_dev_mode() else "Results")

    result = st.session_state.get("experiment_result")
    if result is None:
        msg = ("Run an experiment first (Experiment Dashboard > Phase Execution)."
               if is_dev_mode()
               else "Run an experiment first from the Dashboard.")
        st.info(msg)
        return

    if not is_dev_mode():
        st.caption("For shift review, see the **Auditor** tab on the Shift Dashboard.")

    if is_dev_mode():
        tab1, tab2, tab3, tab4 = st.tabs(["Table 3", "Figures", "V/R Breakdown", "Export"])
        with tab1:
            _table3_section(result)
        with tab2:
            _figures_section(result)
        with tab3:
            _vr_breakdown_section(result)
        with tab4:
            _export_section(result)
    else:
        tab1, tab2 = st.tabs(["Summary", "Charts"])
        with tab1:
            _operator_summary(result)
        with tab2:
            _operator_charts(result)
