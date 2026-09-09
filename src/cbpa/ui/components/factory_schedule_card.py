"""Factory schedule display card — shows per-cell parameters and factory metrics."""

from __future__ import annotations

import streamlit as st

from cbpa.models.metrics import FactoryMetrics
from cbpa.models.schedule import FactorySchedule


def factory_schedule_card(
    schedule: FactorySchedule,
    metrics: FactoryMetrics | None = None,
) -> None:
    """Render a card showing per-cell schedule parameters and factory KPIs."""
    st.markdown(
        f'<div style="border-left:4px solid #1565C0;padding:8px 12px;'
        f'margin-bottom:8px;background:#fafafa;border-radius:4px;">'
        f'<strong style="color:#1565C0;">{schedule.name}</strong>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Per-cell parameters
    cols = st.columns(len(schedule.cell_schedules))
    for col, (cell_id, cs) in zip(cols, schedule.cell_schedules.items()):
        with col:
            st.markdown(f"**Cell {cell_id}** ({cs.assigned_operator})")
            st.markdown(f"R1: {cs.r1_speed_fraction:.0%} | R2: {cs.r2_speed_fraction:.0%}")
            st.markdown(f"Human: {cs.human_cycle_rate_multiplier:.2f}x | Buf: {cs.buffer_time_s:.1f}s")
            st.markdown(f"Variants: {', '.join(cs.assigned_variants)}")

    # Routing
    if schedule.variant_routing:
        routing_str = ", ".join(
            f"{v}→{'→'.join(cells) if cells else '(suspended)'}"
            for v, cells in schedule.variant_routing.items()
        )
        st.caption(f"Routing: {routing_str}")

    # Factory metrics
    if metrics is not None:
        st.markdown("---")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Total Throughput", f"{metrics.total_throughput_uph:.1f} u/h")
        with mc2:
            st.metric("Factory Noise", f"{metrics.factory_noise_db:.1f} dB")
        with mc3:
            st.metric("Cell Balance", f"{metrics.cell_balance_loss_pct:.1f}%")
        with mc4:
            st.metric("AGV Util.", f"{metrics.agv_utilization:.1%}")

        # Per-operator fatigue
        if metrics.operator_fatigue:
            fat_cols = st.columns(len(metrics.operator_fatigue))
            for col, (op_id, fat) in zip(fat_cols, metrics.operator_fatigue.items()):
                with col:
                    color = "normal" if fat <= 0.35 else ("off" if fat <= 0.4 else "inverse")
                    st.metric(f"{op_id} Fatigue", f"{fat:.3f}", delta_color=color)
