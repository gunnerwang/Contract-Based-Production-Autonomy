"""Schedule parameter display card."""

from __future__ import annotations

import streamlit as st

from cbpa.models.schedule import Schedule
from cbpa.ui.theme import SCHEDULE_COLORS, display_schedule


def schedule_card(schedule: Schedule, metrics=None, vr_score=None) -> None:
    """Render a compact card showing schedule parameters and optional KPIs."""
    from cbpa.ui.state import is_dev_mode

    dev = is_dev_mode()
    color = SCHEDULE_COLORS.get(schedule.name, "#757575")
    name_display = display_schedule(schedule.name, dev)

    st.markdown(
        f'<div style="border-left:4px solid {color};padding:8px 12px;'
        f'margin-bottom:8px;background:#fafafa;border-radius:4px;">'
        f'<strong style="color:{color};">{name_display}</strong>'
        f'{"  ⚠️ Aggressive" if schedule.is_aggressive else ""}'
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    if dev:
        with col1:
            st.markdown(f"**R1 Speed:** {schedule.r1_speed_fraction:.0%}")
            st.markdown(f"**R2 Speed:** {schedule.r2_speed_fraction:.0%}")
        with col2:
            st.markdown(f"**Human Rate:** {schedule.human_cycle_rate_multiplier:.2f}x")
            st.markdown(f"**Buffer:** {schedule.buffer_time_s:.1f}s")
    else:
        with col1:
            st.markdown(f"**Robot 1 Speed:** {schedule.r1_speed_fraction:.0%}")
            st.markdown(f"**Robot 2 Speed:** {schedule.r2_speed_fraction:.0%}")
        with col2:
            st.markdown(f"**Worker Pace:** {schedule.human_cycle_rate_multiplier:.2f}x")
            st.markdown(f"**Buffer:** {schedule.buffer_time_s:.1f}s")

    target_label = "Demand Target" if dev else "Target Production"
    st.markdown(f"**{target_label}:** {schedule.demand_target_uph:.1f} u/h")

    if metrics is not None:
        st.markdown("---")
        mc1, mc2, mc3 = st.columns(3)
        if dev:
            with mc1:
                st.metric("Throughput", f"{metrics.throughput_uph:.1f} u/h")
            with mc2:
                st.metric("Fatigue", f"{metrics.fatigue_index:.2f}")
            with mc3:
                st.metric("Noise", f"{metrics.noise_db:.1f} dB")
        else:
            with mc1:
                st.metric("Production Rate", f"{metrics.throughput_uph:.1f} u/h")
            with mc2:
                st.metric("Worker Fatigue", f"{metrics.fatigue_index:.2f}")
            with mc3:
                st.metric("Noise Level", f"{metrics.noise_db:.1f} dB")

    if vr_score is not None and dev:
        st.metric("V/R Score", f"{vr_score.vr_score:.3f}")
