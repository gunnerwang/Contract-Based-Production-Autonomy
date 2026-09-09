"""KPI gauge/metric cards with color-coded thresholds."""

from __future__ import annotations

import streamlit as st

from cbpa.ui.theme import CONSTRAINT_OK, CONSTRAINT_VIOLATION, CONSTRAINT_WARNING


def kpi_metric_card(
    label: str,
    value: float,
    unit: str = "",
    good_threshold: float | None = None,
    warn_threshold: float | None = None,
    higher_is_worse: bool = False,
    delta: float | None = None,
) -> None:
    """Render a single KPI metric with color-coded status."""
    color = CONSTRAINT_OK
    if good_threshold is not None and warn_threshold is not None:
        if higher_is_worse:
            if value >= warn_threshold:
                color = CONSTRAINT_VIOLATION
            elif value >= good_threshold:
                color = CONSTRAINT_WARNING
        else:
            if value <= warn_threshold:
                color = CONSTRAINT_VIOLATION
            elif value <= good_threshold:
                color = CONSTRAINT_WARNING

    display_val = f"{value:.3f}" if isinstance(value, float) and value < 1 else f"{value:.1f}"
    delta_str = None
    delta_color = "normal"
    if delta is not None:
        delta_str = f"{delta:+.2f}"
        if higher_is_worse:
            delta_color = "inverse"

    st.metric(
        label=f"{label} ({unit})" if unit else label,
        value=display_val,
        delta=delta_str,
        delta_color=delta_color,
    )
    st.markdown(
        f'<div style="height:4px;background:{color};border-radius:2px;"></div>',
        unsafe_allow_html=True,
    )


def kpi_gauge_row(metrics_dict: dict) -> None:
    """Render a row of KPI gauges from a metrics dictionary."""
    from cbpa.ui.state import is_dev_mode
    from cbpa.ui.theme import GAUGE_THRESHOLDS, display_kpi

    dev = is_dev_mode()
    cols = st.columns(len(metrics_dict))
    for col, (key, value) in zip(cols, metrics_dict.items()):
        thresholds = GAUGE_THRESHOLDS.get(key, {})
        with col:
            kpi_metric_card(
                label=display_kpi(key, dev),
                value=value,
                unit=thresholds.get("unit", ""),
                good_threshold=thresholds.get("good"),
                warn_threshold=thresholds.get("warn"),
                higher_is_worse=thresholds.get("higher_is_worse", False),
            )
