"""Color palette and styling constants for the CBPA Streamlit UI."""

from __future__ import annotations

# Schedule colors (matches publication figures)
S1_COLOR = "#2196F3"  # Blue — initial baseline
S2_COLOR = "#F44336"  # Red — aggressive / rejected
S3_COLOR = "#4CAF50"  # Green — balanced / final

SCHEDULE_COLORS = {"S1": S1_COLOR, "S2": S2_COLOR, "S3": S3_COLOR}

# Phase colors
PHASE_COLORS = {
    1: "#2196F3",  # Blue — initial deployment
    2: "#FF9800",  # Orange — demand surge
    3: "#F44336",  # Red — autonomous replan (rejected)
    4: "#9C27B0",  # Purple — escalation
    5: "#4CAF50",  # Green — adaptation + stabilization
}

# KPI gauge thresholds
GAUGE_THRESHOLDS = {
    "throughput_uph": {"good": 50, "warn": 44, "unit": "u/h"},
    "defect_rate": {"good": 0.008, "warn": 0.01, "unit": "fraction"},
    "noise_db": {"good": 78, "warn": 80, "unit": "dB", "higher_is_worse": True},
    "fatigue_index": {"good": 0.35, "warn": 0.40, "unit": "index", "higher_is_worse": True},
    "energy_kwh": {"good": 400, "warn": 450, "unit": "kWh", "higher_is_worse": True},
    "deadline_gap_pct": {"good": 5, "warn": 15, "unit": "%", "higher_is_worse": True},
}

# Constraint indicator colors
CONSTRAINT_OK = "#4CAF50"      # Green
CONSTRAINT_WARNING = "#FF9800"  # Orange/yellow
CONSTRAINT_VIOLATION = "#F44336"  # Red

# Alert severity
SEVERITY_COLORS = {
    "violation": "#F44336",
    "warning": "#FF9800",
    "info": "#2196F3",
}

# Layer labels
LAYER_LABELS = {
    "L1": "Outcome Specification",
    "L2": "Planning & Scoring",
    "L3": "Verification",
    "L4": "Execution & Monitor",
    "L5": "Adaptation & Learning",
    "Meta": "Governance / Escalation",
}

# Phase labels — aligned with CBPA lifecycle rounds
PHASE_LABELS = {
    1: "Contract & Deploy (L1→L2→L3→L4)",
    2: "Monitor & Detect (L4)",
    3: "Replan Attempt (L2→L3)",
    4: "Escalation (Meta)",
    5: "Adapt & Stabilize (L1→L5→L2→L3→L4→L5)",
    6: "Macro Adapt (L5→L1→L2→L3→L4)",
}

# ── Operator-mode display name mappings ──────────────────────────────

SCHEDULE_DISPLAY_NAMES = {
    "S1": "Plan A (Baseline)",
    "S2": "Plan B (Aggressive)",
    "S3": "Plan C (Balanced)",
}

LAYER_DISPLAY_NAMES = {
    "L1": "Goal Setting",
    "L2": "Planning & Scoring",
    "L3": "Safety Check",
    "L4": "Run & Monitor",
    "L5": "Learn & Improve",
    "Meta": "Manager Review",
}

KPI_DISPLAY_NAMES = {
    "throughput_uph": "Production Rate",
    "defect_rate": "Defect Rate",
    "noise_db": "Noise Level",
    "fatigue_index": "Worker Fatigue",
    "energy_kwh": "Energy Usage",
    "deadline_gap_pct": "Deadline Gap",
}

PHASE_DISPLAY_NAMES = {
    1: "Contract & Deploy",
    2: "Detect Disturbance",
    3: "Replan Attempt",
    4: "Manager Decision",
    5: "Adapt & Stabilize",
    6: "Macro Adaptation",
}

OPTION_DISPLAY_NAMES = {
    "Option A": "Extend Shift Time",
    "Option B": "Allow Higher Fatigue",
    "Option C": "Keep Safety Limits",
}

MODE_LABELS = {
    "legislator": "Define Contract",
    "partner": "Run & Monitor",
    "auditor": "Shift Review",
}
MODE_DESCRIPTIONS = {
    "legislator": "Set goals, safety limits, and shift context",
    "partner": "Monitor execution, provide feedback, handle escalations",
    "auditor": "Review shift outcomes, adjust future contracts",
}


def display_schedule(key: str, dev_mode: bool) -> str:
    """Return display name for a schedule key."""
    if dev_mode:
        return key
    return SCHEDULE_DISPLAY_NAMES.get(key, key)


def display_kpi(key: str, dev_mode: bool) -> str:
    """Return display name for a KPI key."""
    if dev_mode:
        return key.replace("_", " ").title()
    return KPI_DISPLAY_NAMES.get(key, key.replace("_", " ").title())


def display_layer(key: str, dev_mode: bool) -> str:
    """Return display name for a layer key."""
    if dev_mode:
        return key
    return LAYER_DISPLAY_NAMES.get(key, key)


def display_phase(phase: int, dev_mode: bool) -> str:
    """Return display name for a phase number."""
    if dev_mode:
        return PHASE_LABELS.get(phase, f"Phase {phase}")
    return PHASE_DISPLAY_NAMES.get(phase, f"Step {phase}")
