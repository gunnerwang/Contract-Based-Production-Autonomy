"""Configuration service: wraps ScenarioConfig for UI consumption."""

from __future__ import annotations

from dataclasses import asdict, fields

from cbpa.config.scenario import CellConfig, ScheduleParams, ScenarioConfig

# ── Operator-mode presets ────────────────────────────────────────────
PRESETS: dict[str, dict] = {
    "Standard Shift": {
        "summary": "Default 8-hour shift with standard robot speeds and worker pace.",
        "metrics_preview": {"Shift": "8 h", "Robot Speed": "65 %", "Worker Pace": "Normal"},
        "constraint_notes": {
            "fatigue_target": 0.40,
            "noise_target": 80.0,
            "note": "Standard contract limits apply.",
        },
        "config": {},  # all defaults
    },
    "High Demand": {
        "summary": "Faster robots and extended shift to meet a 20 % demand surge.",
        "metrics_preview": {"Shift": "10 h", "Robot Speed": "80 %", "Worker Pace": "Slightly faster"},
        "constraint_notes": {
            "fatigue_target": 0.40,
            "noise_target": 80.0,
            "note": "Same safety limits, but higher risk of approaching them.",
        },
        "config": {
            "cell.shift_hours": 10.0,
            "cell.demand_base_uph": 60.0,
            "schedules.s1_r1_speed": 0.80,
            "schedules.s1_r2_speed": 0.75,
            "schedules.s1_human_rate": 1.05,
            "schedules.s1_buffer_s": 2.0,
            "demand_spike_pct": 25.0,
        },
    },
    "Ergonomic Priority": {
        "summary": "Slower robots, longer buffers — minimizes worker fatigue and noise.",
        "metrics_preview": {"Shift": "8 h", "Robot Speed": "55 %", "Worker Pace": "Relaxed"},
        "constraint_notes": {
            "fatigue_target": 0.30,
            "noise_target": 75.0,
            "note": "Tighter operating targets (below contract limits) for extra comfort margin.",
        },
        "config": {
            "schedules.s1_r1_speed": 0.55,
            "schedules.s1_r2_speed": 0.50,
            "schedules.s1_human_rate": 0.95,
            "schedules.s1_buffer_s": 5.0,
            "demand_spike_pct": 10.0,
        },
    },
}


class ConfigService:
    """Helpers for creating, inspecting, and validating ScenarioConfig."""

    @staticmethod
    def create_default() -> ScenarioConfig:
        return ScenarioConfig()

    @staticmethod
    def from_dict(d: dict) -> ScenarioConfig:
        """Build a ScenarioConfig from a flat dict (e.g. from UI sliders).

        Keys use dot-separated paths: ``cell.shift_hours``,
        ``schedules.s1_r1_speed``, ``demand_spike_pct``, etc.
        """
        cell_kwargs: dict = {}
        sched_kwargs: dict = {}
        top_kwargs: dict = {}

        cell_fields = {f.name for f in fields(CellConfig)}
        sched_fields = {f.name for f in fields(ScheduleParams)}

        for key, val in d.items():
            if key.startswith("cell."):
                fname = key.split(".", 1)[1]
                if fname in cell_fields:
                    cell_kwargs[fname] = val
            elif key.startswith("schedules."):
                fname = key.split(".", 1)[1]
                if fname in sched_fields:
                    sched_kwargs[fname] = val
            elif key == "vr_value_weights":
                top_kwargs["vr_value_weights"] = list(val)
            elif key == "vr_resource_weights":
                top_kwargs["vr_resource_weights"] = list(val)
            elif key in ("demand_spike_pct", "spike_time_hours", "table3_tolerance_pct"):
                top_kwargs[key] = val

        return ScenarioConfig(
            cell=CellConfig(**cell_kwargs) if cell_kwargs else CellConfig(),
            schedules=ScheduleParams(**sched_kwargs) if sched_kwargs else ScheduleParams(),
            **top_kwargs,
        )

    @staticmethod
    def to_flat_dict(config: ScenarioConfig) -> dict:
        """Flatten config into a dot-separated key dict for UI display."""
        flat: dict = {}
        for f in fields(config.cell):
            flat[f"cell.{f.name}"] = getattr(config.cell, f.name)
        for f in fields(config.schedules):
            flat[f"schedules.{f.name}"] = getattr(config.schedules, f.name)
        flat["demand_spike_pct"] = config.demand_spike_pct
        flat["spike_time_hours"] = config.spike_time_hours
        flat["vr_value_weights"] = config.vr_value_weights
        flat["vr_resource_weights"] = config.vr_resource_weights
        return flat

    @staticmethod
    def from_preset(name: str) -> ScenarioConfig:
        """Build a ScenarioConfig from a named preset."""
        if name not in PRESETS:
            raise ValueError(f"Unknown preset: {name!r}. Choose from {list(PRESETS.keys())}")
        overrides = PRESETS[name]["config"]
        if not overrides:
            return ScenarioConfig()
        return ConfigService.from_dict(overrides)

    @staticmethod
    def preset_names() -> list[str]:
        """Return available preset names."""
        return list(PRESETS.keys())

    @staticmethod
    def validate_weights(weights: list[float], label: str = "weights") -> list[str]:
        """Return list of validation errors (empty = OK)."""
        errors: list[str] = []
        if not weights:
            errors.append(f"{label}: must not be empty")
            return errors
        if any(w < 0 for w in weights):
            errors.append(f"{label}: all weights must be >= 0")
        total = sum(weights)
        if abs(total - 1.0) > 0.01:
            errors.append(f"{label}: weights sum to {total:.3f}, expected 1.0")
        return errors
