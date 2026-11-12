"""Explicit conversion to the executable plant ontology; no unit inference by magnitude.

Empty units inherit the named field's canonical unit. Unknown units and rate vs
quantity mismatches refuse compilation. SI energy prefixes remain case-sensitive.
"""
from decimal import Decimal
import math


class UnitCompilationError(ValueError):
    """A quantity cannot be safely compiled for the executable ontology."""


_NAMES = {
    "FatigueIndex": "fatigue", "fatigue_index": "fatigue", "fatigue": "fatigue", "operator_fatigue": "fatigue",
    "Noise": "noise", "noise": "noise", "noise_db": "noise", "noise_level": "noise", "FactoryNoise": "noise", "factory_noise": "noise",
    "Energy": "energy", "energy": "energy", "energy_kwh": "energy",
    "DefectRate": "defect", "defect_rate": "defect", "defect": "defect",
    "FactoryThroughput": "throughput", "Throughput": "throughput", "throughput": "throughput", "throughput_uph": "throughput",
    "DeadlineGap": "percent", "deadline_gap": "percent", "deadline_gap_pct": "percent",
    "CellBalanceLoss": "percent", "CellBalance": "percent",
    "CyberRiskLevel": "cyber", "cyber_risk": "cyber",
    "AGVTransferTime": "seconds", "ChangeoverTime": "minutes",
}
_CANONICAL = {"fatigue": "dimensionless", "noise": "dB", "energy": "kWh", "defect": "fraction",
              "throughput": "u/h", "percent": "%", "cyber": "level", "seconds": "s", "minutes": "min"}
_ALIASES = {
    "fatigue": {"dimensionless": "1", "index": "1", "fraction": "1", "%": "0.01", "percent": "0.01"},
    "defect": {"fraction": "1", "dimensionless": "1", "%": "0.01", "percent": "0.01"},
    "noise": {"dB": "1", "db": "1"},
    "energy": {"kWh": "1", "kwh": "1", "kWh per shift": "1", "kWh/shift": "1", "Wh": "0.001", "Wh per shift": "0.001", "Wh/shift": "0.001", "MWh": "1000", "MWh per shift": "1000", "MWh/shift": "1000"},
    "throughput": {"u/h": "1", "units/h": "1", "units per hour": "1", "uph": "1", "u/min": "60", "units/min": "60"},
    "percent": {"%": "1", "percent": "1", "fraction": "100", "dimensionless": "100"},
    "cyber": {"level": "1", "dimensionless": "1"},
    "seconds": {"s": "1", "sec": "1", "seconds": "1", "ms": "0.001", "min": "60"},
    "minutes": {"min": "1", "minutes": "1", "s": "0.01666666666666666666666666667", "sec": "0.01666666666666666666666666667"},
}


def unit_spec(name, unit):
    kind = "fatigue" if name.startswith("FatigueIndex_") else _NAMES.get(name)
    if kind is None:
        raise UnitCompilationError(f"{name}: unsupported quantity")
    cleaned = " ".join(unit.strip().split())
    if not cleaned:
        return _CANONICAL[kind], Decimal(1)
    factor = _ALIASES[kind].get(cleaned)
    if factor is None:
        raise UnitCompilationError(f"{name}: unsupported unit {unit!r}")
    return _CANONICAL[kind], Decimal(factor)


def canonical_unit_error(name, unit):
    """Raw evaluator inputs may use only representations with unit scale one."""
    try:
        _, factor = unit_spec(name, unit)
        if factor != 1:
            return f"{name}: unit {unit!r} requires compilation before evaluation"
    except UnitCompilationError as exc:
        return str(exc)
    return None


def normalize_contract_units(contract):
    """Return an atomic copy and conversion audit, preserving all comparisons.

    Positive linear conversion changes representation, never the physical limit.
    No priority, context, ambiguity, approval or constraint membership is changed.
    """
    result = contract.model_copy(deep=True)
    changes = []
    for group, attr in [("hard_constraints", "limit"), ("kpi_targets", "threshold")]:
        for item in getattr(result, group):
            # Unknown fields remain visible for schema/constraint rejection;
            # unit compilation must not drop an unsupported obligation.
            if item.name not in _NAMES and not item.name.startswith("FatigueIndex_"):
                continue
            canonical, factor = unit_spec(item.name, item.unit)
            value = getattr(item, attr)
            converted = value
            if value is not None:
                if not math.isfinite(value) or value < 0:
                    raise UnitCompilationError(f"{item.name}: invalid quantity {value}")
                converted = float(Decimal(str(value)) * factor)
                if not math.isfinite(converted):
                    raise UnitCompilationError(f"{item.name}: conversion overflow")
            if item.unit != canonical or value != converted:
                changes.append({"group": group, "name": item.name, "old_value": value,
                                "old_unit": item.unit, "value": converted, "unit": canonical,
                                "factor": str(factor)})
            setattr(item, attr, converted)
            item.unit = canonical
    return result, changes
