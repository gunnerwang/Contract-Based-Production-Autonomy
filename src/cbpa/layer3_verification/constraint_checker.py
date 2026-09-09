"""Hard constraint checking: uncheckable obligations fail closed."""
from __future__ import annotations
import math
import operator
import numpy as np
from cbpa.models.contract import OutcomeContract
from cbpa.models.metrics import FactoryMetrics, FeasibilityReport, SimulationMetrics, StochasticFeasibilityReport

_OPS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt, "==": operator.eq}

_METRIC_MAP = {
    "FatigueIndex": "fatigue_index",
    "fatigue_index": "fatigue_index",
    "fatigue": "fatigue_index",
    "operator_fatigue": "fatigue_index",
    "Noise": "noise_db",
    "noise": "noise_db",
    "noise_db": "noise_db",
    "noise_level": "noise_db",
    "Energy": "energy_kwh",
    "energy": "energy_kwh",
    "energy_kwh": "energy_kwh",
    "DefectRate": "defect_rate",
    "defect_rate": "defect_rate",
    "defect": "defect_rate",
    "Throughput": "throughput_uph",
    "throughput": "throughput_uph",
    "throughput_uph": "throughput_uph",
    "DeadlineGap": "deadline_gap_pct",
    "deadline_gap": "deadline_gap_pct",
    "deadline_gap_pct": "deadline_gap_pct",
    "CyberRiskLevel": None,  # checked separately
    "cyber_risk": None,
}

class ConstraintChecker:
    """Evaluate every encoded hard constraint, including evaluation failures."""

    _FACTORY_METRIC_MAP = {
        "FactoryNoise": "factory_noise_db",
        "factory_noise": "factory_noise_db",
        "CellBalanceLoss": "cell_balance_loss_pct",
        "AGVTransferTime": "agv_utilization",
    }

    def __init__(self, cyber_risk_level: float = 1.0):
        self.cyber_risk_level = cyber_risk_level

    @classmethod
    def contract_errors(cls, contract: OutcomeContract, *, factory=False) -> list[str]:
        errors = []
        if not contract.hard_constraints:
            errors.append("Contract has no hard constraints")
        seen = set()
        for c in contract.hard_constraints:
            if c.name in seen:
                errors.append(f"{c.name}: duplicate constraint name")
            seen.add(c.name)
            supported = (
                c.name in cls._FACTORY_METRIC_MAP
                or (c.name.startswith("FatigueIndex_") and len(c.name) > len("FatigueIndex_"))
                or c.name in {"CyberRiskLevel", "cyber_risk"}
            ) if factory else c.name in _METRIC_MAP
            if not supported:
                errors.append(f"{c.name}: unsupported hard constraint")
            if c.operator not in _OPS:
                errors.append(f"{c.name}: unsupported operator {c.operator!r}")
            if not cls._finite(c.limit):
                errors.append(f"{c.name}: non-finite or non-numeric limit")
        return errors

    @staticmethod
    def constraint_spec(contract):
        return sorted((c.name, c.operator, c.limit) for c in contract.hard_constraints)

    @staticmethod
    def _finite(value):
        return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and math.isfinite(value)

    def _actual(self, name, metrics, factory):
        if name in {"CyberRiskLevel", "cyber_risk"}:
            return self.cyber_risk_level
        if factory and name.startswith("FatigueIndex_"):
            fatigue = getattr(metrics, "operator_fatigue", None)
            return fatigue.get(name.split("_", 1)[1]) if isinstance(fatigue, dict) else None
        field = (self._FACTORY_METRIC_MAP if factory else _METRIC_MAP).get(name)
        if field is None:
            return None
        # Pydantic defaults are not measurements. The evaluator must supply the field.
        if hasattr(metrics, "model_fields_set") and field not in metrics.model_fields_set:
            return None
        value = getattr(metrics, field, None)
        if factory and name == "AGVTransferTime" and self._finite(value):
            return value * 60.0
        return value

    def _check(self, contract, metrics, *, factory):
        errors = self.contract_errors(contract, factory=factory)
        violations = []
        margins = {}
        outcomes = {}
        for c in contract.hard_constraints:
            outcomes[c.name] = False
            if any(e.startswith(c.name + ":") for e in errors):
                continue
            actual = self._actual(c.name, metrics, factory)
            if not self._finite(actual):
                errors.append(f"{c.name}: missing, non-finite, or non-numeric measurement")
                continue
            passed = bool(_OPS[c.operator](actual, c.limit))
            outcomes[c.name] = passed
            # Positive margins are feasible for upper and lower bounds. Equality
            # and strict-bound violations use outcomes, not the margin sign.
            margin = actual - c.limit if c.operator in {">", ">="} else c.limit - actual
            if c.operator == "==":
                margin = -abs(actual - c.limit)
            margins[c.name] = round(margin, 4)
            if not passed:
                violations.append(f"{c.name}: {actual} violates {c.operator} {c.limit}")
        return FeasibilityReport(
            is_feasible=not errors and all(outcomes.values()),
            verification_scope="factory" if factory else "cell",
            checked_constraints=self.constraint_spec(contract),
            violations=violations + errors, constraint_margins=margins,
            constraint_results=outcomes, evaluation_errors=errors,
        )

    def check(self, contract: OutcomeContract, metrics: SimulationMetrics) -> FeasibilityReport:
        return self._check(contract, metrics, factory=False)

    def check_factory(self, contract: OutcomeContract, metrics: FactoryMetrics) -> FeasibilityReport:
        return self._check(contract, metrics, factory=True)

    def check_stochastic(self, contract, metrics_samples, seed=42, confidence_threshold=0.95):
        return self._stochastic(contract, metrics_samples, seed, confidence_threshold, factory=False)

    def check_factory_stochastic(self, contract, metrics_samples, seed=42, confidence_threshold=0.95):
        return self._stochastic(contract, metrics_samples, seed, confidence_threshold, factory=True)

    def _stochastic(self, contract, samples, seed, threshold, *, factory):
        reports = [self._check(contract, m, factory=factory) for m in samples]
        n = len(reports)
        errors = self.contract_errors(contract, factory=factory)
        if not n:
            errors.append("No Monte Carlo samples")
        for i, report in enumerate(reports):
            errors.extend(f"sample {i}: {error}" for error in report.evaluation_errors)
        names = sorted({c.name for c in contract.hard_constraints})
        probabilities = {}
        margins = {}
        for name in names:
            probabilities[name] = round(sum(not r.constraint_results.get(name, False) for r in reports) / n, 4) if n else 1.0
            # No fabricated zero margin when a measurement is missing.
            if n and all(name in r.constraint_margins for r in reports):
                margins[name] = round(float(np.percentile([r.constraint_margins[name] for r in reports], 5)), 4)
        p = round(sum(r.is_feasible for r in reports) / n, 4) if n else 0.0
        sensitivity = []
        if not factory and not errors:
            metric_arrays = {field: np.array([getattr(m, field, np.nan) for m in samples], dtype=float)
                             for field in ["fatigue_index", "noise_db", "throughput_uph"]}
            for name in names:
                values = np.array([r.constraint_margins[name] for r in reports])
                if values.std() < 1e-9:
                    continue
                correlations = [(field, abs(float(np.corrcoef(values, values2)[0, 1])))
                                for field, values2 in metric_arrays.items()
                                if np.isfinite(values2).all() and values2.std() >= 1e-9]
                if correlations:
                    field, score = max(correlations, key=lambda item: item[1])
                    if score > 0:
                        sensitivity.append((field, name, round(score, 4)))
            sensitivity.sort(key=lambda item: item[2], reverse=True)
        return StochasticFeasibilityReport(
            p_feasible=p, constraint_violation_probabilities=probabilities,
            verification_scope="factory" if factory else "cell",
            checked_constraints=self.constraint_spec(contract),
            worst_case_margins=margins, sensitivity_ranking=sensitivity,
            n_samples=n, seed=seed, certified=not errors and p >= threshold,
            confidence_threshold=threshold, evaluation_errors=errors,
        )
