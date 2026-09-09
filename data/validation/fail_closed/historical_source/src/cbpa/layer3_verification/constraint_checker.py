"""Hard constraint checking against K."""

from __future__ import annotations

import logging
import operator

import numpy as np

from cbpa.models.contract import OutcomeContract
from cbpa.models.metrics import (
    FactoryMetrics,
    FeasibilityReport,
    SimulationMetrics,
    StochasticFeasibilityReport,
)

logger = logging.getLogger(__name__)

_OPS = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
    "==": operator.eq,
}

# Map constraint names to SimulationMetrics field names.
# Includes aliases so LLM-generated constraint names still resolve.
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

# Which Monte Carlo input dimensions correspond to which label
_MC_PARAM_LABELS = [
    "r1_speed_fraction",
    "r2_speed_fraction",
    "human_rate_multiplier",
    "r1_cycle_time",
    "r2_cycle_time",
]


class ConstraintChecker:
    """Checks simulation metrics against the contract's hard constraints K."""

    def __init__(self, cyber_risk_level: float = 1.0):
        self.cyber_risk_level = cyber_risk_level

    # ------------------------------------------------------------------
    # Deterministic (original) path — unchanged API
    # ------------------------------------------------------------------

    def check(
        self,
        contract: OutcomeContract,
        metrics: SimulationMetrics,
    ) -> FeasibilityReport:
        violations: list[str] = []
        margins: dict[str, float] = {}

        for constraint in contract.hard_constraints:
            op_fn = _OPS.get(constraint.operator, operator.le)

            if constraint.name == "CyberRiskLevel":
                actual = self.cyber_risk_level
            else:
                field = _METRIC_MAP.get(constraint.name)
                if field is None:
                    logger.warning(
                        "hard constraint '%s' does not map to a monitored "
                        "metric and is NOT checked; canonicalise it at Layer 1",
                        constraint.name,
                    )
                    continue
                actual = getattr(metrics, field, None)
                if actual is None:
                    continue

            limit = constraint.limit
            margin = limit - actual  # positive = within limit

            margins[constraint.name] = round(margin, 4)

            if not op_fn(actual, limit):
                violations.append(
                    f"{constraint.name}: {actual} violates {constraint.operator} {limit}"
                )

        return FeasibilityReport(
            is_feasible=len(violations) == 0,
            violations=violations,
            constraint_margins=margins,
        )

    # ------------------------------------------------------------------
    # Factory-level constraint checking
    # ------------------------------------------------------------------

    # Map factory contract constraint names to FactoryMetrics fields or
    # operator fatigue keys.  Prefix "FatigueIndex_" indicates per-operator.
    _FACTORY_METRIC_MAP = {
        "FactoryNoise": "factory_noise_db",
        "factory_noise": "factory_noise_db",
        "CellBalanceLoss": "cell_balance_loss_pct",
        "AGVTransferTime": None,  # checked via agv_utilization proxy
    }

    def check_factory(
        self,
        contract: OutcomeContract,
        metrics: FactoryMetrics,
    ) -> FeasibilityReport:
        """Check factory-level metrics against the contract's hard constraints.

        Handles per-operator fatigue constraints (``FatigueIndex_H1``,
        ``FatigueIndex_H2``, etc.) by looking up operator_fatigue dict,
        combined factory noise, and other factory-level fields.
        """
        violations: list[str] = []
        margins: dict[str, float] = {}

        for constraint in contract.hard_constraints:
            op_fn = _OPS.get(constraint.operator, operator.le)

            # Per-operator fatigue: FatigueIndex_H1, FatigueIndex_H2, etc.
            if constraint.name.startswith("FatigueIndex_"):
                op_id = constraint.name.split("_", 1)[1]
                actual = metrics.operator_fatigue.get(op_id)
                if actual is None:
                    continue
            elif constraint.name == "CyberRiskLevel":
                actual = self.cyber_risk_level
            elif constraint.name == "AGVTransferTime":
                # Use agv_utilization as proxy: if > 1.0 it implies
                # transfer time exceeds capacity.  Map to seconds estimate.
                actual = metrics.agv_utilization * 60.0  # rough proxy
            else:
                field = self._FACTORY_METRIC_MAP.get(constraint.name)
                if field is None:
                    continue
                actual = getattr(metrics, field, None)
                if actual is None:
                    continue

            limit = constraint.limit
            margin = limit - actual
            margins[constraint.name] = round(margin, 4)

            if not op_fn(actual, limit):
                violations.append(
                    f"{constraint.name}: {actual} violates {constraint.operator} {limit}"
                )

        return FeasibilityReport(
            is_feasible=len(violations) == 0,
            violations=violations,
            constraint_margins=margins,
        )

    def check_factory_stochastic(
        self,
        contract: OutcomeContract,
        metrics_samples: list[FactoryMetrics],
        seed: int = 42,
        confidence_threshold: float = 0.95,
    ) -> StochasticFeasibilityReport:
        """Stochastic feasibility check over factory-level Monte Carlo samples."""
        n = len(metrics_samples)
        if n == 0:
            return StochasticFeasibilityReport(
                p_feasible=0.0,
                constraint_violation_probabilities={},
                worst_case_margins={},
                sensitivity_ranking=[],
                n_samples=0,
                seed=seed,
                certified=False,
                confidence_threshold=confidence_threshold,
            )

        # Evaluate each sample
        sample_reports = [self.check_factory(contract, m) for m in metrics_samples]

        # p_feasible
        n_feasible = sum(1 for r in sample_reports if r.is_feasible)
        p_feasible = round(n_feasible / n, 4)

        # Per-constraint violation probability
        constraint_names = set()
        for r in sample_reports:
            constraint_names.update(r.constraint_margins.keys())

        violation_probs: dict[str, float] = {}
        worst_case_margins: dict[str, float] = {}

        for c_name in sorted(constraint_names):
            margins_list = [
                r.constraint_margins.get(c_name, 0.0) for r in sample_reports
            ]
            n_violations = sum(1 for m in margins_list if m < 0)
            violation_probs[c_name] = round(n_violations / n, 4)
            worst_case_margins[c_name] = round(
                float(np.percentile(margins_list, 5)), 4
            )

        return StochasticFeasibilityReport(
            p_feasible=p_feasible,
            constraint_violation_probabilities=violation_probs,
            worst_case_margins=worst_case_margins,
            sensitivity_ranking=[],
            n_samples=n,
            seed=seed,
            certified=p_feasible >= confidence_threshold,
            confidence_threshold=confidence_threshold,
        )

    # ------------------------------------------------------------------
    # Stochastic (Monte Carlo) path — new capability
    # ------------------------------------------------------------------

    def check_stochastic(
        self,
        contract: OutcomeContract,
        metrics_samples: list[SimulationMetrics],
        seed: int = 42,
        confidence_threshold: float = 0.95,
    ) -> StochasticFeasibilityReport:
        """Assess feasibility probabilistically over a Monte Carlo sample set.

        For each sample in *metrics_samples*, every hard constraint in
        *contract* is evaluated.  The method then computes:

        - **p_feasible**: fraction of samples where *all* constraints pass.
        - **constraint_violation_probabilities**: per-constraint P(violation).
        - **worst_case_margins**: 5th-percentile of (limit − actual) per
          constraint, giving a conservative safety margin.
        - **sensitivity_ranking**: identifies which input perturbation
          (stored as metadata on each sample, inferred from the spread of
          metric values) contributes most to driving each constraint toward
          violation.  Sensitivity is estimated as the Pearson correlation
          between the per-sample metric value and per-sample constraint
          margin, without requiring explicit parameter labels — instead we
          report the constraint dimension most correlated with its own
          margin spread, ranked by absolute correlation magnitude.

        Parameters
        ----------
        contract:
            The ``OutcomeContract`` whose ``hard_constraints`` are evaluated.
        metrics_samples:
            List of ``SimulationMetrics`` from
            ``CellEvaluator.evaluate_monte_carlo()``.
        seed:
            Passed through to the report for provenance; must match the seed
            used when generating *metrics_samples*.
        """
        n = len(metrics_samples)
        if n == 0:
            return StochasticFeasibilityReport(
                p_feasible=0.0,
                constraint_violation_probabilities={},
                worst_case_margins={},
                sensitivity_ranking=[],
                n_samples=0,
                seed=seed,
                certified=False,
                confidence_threshold=confidence_threshold,
            )

        # Collect the relevant constraints (skip unresolvable ones)
        active_constraints: list = []
        for constraint in contract.hard_constraints:
            if constraint.name == "CyberRiskLevel":
                active_constraints.append(("CyberRiskLevel", None, constraint))
            else:
                field = _METRIC_MAP.get(constraint.name)
                if field is not None:
                    active_constraints.append((constraint.name, field, constraint))

        # Per-sample per-constraint: True = passed, margin array
        # Shape: (n_constraints, n_samples)
        n_c = len(active_constraints)
        passed_matrix = np.ones((n_c, n), dtype=bool)
        margin_matrix = np.zeros((n_c, n), dtype=float)

        # Also collect the raw metric arrays for sensitivity analysis
        # We use the metric fields we can observe directly
        metric_arrays: dict[str, np.ndarray] = {
            "fatigue_index": np.array([s.fatigue_index for s in metrics_samples]),
            "noise_db": np.array([s.noise_db for s in metrics_samples]),
            "throughput_uph": np.array([s.throughput_uph for s in metrics_samples]),
        }

        for ci, (c_name, field, constraint) in enumerate(active_constraints):
            op_fn = _OPS.get(constraint.operator, operator.le)
            limit = constraint.limit

            for si, sample in enumerate(metrics_samples):
                if c_name == "CyberRiskLevel":
                    actual = self.cyber_risk_level
                else:
                    actual = getattr(sample, field)  # type: ignore[arg-type]

                margin_matrix[ci, si] = limit - actual
                passed_matrix[ci, si] = bool(op_fn(actual, limit))

        # p_feasible: all constraints must pass simultaneously
        all_pass = passed_matrix.all(axis=0)  # shape (n,)
        p_feasible = float(all_pass.sum()) / n

        # Per-constraint violation probability
        violation_probs: dict[str, float] = {}
        for ci, (c_name, _, _) in enumerate(active_constraints):
            n_violations = int((~passed_matrix[ci]).sum())
            violation_probs[c_name] = round(n_violations / n, 4)

        # Worst-case margins: 5th percentile of (limit - actual)
        worst_case_margins: dict[str, float] = {}
        for ci, (c_name, _, _) in enumerate(active_constraints):
            p5 = float(np.percentile(margin_matrix[ci], 5))
            worst_case_margins[c_name] = round(p5, 4)

        # Sensitivity ranking
        # Strategy: for each constraint, find which observable metric series
        # is most correlated with the constraint's margin series.
        # This reveals which physical quantity's variability drives risk.
        sensitivity_ranking: list[tuple[str, str, float]] = []

        for ci, (c_name, _, _) in enumerate(active_constraints):
            margins_i = margin_matrix[ci]
            if margins_i.std() < 1e-9:
                # Completely deterministic — no sensitivity to report
                continue

            best_param = ""
            best_corr = 0.0
            for param_label, metric_arr in metric_arrays.items():
                if metric_arr.std() < 1e-9:
                    continue
                corr = float(np.corrcoef(metric_arr, margins_i)[0, 1])
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_param = param_label

            if best_param:
                sensitivity_ranking.append(
                    (best_param, c_name, round(abs(best_corr), 4))
                )

        # Sort by descending sensitivity score
        sensitivity_ranking.sort(key=lambda t: t[2], reverse=True)

        p_feasible_rounded = round(p_feasible, 4)
        return StochasticFeasibilityReport(
            p_feasible=p_feasible_rounded,
            constraint_violation_probabilities=violation_probs,
            worst_case_margins=worst_case_margins,
            sensitivity_ranking=sensitivity_ranking,
            n_samples=n,
            seed=seed,
            certified=p_feasible_rounded >= confidence_threshold,
            confidence_threshold=confidence_threshold,
        )
