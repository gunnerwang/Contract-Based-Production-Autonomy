"""Factory-level evaluator wrapping per-cell evaluators with cross-cell physics.

Computes combined noise, per-operator fatigue, AGV bottleneck, cell balance,
and aggregate factory metrics.
"""

from __future__ import annotations

import logging

from cbpa.config.defaults import VARIANT_CYCLE_FACTORS
from cbpa.config.scenario import CellConfig, FactoryConfig, OperatorProfile
from cbpa.models.metrics import FactoryMetrics, SimulationMetrics
from cbpa.models.schedule import CellSchedule, FactorySchedule
from cbpa.physics.agv_model import AGVModel
from cbpa.physics.cell_evaluator import CellEvaluator
from cbpa.physics.fatigue_model import FatigueModel
from cbpa.physics.noise_model import NoiseModel

logger = logging.getLogger(__name__)


class FactoryEvaluator:
    """Evaluates a FactorySchedule across multiple cells."""

    def __init__(
        self,
        factory: FactoryConfig | None = None,
        mode: str = "analytical",
    ):
        self.factory = factory or FactoryConfig()
        self.mode = mode

        # Per-cell evaluators
        self.cell_evaluators: dict[str, CellEvaluator] = {}
        for cell_id, cell_cfg in self.factory.cells.items():
            self.cell_evaluators[cell_id] = CellEvaluator(
                cell=cell_cfg, mode=mode
            )

        self.agv_model = AGVModel(
            base_transfer_s=self.factory.agv_transfer_time_s
        )
        self.fatigue_model = FatigueModel()

    def _require_complete_schedule(self, schedule: FactorySchedule) -> None:
        expected = set(self.factory.cells)
        supplied = set(schedule.cell_schedules)
        if not expected or supplied != expected:
            raise ValueError(f"Incomplete factory schedule: expected cells {sorted(expected)}, got {sorted(supplied)}")
        missing = sorted(cid for cid in supplied if self.cell_evaluators.get(cid) is None)
        if missing:
            raise ValueError(f"Missing factory cell evaluators: {missing}")
        known_operators = {op.id for op in self.factory.operators}
        for operator_id, cell_id in schedule.operator_assignments.items():
            if operator_id not in known_operators or cell_id not in supplied:
                raise ValueError(f"Unresolvable operator assignment: {operator_id} -> {cell_id}")

    def evaluate(self, factory_schedule: FactorySchedule) -> FactoryMetrics:
        """Evaluate a factory schedule across all cells.

        Returns aggregated FactoryMetrics with per-cell breakdowns.
        """
        self._require_complete_schedule(factory_schedule)
        cell_metrics: dict[str, SimulationMetrics] = {}

        # Evaluate each cell independently
        for cell_id, cell_sched in factory_schedule.cell_schedules.items():
            evaluator = self.cell_evaluators.get(cell_id)
            cell_metrics[cell_id] = evaluator.evaluate(cell_sched)

        if not cell_metrics:
            return FactoryMetrics()

        # Combined factory noise (logarithmic sum)
        cell_noises = [m.noise_db for m in cell_metrics.values()]
        factory_noise = NoiseModel.compute_factory(*cell_noises)

        # Per-cell throughputs
        throughputs = {
            cid: m.throughput_uph for cid, m in cell_metrics.items()
        }
        total_throughput = sum(throughputs.values())

        # Cell balance loss: (max - min) / max * 100
        if throughputs:
            max_t = max(throughputs.values())
            min_t = min(throughputs.values())
            balance_loss = (
                (max_t - min_t) / max_t * 100.0 if max_t > 0 else 0.0
            )
        else:
            balance_loss = 0.0

        # AGV utilization (if two cells exist)
        tp_values = list(throughputs.values())
        if len(tp_values) >= 2:
            agv_util = self.agv_model.utilization(tp_values[0], tp_values[1])
        else:
            agv_util = 0.0

        # Per-operator fatigue based on assignments
        operator_fatigue: dict[str, float] = {}
        for op in self.factory.operators:
            assigned_cell = factory_schedule.operator_assignments.get(op.id)
            if assigned_cell and assigned_cell in factory_schedule.cell_schedules:
                cs = factory_schedule.cell_schedules[assigned_cell]
                fatigue = self.fatigue_model.compute(
                    cs.r1_speed_fraction, cs.r2_speed_fraction,
                    cs.human_cycle_rate_multiplier,
                )
                operator_fatigue[op.id] = fatigue
            else:
                operator_fatigue[op.id] = 0.0

        # Total energy
        factory_energy = sum(m.energy_kwh for m in cell_metrics.values())

        # Weighted average defect rate (by throughput)
        if total_throughput > 0:
            factory_defect = sum(
                m.defect_rate * m.throughput_uph
                for m in cell_metrics.values()
            ) / total_throughput
        else:
            factory_defect = 0.0

        return FactoryMetrics(
            cell_metrics=cell_metrics,
            total_throughput_uph=round(total_throughput, 1),
            cell_balance_loss_pct=round(balance_loss, 1),
            factory_noise_db=factory_noise,
            agv_utilization=agv_util,
            operator_fatigue=operator_fatigue,
            factory_energy_kwh=round(factory_energy, 1),
            factory_defect_rate=round(factory_defect, 4),
        )

    def evaluate_monte_carlo(
        self,
        factory_schedule: FactorySchedule,
        n_samples: int = 200,
        seed: int = 42,
    ) -> list[FactoryMetrics]:
        """Monte Carlo evaluation: perturb each cell independently, combine.

        For each sample, every cell is evaluated with its per-cell MC
        perturbation, then cross-cell metrics (noise, fatigue, AGV) are
        recomputed from the perturbed per-cell results.
        """
        import numpy as np

        self._require_complete_schedule(factory_schedule)
        rng = np.random.default_rng(seed)
        samples: list[FactoryMetrics] = []

        for i in range(n_samples):
            cell_metrics: dict[str, SimulationMetrics] = {}

            # Stable across processes and dictionary insertion order.
            for cell_index, cell_id in enumerate(sorted(factory_schedule.cell_schedules)):
                cell_sched = factory_schedule.cell_schedules[cell_id]
                evaluator = self.cell_evaluators.get(cell_id)
                sub_seed = seed + i * 100 + cell_index
                mc_results = evaluator.evaluate_monte_carlo(
                    cell_sched, n_samples=1, seed=sub_seed
                )
                cell_metrics[cell_id] = mc_results[0]

            if not cell_metrics:
                continue

            # Recompute cross-cell metrics from perturbed per-cell results
            cell_noises = [m.noise_db for m in cell_metrics.values()]
            factory_noise = NoiseModel.compute_factory(*cell_noises)

            throughputs = {cid: m.throughput_uph for cid, m in cell_metrics.items()}
            total_throughput = sum(throughputs.values())

            max_t = max(throughputs.values()) if throughputs else 0
            min_t = min(throughputs.values()) if throughputs else 0
            balance_loss = (max_t - min_t) / max_t * 100.0 if max_t > 0 else 0.0

            tp_values = list(throughputs.values())
            agv_util = (
                self.agv_model.utilization(tp_values[0], tp_values[1])
                if len(tp_values) >= 2 else 0.0
            )

            operator_fatigue: dict[str, float] = {}
            for op in self.factory.operators:
                assigned_cell = factory_schedule.operator_assignments.get(op.id)
                if assigned_cell and assigned_cell in factory_schedule.cell_schedules:
                    cs = factory_schedule.cell_schedules[assigned_cell]
                    fatigue = self.fatigue_model.compute(
                        cs.r1_speed_fraction, cs.r2_speed_fraction,
                        cs.human_cycle_rate_multiplier,
                    )
                    # Add small MC perturbation to fatigue
                    fatigue += rng.normal(0.0, fatigue * 0.05 / 3)
                    operator_fatigue[op.id] = max(0.0, fatigue)
                else:
                    operator_fatigue[op.id] = 0.0

            factory_energy = sum(m.energy_kwh for m in cell_metrics.values())
            factory_defect = (
                sum(m.defect_rate * m.throughput_uph for m in cell_metrics.values())
                / total_throughput if total_throughput > 0 else 0.0
            )

            samples.append(FactoryMetrics(
                cell_metrics=cell_metrics,
                total_throughput_uph=round(total_throughput, 1),
                cell_balance_loss_pct=round(balance_loss, 1),
                factory_noise_db=factory_noise,
                agv_utilization=agv_util,
                operator_fatigue=operator_fatigue,
                factory_energy_kwh=round(factory_energy, 1),
                factory_defect_rate=round(factory_defect, 4),
            ))

        return samples

    def evaluate_with_variant_weights(
        self,
        factory_schedule: FactorySchedule,
        variant_mix: dict[str, float] | None = None,
    ) -> FactoryMetrics:
        """Evaluate with variant-dependent cycle time adjustments.

        Args:
            factory_schedule: The factory schedule to evaluate.
            variant_mix: Fraction of demand per variant (e.g. {"V_A": 0.5, "V_B": 0.3, "V_C": 0.2}).
                         Defaults to equal split across assigned variants.
        """
        if variant_mix is None:
            # Equal split across all factory variants
            n = len(self.factory.product_variants)
            variant_mix = {v: 1.0 / n for v in self.factory.product_variants}

        # For each cell, compute a weighted cycle-time multiplier based on
        # which variants it handles and the variant mix.
        adjusted_schedules: dict[str, CellSchedule] = {}
        for cell_id, cell_sched in factory_schedule.cell_schedules.items():
            assigned_variants = cell_sched.assigned_variants
            if not assigned_variants:
                adjusted_schedules[cell_id] = cell_sched
                continue

            # Weighted average cycle-time factor across assigned variants
            total_weight = sum(
                variant_mix.get(v, 0.0) for v in assigned_variants
            )
            if total_weight <= 0:
                adjusted_schedules[cell_id] = cell_sched
                continue

            r1_factor = sum(
                VARIANT_CYCLE_FACTORS.get(v, {}).get("r1", 1.0)
                * variant_mix.get(v, 0.0)
                for v in assigned_variants
            ) / total_weight
            r2_factor = sum(
                VARIANT_CYCLE_FACTORS.get(v, {}).get("r2", 1.0)
                * variant_mix.get(v, 0.0)
                for v in assigned_variants
            ) / total_weight

            # Adjust effective speeds (slower variants reduce effective speed)
            adjusted = cell_sched.model_copy(update={
                "r1_speed_fraction": min(
                    1.0, cell_sched.r1_speed_fraction / r1_factor
                ),
                "r2_speed_fraction": min(
                    1.0, cell_sched.r2_speed_fraction / r2_factor
                ),
            })
            adjusted_schedules[cell_id] = adjusted

        adjusted_factory = factory_schedule.model_copy(update={
            "cell_schedules": adjusted_schedules
        })
        return self.evaluate(adjusted_factory)
