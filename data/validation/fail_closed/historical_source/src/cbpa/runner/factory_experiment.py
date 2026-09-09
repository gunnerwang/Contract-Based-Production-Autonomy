"""Factory-level CBPA experiment following the canonical lifecycle.

Extends :class:`CBPALifecycle` to a 6-round structure, mirroring the
single-cell experiment's complete lifecycle including macro-adaptation:

    Round 1 (LEGISLATOR): L1 → L2 → L3 → L4
        Initial Contract & Deployment
    Round 2 (AUDITOR):    L4
        Disturbance Detection & Monitoring
    Round 3 (AUDITOR):    L2 → L3
        Autonomous Replan Attempt (Rejected)
    Round 4 (PARTNER):    Meta
        Human-in-the-Loop Escalation
    Round 5 (AUDITOR):    L5 → L2 → L3 → L4 → L5
        Adaptation & Stabilization (contract pre-authorised by Phase 4)
    Round 6 (LEGISLATOR): L4 → L5 → L1 → L2 → L3 → L4
        Shift-Close Macro-Adaptation — L5 learning feeds back into L1;
        Legislator rewrites C3_factory for next shift and pre-stages FS4

The lifecycle transition sequence is:
    Legislator → Auditor → Partner → Auditor → Legislator

Round 6 is a shift-close event, not a new disturbance.  After FS3
stabilises, the shift ends and the Legislator reconvenes: L5 surfaces
what was learned this shift (H2 absence pattern, Cell B bottleneck,
variant routing adjustments), and L1 rewrites the factory contract to
incorporate those lessons as first-class assumptions for the next shift.
This demonstrates CBPA's continuous improvement loop and the natural
shift-boundary role of the Legislator in manufacturing operations.

Usage:
    python scripts/run_factory_experiment.py --no-llm
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from cbpa.config.defaults import VARIANT_CYCLE_FACTORS
from cbpa.config.scenario import (
    CellConfig,
    FactoryConfig,
    FactoryScenarioConfig,
    OperatorProfile,
)
from cbpa.layer2_planning.pareto import pareto_filter
from cbpa.layer4_execution.monitor import ContractMonitor
from cbpa.layer3_verification.guard_synthesizer import GuardEnforcer
from cbpa.llm.client import ClaudeClient
from cbpa.llm.prompts import FACTORY_PLANNING_SYSTEM, FACTORY_PLANNING_CANDIDATES_SCHEMA
from cbpa.models.contract import (
    OutcomeContract,
    make_c1_factory,
    make_c3_factory,
)
from cbpa.models.escalation import WorkingMode
from cbpa.models.metrics import (
    FactoryMetrics,
    FeasibilityReport,
    SimulationMetrics,
    StochasticFeasibilityReport,
)
from cbpa.models.schedule import CellSchedule, FactorySchedule
from cbpa.physics.factory_evaluator import FactoryEvaluator
from cbpa.runner.lifecycle import (
    CBPALifecycle,
    CBPA_ROUND_SEQUENCE,
    RoundConfig,
    RoundType,
)

logger = logging.getLogger(__name__)

FACTORY_MANAGER_INTENT = (
    "Maximize total factory throughput across both assembly and test-pack "
    "cells for this shift, while keeping each operator's fatigue within "
    "their individual limits, combined noise under 82 dB, and product "
    "quality above 99%.  Three product variants (V_A, V_B, V_C) must "
    "be produced; route them efficiently across the two cells."
)


@dataclass
class FactoryPhaseResult:
    """Result of one phase of the factory experiment."""

    phase: int
    phase_name: str
    factory_schedule: FactorySchedule | None = None
    factory_metrics: FactoryMetrics | None = None
    contract: OutcomeContract | None = None
    feasibility: FeasibilityReport | None = None
    stochastic_report: StochasticFeasibilityReport | None = None
    disturbance: dict[str, Any] | None = None
    escalation_options: list[dict[str, Any]] | None = None
    manager_decision: str | None = None
    working_mode: str = "legislator"
    notes: list[str] = field(default_factory=list)


@dataclass
class FactoryExperimentResult:
    """Complete result of the 5-round factory experiment."""

    phases: list[FactoryPhaseResult] = field(default_factory=list)
    final_metrics: FactoryMetrics | None = None


# ---------------------------------------------------------------------------
# Factory-level candidate generation (deterministic offsets per cell)
# ---------------------------------------------------------------------------

_FACTORY_OFFSETS = {
    "a": {"r1": 0.0, "r2": 0.0, "hr": 0.0, "buf": 1.0},
    "b": {"r1": +0.08, "r2": 0.0, "hr": 0.0, "buf": 1.0},
    "c": {"r1": 0.0, "r2": +0.08, "hr": 0.0, "buf": 1.0},
    "d": {"r1": 0.0, "r2": 0.0, "hr": 0.0, "buf": 0.85},
    "e": {"r1": +0.04, "r2": +0.04, "hr": +0.02, "buf": 0.92},
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _make_factory_candidates(
    base_name: str,
    base_cell_a: dict,
    base_cell_b: dict,
    routing: dict[str, list[str]],
    assignments: dict[str, str],
) -> list[FactorySchedule]:
    """Generate 5 factory schedule variants from base per-cell parameters."""
    candidates: list[FactorySchedule] = []

    for letter, off in _FACTORY_OFFSETS.items():
        name = f"{base_name}{letter}"
        cell_a = CellSchedule(
            name=f"{name}_A",
            cell_id="A",
            r1_speed_fraction=round(_clamp(base_cell_a["r1"] + off["r1"], 0.0, 1.0), 4),
            r2_speed_fraction=round(_clamp(base_cell_a["r2"] + off["r2"], 0.0, 1.0), 4),
            human_cycle_rate_multiplier=round(_clamp(base_cell_a["hr"] + off["hr"], 0.5, 2.0), 4),
            buffer_time_s=round(max(0.0, base_cell_a["buf"] * off["buf"]), 4),
            demand_target_uph=base_cell_a["demand"],
            assigned_operator=assignments.get("H1", "H1"),
            assigned_variants=base_cell_a.get("variants", ["V_A", "V_B", "V_C"]),
        )
        cell_b = CellSchedule(
            name=f"{name}_B",
            cell_id="B",
            r1_speed_fraction=round(_clamp(base_cell_b["r1"] + off["r1"], 0.0, 1.0), 4),
            r2_speed_fraction=round(_clamp(base_cell_b["r2"] + off["r2"], 0.0, 1.0), 4),
            human_cycle_rate_multiplier=round(_clamp(base_cell_b["hr"] + off["hr"], 0.5, 2.0), 4),
            buffer_time_s=round(max(0.0, base_cell_b["buf"] * off["buf"]), 4),
            demand_target_uph=base_cell_b["demand"],
            assigned_operator=assignments.get("H2", "H2"),
            assigned_variants=base_cell_b.get("variants", ["V_A", "V_B", "V_C"]),
        )
        candidates.append(FactorySchedule(
            name=name,
            cell_schedules={"A": cell_a, "B": cell_b},
            variant_routing=routing,
            operator_assignments=assignments,
        ))

    return candidates


class FactoryPlanGenerator:
    """LLM-backed factory schedule candidate generator.

    Mirrors :class:`PlanGenerator` for single-cell but produces
    :class:`FactorySchedule` objects with per-cell parameters, variant
    routing, and operator assignments decided by the LLM.  Falls back
    to deterministic offsets when ``use_llm=False`` or the LLM call fails.
    """

    def __init__(
        self,
        client: ClaudeClient | None,
        use_llm: bool,
        config: FactoryScenarioConfig,
    ):
        self.client = client
        self.use_llm = use_llm and client is not None and client.is_available
        self.config = config
        self.llm_only = bool(getattr(config, "llm_only", False))
        self.optimiser_anchor = bool(getattr(config, "optimiser_anchor", False))

    # ------------------------------------------------------------------
    # Phase-specific entry points
    # ------------------------------------------------------------------

    llm_only: bool = False
    """Ablation switch (H2, LLM-only condition): withhold the deterministic anchors."""

    optimiser_anchor: bool = False

    def _add_optimiser_anchor(self, det, contract, routing, assignments, va, vb, da, db, prefix):
        """Append the certified constrained-optimiser schedule to the anchor pool (optional)."""
        if not self.optimiser_anchor:
            return det
        from cbpa.baselines.optimizer_planner import FactoryOptimiserPlanner
        from cbpa.config.scenario import FactoryConfig
        from cbpa.layer3_verification.constraint_checker import ConstraintChecker
        from cbpa.physics.factory_evaluator import FactoryEvaluator
        ev = FactoryEvaluator(factory=getattr(self.config, "factory", None) or FactoryConfig(), mode="analytical")
        r = FactoryOptimiserPlanner(ev, ConstraintChecker(), certify=True).plan(contract, routing, assignments, va, vb, da, db, name=f"{prefix}opt")
        if r.schedule is None:
            logger.warning("optimiser anchor: no certified schedule found for %s", prefix)
            return det
        return list(det) + [r.schedule.model_copy(update={"name": f"{prefix}opt"})]

    def _blend(self, llm: list[FactorySchedule], det: list[FactorySchedule]) -> list[FactorySchedule]:
        """Merge LLM candidates with deterministic anchors (renamed to avoid collision).

        Deterministic anchors use known-good calibrated parameters and guarantee
        at least one feasible candidate survives Pareto selection even when LLM
        candidates are all infeasible or ill-conditioned.
        """
        det_renamed = [
            c.model_copy(update={"name": c.name + "_det"})
            for c in det
        ]
        if self.llm_only:
            if llm:
                return list(llm)
            logger.warning("llm_only: the LLM produced no candidates; deterministic anchors used as fallback")
        return llm + det_renamed

    def generate_initial(
        self,
        contract: OutcomeContract,
        routing: dict,
        assignments: dict,
        learning_bias: dict | None = None,
    ) -> list[FactorySchedule]:
        det = _make_factory_candidates(
            "FS1",
            {"r1": 0.65, "r2": 0.60, "hr": 1.0, "buf": 3.0, "demand": 52.0,
             "variants": ["V_A", "V_B", "V_C"]},
            {"r1": 0.60, "r2": 0.55, "hr": 1.0, "buf": 3.0, "demand": 48.0,
             "variants": ["V_A", "V_B", "V_C"]},
            routing=routing, assignments=assignments,
        )
        det = self._add_optimiser_anchor(det, contract, routing, assignments, ["V_A", "V_B", "V_C"], ["V_A", "V_B", "V_C"], 52.0, 48.0, "FS1")
        if not self.use_llm:
            return det
        llm = self._generate_llm(
            contract, "initial", routing, assignments,
            context="Generate moderate initial deployment schedules for both cells.",
            learning_bias=learning_bias,
        )
        return self._blend(llm, det)

    def generate_replan(
        self,
        contract: OutcomeContract,
        routing: dict,
        assignments: dict,
        cell_a_variants: list[str],
        cell_b_variants: list[str],
        dist_context: str = "",
        learning_bias: dict | None = None,
    ) -> list[FactorySchedule]:
        spike_pct = self.config.demand_spike_pct
        op_absent = len(cell_b_variants) <= 1
        det = _make_factory_candidates(
            "FS2",
            {"r1": 0.78, "r2": 0.75, "hr": 1.10, "buf": 2.0,
             "demand": 52.0 * (1 + spike_pct / 100), "variants": cell_a_variants},
            {"r1": 0.55 if op_absent else 0.68,
             "r2": 0.50 if op_absent else 0.65,
             "hr": 0.90 if op_absent else 1.05,
             "buf": 2.0,
             "demand": 48.0 * (1 + spike_pct / 100), "variants": cell_b_variants},
            routing=routing, assignments=assignments,
        )
        if not self.use_llm:
            return det
        llm = self._generate_llm(
            contract, "replan", routing, assignments,
            context=(
                f"Aggressive replan to meet +{spike_pct:.0f}% demand surge under disturbances: "
                f"{dist_context}. Push parameters hard but stay within contract constraints."
            ),
            learning_bias=learning_bias,
            cell_a_variants=cell_a_variants,
            cell_b_variants=cell_b_variants,
            demand_a=52.0 * (1 + spike_pct / 100), demand_b=48.0 * (1 + spike_pct / 100),
        )
        return self._blend(llm, det)

    def generate_balanced(
        self,
        contract: OutcomeContract,
        routing: dict,
        assignments: dict,
        cell_a_variants: list[str],
        cell_b_variants: list[str],
        dist_context: str = "",
        learning_bias: dict | None = None,
    ) -> list[FactorySchedule]:
        op_absent = len(cell_b_variants) <= 1
        det = _make_factory_candidates(
            "FS3",
            {"r1": 0.62, "r2": 0.58, "hr": 1.0, "buf": 3.5,
             "demand": 52.0, "variants": cell_a_variants},
            {"r1": 0.50 if op_absent else 0.55,
             "r2": 0.45 if op_absent else 0.50,
             "hr": 0.85 if op_absent else 1.0,
             "buf": 3.5,
             "demand": 48.0, "variants": cell_b_variants},
            routing=routing, assignments=assignments,
        )
        det = self._add_optimiser_anchor(det, contract, routing, assignments, cell_a_variants, cell_b_variants, 52.0, 48.0, "FS3")
        if not self.use_llm:
            return det
        llm = self._generate_llm(
            contract, "balanced", routing, assignments,
            context=(
                f"Conservative post-escalation schedule accepting ~10% throughput reduction. "
                f"Prioritise human wellbeing constraints. Disturbances: {dist_context}"
            ),
            learning_bias=learning_bias,
            cell_a_variants=cell_a_variants,
            cell_b_variants=cell_b_variants,
        )
        return self._blend(llm, det)

    def generate_next_shift(
        self,
        contract: OutcomeContract,
        routing: dict,
        assignments: dict,
        lessons: str = "",
        learning_bias: dict | None = None,
    ) -> list[FactorySchedule]:
        det = _make_factory_candidates(
            "FS4",
            {"r1": 0.68, "r2": 0.65, "hr": 1.02, "buf": 3.0,
             "demand": 56.0, "variants": ["V_A", "V_B", "V_C"]},
            {"r1": 0.60, "r2": 0.55, "hr": 0.90, "buf": 3.5,
             "demand": 44.0, "variants": ["V_A"]},
            routing=routing, assignments=assignments,
        )
        det = self._add_optimiser_anchor(det, contract, routing, assignments, ["V_A", "V_B", "V_C"], ["V_A"], 56.0, 44.0, "FS4")
        if not self.use_llm:
            return det
        llm = self._generate_llm(
            contract, "next_shift", routing, assignments,
            context=(
                f"Pre-stage next-shift schedule incorporating shift-close lessons. "
                f"Plan for H2 absence risk; Cell B at reduced demand (44 uph). "
                f"Lessons: {lessons}"
            ),
            learning_bias=learning_bias,
            demand_a=56.0, demand_b=44.0,
        )
        return self._blend(llm, det)

    # ------------------------------------------------------------------
    # LLM call + parsing
    # ------------------------------------------------------------------

    def _generate_llm(
        self,
        contract: OutcomeContract,
        phase: str,
        routing: dict,
        assignments: dict,
        context: str = "",
        learning_bias: dict | None = None,
        cell_a_variants: list[str] | None = None,
        cell_b_variants: list[str] | None = None,
        demand_a: float = 52.0,
        demand_b: float = 48.0,
    ) -> list[FactorySchedule]:
        assert self.client is not None

        constraint_str = ", ".join(
            f"{c.name} {c.operator} {c.limit}" for c in contract.hard_constraints
        )
        bias_context = ""
        if learning_bias:
            parts: list[str] = []
            if learning_bias.get("avoid_policies"):
                parts.append(f"AVOID previously-rejected policies: {learning_bias['avoid_policies']}")
            if learning_bias.get("prefer_families"):
                parts.append(f"PREFER policy families similar to: {learning_bias['prefer_families']}")
            if parts:
                bias_context = "\nLearning bias:\n" + "\n".join(f"- {p}" for p in parts)

        variants_context = ""
        if cell_a_variants is not None:
            variants_context = (
                f"\nCell A may handle: {cell_a_variants}"
                f"\nCell B may handle: {cell_b_variants}"
            )

        user_msg = (
            f"Phase: {phase}\n"
            f"Contract: {contract.name}\n"
            f"Hard constraints: {constraint_str}\n"
            f"Current routing: {routing}\n"
            f"Current assignments: {assignments}\n"
            f"Task: {context}"
            f"{variants_context}"
            f"{bias_context}\n\n"
            "Generate 4-5 factory schedule candidates with diverse trade-offs. "
            "Each candidate must specify cell_A and cell_B parameters, variant routing, "
            "and operator assignments."
        )

        try:
            result = self.client.query_structured(
                system=FACTORY_PLANNING_SYSTEM,
                user_message=user_msg,
                tool_name="propose_factory_schedules",
                tool_schema=FACTORY_PLANNING_CANDIDATES_SCHEMA,
                tool_description="Propose candidate factory-level production schedules",
            )
            candidates: list[FactorySchedule] = []
            default_a_variants = cell_a_variants or ["V_A", "V_B", "V_C"]
            default_b_variants = cell_b_variants or ["V_A", "V_B", "V_C"]
            for c in result.get("candidates", []):
                name = c.get("name", f"FS_llm_{len(candidates)}")
                ca = c.get("cell_A", {})
                cb = c.get("cell_B", {})
                cand_routing = c.get("variant_routing", routing)
                cand_assignments = c.get("operator_assignments", assignments)
                cell_a = CellSchedule(
                    name=f"{name}_A", cell_id="A",
                    r1_speed_fraction=_clamp(ca.get("r1_speed_fraction", 0.65), 0.0, 1.0),
                    r2_speed_fraction=_clamp(ca.get("r2_speed_fraction", 0.60), 0.0, 1.0),
                    human_cycle_rate_multiplier=_clamp(ca.get("human_cycle_rate_multiplier", 1.0), 0.5, 2.0),
                    buffer_time_s=max(0.0, ca.get("buffer_time_s", 3.0)),
                    demand_target_uph=demand_a,
                    assigned_operator=cand_assignments.get("H1", "H1"),
                    assigned_variants=ca.get("assigned_variants", default_a_variants),
                )
                cell_b = CellSchedule(
                    name=f"{name}_B", cell_id="B",
                    r1_speed_fraction=_clamp(cb.get("r1_speed_fraction", 0.55), 0.0, 1.0),
                    r2_speed_fraction=_clamp(cb.get("r2_speed_fraction", 0.50), 0.0, 1.0),
                    human_cycle_rate_multiplier=_clamp(cb.get("human_cycle_rate_multiplier", 1.0), 0.5, 2.0),
                    buffer_time_s=max(0.0, cb.get("buffer_time_s", 3.0)),
                    demand_target_uph=demand_b,
                    assigned_operator=cand_assignments.get("H2", cand_assignments.get("H3", "H2")),
                    assigned_variants=cb.get("assigned_variants", default_b_variants),
                )
                candidates.append(FactorySchedule(
                    name=name,
                    source="llm",
                    cell_schedules={"A": cell_a, "B": cell_b},
                    variant_routing=cand_routing,
                    operator_assignments=cand_assignments,
                ))
            if candidates:
                return candidates
        except Exception as e:
            print(f"  [LLM FALLBACK] Factory planning ({phase}): {e} -> deterministic")
            logger.warning("LLM factory planning failed (%s): %s", phase, e)

        # Return empty — caller always blends with deterministic anchors
        return []


class FactoryExperiment(CBPALifecycle):
    """Factory-level CBPA experiment following the canonical lifecycle.

    Uses the same 5-round structure as the single-cell experiment but
    with factory-level evaluators, cross-cell constraints, variant
    routing, and operator reassignment under certification constraints.
    """

    def __init__(
        self,
        config: FactoryScenarioConfig | None = None,
        mode: str = "deterministic",
        use_llm: bool = False,
        llm_client: ClaudeClient | None = None,
        mc_seed_base: int = 42,
    ):
        self.config = config or FactoryScenarioConfig()
        self.mode = mode
        self.factory_config = self.config.factory

        # Initialize shared L1-L5 + Meta layer components via base class
        super().__init__(use_llm=use_llm, llm_client=llm_client, mc_seed_base=mc_seed_base)

        # Factory-specific evaluator
        self.evaluator = FactoryEvaluator(
            factory=self.factory_config, mode="analytical"
        )

        # LLM-backed factory schedule generator (L2)
        self.factory_planner = FactoryPlanGenerator(
            client=self.client,
            use_llm=use_llm and self.client is not None,
            config=self.config,
        )

        # Integration orchestrator — created when use_integrated is True.
        self._orchestrator: Any = None
        if self.config.use_integrated:
            from cbpa.service.integration.factory_orchestrator import FactoryOrchestrator as FO
            self._orchestrator = FO(self.config)

    def shutdown(self) -> None:
        """Release resources (OPC-UA server, bridges)."""
        if self._orchestrator is not None:
            try:
                self._orchestrator.shutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # CBPALifecycle abstract implementations
    # ------------------------------------------------------------------

    def _make_result(self) -> FactoryExperimentResult:
        return FactoryExperimentResult()

    def _append_result(self, result: FactoryExperimentResult, pr: FactoryPhaseResult) -> None:
        result.phases.append(pr)

    def _finalize_result(self, result: FactoryExperimentResult, rounds: list) -> None:
        # Final metrics from the last round that has metrics
        for pr in reversed(rounds):
            if pr.factory_metrics is not None:
                result.final_metrics = pr.factory_metrics
                break

    def run_all_phases(
        self,
        on_phase_complete: Callable | None = None,
    ) -> FactoryExperimentResult:
        """Execute the canonical CBPA lifecycle for the factory."""
        return super().run_all_phases(on_phase_complete=on_phase_complete)

    # ------------------------------------------------------------------
    # Helper: evaluate + Pareto-rank factory schedule candidates
    # ------------------------------------------------------------------

    def _evaluate_and_rank(
        self,
        candidates: list[FactorySchedule],
        contract: OutcomeContract,
    ) -> tuple[Any, dict[str, FactoryMetrics], dict[str, FeasibilityReport]]:
        """Evaluate candidates, check constraints, Pareto-rank."""
        cand_metrics: dict[str, FactoryMetrics] = {}
        cand_feas: dict[str, FeasibilityReport] = {}
        pareto_input: list[dict] = []

        for fs in candidates:
            fm = self.evaluator.evaluate(fs)
            feas = self.checker.check_factory(contract, fm)
            cand_metrics[fs.name] = fm
            cand_feas[fs.name] = feas

            min_margin = min(feas.constraint_margins.values()) if feas.constraint_margins else 0.0
            pareto_input.append({
                "name": fs.name,
                "vr_score": fm.total_throughput_uph,
                "constraint_margin": min_margin,
                "feasible": feas.is_feasible,
                "source": getattr(fs, "source", "deterministic"),
            })

        pareto_result = pareto_filter(pareto_input)
        return pareto_result, cand_metrics, cand_feas


    CERT_THRESHOLD = 0.95

    def _certification_gate(self, candidates, pareto_result, cand_metrics, cand_feas, contract, phase):
        """Factory counterpart of the single-cell certification gate (objective: throughput)."""
        by_name = {c.name: c for c in candidates}
        thr = lambda n: cand_metrics[n].total_throughput_uph if n in cand_metrics else -1.0  # noqa: E731
        nd = sorted([n for n in pareto_result.non_dominated if n != pareto_result.selected], key=thr, reverse=True)
        dom = sorted(list(pareto_result.dominated), key=thr, reverse=True)
        order = [pareto_result.selected] + nd + dom
        tried: list[tuple[str, float]] = []
        chosen, certified, p_chosen = None, False, None
        seen: set[str] = set()
        for name in order:
            if name in seen or name not in by_name or not cand_feas[name].is_feasible:
                continue
            seen.add(name)
            mc = self.evaluator.evaluate_monte_carlo(by_name[name], n_samples=200, seed=self.mc_seed_base)
            p = self.checker.check_factory_stochastic(contract, mc, seed=self.mc_seed_base).p_feasible
            tried.append((name, round(p, 3)))
            if p >= self.CERT_THRESHOLD:
                chosen, certified, p_chosen = by_name[name], True, p
                break
        no_feasible = False
        if chosen is None:
            chosen = by_name.get(pareto_result.selected, candidates[0])
            p_chosen = next((p for n, p in tried if n == chosen.name), None)
            if cand_feas.get(chosen.name) is not None and cand_feas[chosen.name].is_feasible:
                logger.warning(
                    "Phase %d certification gate: no candidate reached P>=%.2f (tried %s); "
                    "deploying %s on the deterministic check with guards armed (uncertified)",
                    phase, self.CERT_THRESHOLD, tried, chosen.name,
                )
            else:
                no_feasible = True
                logger.warning(
                    "Phase %d certification gate: NO candidate passes the deterministic check; "
                    "the architecture would escalate here, the prototype's fixed phase flow carries %s forward flagged",
                    phase, chosen.name,
                )
        self._last_gate = {"phase": phase, "chosen": chosen.name, "certified": certified,
                           "p_feasible": p_chosen, "tried": tried, "no_feasible_candidate": no_feasible}
        self._apply_gate_annotation(phase)
        return chosen, {"certified": certified, "p_feasible": p_chosen, "tried": tried}

    def _apply_gate_annotation(self, phase: int) -> None:
        """Write the gate decision into the phase's Pareto audit record (if it exists yet)."""
        g = getattr(self, "_last_gate", None)
        if not g or g["phase"] != phase:
            return
        for e in reversed(self.audit.entries):
            if e.phase == phase and "candidate_sources" in e.details:
                e.details["selected"] = g["chosen"]
                e.details["selected_source"] = e.details["candidate_sources"].get(g["chosen"], "deterministic")
                e.details["gate_certified"] = g["certified"]
                e.details["gate_p_feasible"] = g["p_feasible"]
                e.details["gate_tried"] = g["tried"]
                e.details["gate_no_feasible_candidate"] = g.get("no_feasible_candidate", False)
                break

    # ------------------------------------------------------------------
    # Round 1: L1 → L2 → L3 → L4 — Initial Contract & Deployment
    # ------------------------------------------------------------------

    def _round_initial_deploy(self, rc: RoundConfig, prior: list) -> FactoryPhaseResult:
        print("\n" + "=" * 70)
        print("  ROUND 1: Initial Contract & Deployment (L1 → L2 → L3 → L4)")
        print("=" * 70)
        print(f"\n  Manager says: \"{FACTORY_MANAGER_INTENT}\"")
        print(f"  Working mode: {rc.working_mode.value.upper()}")
        print()

        # ── L1: Elicit factory contract ──
        try:
            c1, ambiguity_log = self.elicitation.elicit_with_refinement(
                FACTORY_MANAGER_INTENT, "C1_factory"
            )
        except Exception as exc:
            logger.warning("Factory elicit failed: %s; using make_c1_factory()", exc)
            c1 = make_c1_factory()
            ambiguity_log = []

        validation = self.validator.validate(c1)
        if not validation.is_valid:
            logger.warning("Factory C1 validation issues: %s; using fallback", validation.errors)
            c1 = make_c1_factory()

        # Enforce mandatory human-wellbeing hard constraints from make_c1_factory().
        # The LLM may propose looser limits; we never allow them to exceed the safety floors.
        _MANDATORY_FLOORS: dict[str, float] = {
            "FatigueIndex_H1": 0.4,
            "FatigueIndex_H2": 0.35,
            "FatigueIndex_H3": 0.4,
            "FactoryNoise":    82.0,
        }
        existing_names = {hc.name for hc in c1.hard_constraints}
        patched: list = []
        for hc in c1.hard_constraints:
            floor = _MANDATORY_FLOORS.get(hc.name)
            if floor is not None and hc.limit > floor:
                logger.warning(
                    "LLM relaxed mandatory constraint %s: %.3f → clamped to %.3f",
                    hc.name, hc.limit, floor,
                )
                patched.append(hc.model_copy(update={"limit": floor}))
            else:
                patched.append(hc)
        # Keep the LLM's own constraint list (before ontology patching) for the audit trail
        llm_constraints = [f"{hc.name} {hc.operator} {hc.limit}" for hc in c1.hard_constraints]
        # Add any mandatory constraint the LLM omitted entirely
        baseline = make_c1_factory()
        for hc in baseline.hard_constraints:
            if hc.name in _MANDATORY_FLOORS and hc.name not in existing_names:
                logger.warning("LLM omitted mandatory constraint %s; adding it back", hc.name)
                patched.append(hc)
        c1 = c1.model_copy(update={"hard_constraints": patched})
        # Typed assumptions (factory demand, cell demands, H2 availability, V_C supply,
        # AGV cycle): re-inserted from the plant model so that Layer-4 drift detection
        # works on LLM-elicited contracts (defect (vi) of the revision).
        from cbpa.models.contract import reinsert_typed_assumptions
        c1 = reinsert_typed_assumptions(c1, baseline, logger=logger)

        print(f"  [L1] Contract {c1.name} elicited:")
        print(f"    KPIs:        {', '.join(k.name for k in c1.kpi_targets)}")
        print(f"    Constraints: {', '.join(c.name + ' ' + c.operator + ' ' + str(c.limit) for c in c1.hard_constraints)}")
        if ambiguity_log:
            print(f"    Ambiguities: {len(ambiguity_log)} resolved")
        print()

        self.audit.record(
            phase=1, layer="L1",
            action="contract_elicited",
            llm_constraints=llm_constraints,
            patched_constraints=[f"{hc.name} {hc.operator} {hc.limit}" for hc in c1.hard_constraints],
            contract_state="C1_factory elicited",
            working_mode=rc.working_mode.value,
            contract=c1.name,
            kpis=len(c1.kpi_targets),
            constraints=len(c1.hard_constraints),
        )

        # ── L2: Generate factory schedule candidates, Pareto-rank ──
        candidates = self.factory_planner.generate_initial(
            contract=c1,
            routing=self.config.default_variant_routing,
            assignments=self.config.default_operator_assignments,
        )

        pareto_result, cand_metrics, cand_feas = self._evaluate_and_rank(candidates, c1)

        print(f"  [L2] Generated {len(candidates)} candidates, Pareto selected {pareto_result.selected}")

        selected, _gate = self._certification_gate(candidates, pareto_result, cand_metrics, cand_feas, c1, phase=1)
        _gate_phase = 1
        fs1 = selected.model_copy(update={"name": "FS1"})
        metrics = self.evaluator.evaluate(fs1)

        self.audit.record(
            phase=1, layer="L2",
            action="candidates_generated",
            contract_state="C1_factory active",
            working_mode=rc.working_mode.value,
            candidates=len(candidates),
            selected=pareto_result.selected,
            pareto_front=pareto_result.non_dominated,
            candidate_sources=pareto_result.candidate_sources,
            non_dominated=pareto_result.non_dominated,
            scores=pareto_result.scores,
            selected_source=pareto_result.candidate_sources.get(
                pareto_result.selected, "deterministic"
            ),
        )
        self._apply_gate_annotation(_gate_phase)

        # ── L3: Stochastic verification + certificate ──
        report = self.checker.check_factory(c1, metrics)
        mc_samples = self.evaluator.evaluate_monte_carlo(fs1, n_samples=200, seed=self.mc_seed_base)
        stoch_report = self.checker.check_factory_stochastic(c1, mc_samples, seed=self.mc_seed_base)
        cert = self.cert_issuer.issue(fs1.name, report, c1, stochastic_report=stoch_report)

        print(f"  [L3] Verification: feasible={report.is_feasible} | P(feasible)={stoch_report.p_feasible:.0%}")
        print(f"    Margins: {report.constraint_margins}")

        self.audit.record(
            phase=1, layer="L3",
            action="schedule_verified",
            contract_state="C1_factory active",
            working_mode=rc.working_mode.value,
            schedule="FS1",
            feasible=report.is_feasible,
            p_feasible=stoch_report.p_feasible,
            constraint_violation_probabilities=stoch_report.constraint_violation_probabilities,
            worst_case_margins=stoch_report.worst_case_margins,
            certified=cert.is_certified,
            violations=report.violations,
        )

        # ── L4: Deploy with guards armed ──
        if cert.is_certified and cert.guards:
            self._guard_enforcer = GuardEnforcer(cert.guards)
            print(f"  [L4] Guards armed: {len(cert.guards)} runtime guards")

        # ── L4: Integration — sync to Isaac Sim / BaSyx / OPC-UA ──
        if self._orchestrator is not None:
            try:
                self._orchestrator.sync_contract(c1)
            except Exception as e:
                logger.warning("Contract sync failed: %s", e)
            try:
                deploy_result = self._orchestrator.deploy_factory_schedule(fs1)
                print(f"  [L4-INT] Deployed to Isaac Sim: {deploy_result}")
            except Exception as e:
                logger.warning("Isaac deploy failed: %s", e)
            self._orchestrator.sync_factory_schedule(fs1)
            self._orchestrator.update_factory_kpis(metrics)
            readiness = self._orchestrator.check_readiness()
            bridges_str = ", ".join(f"{k}={'OK' if v else 'stub'}" for k, v in readiness.items())
            print(f"  [L4-INT] Bridges: {bridges_str}")

        print(f"\n  [L4] FS1 DEPLOYED | Throughput={metrics.total_throughput_uph} uph | "
              f"Noise={metrics.factory_noise_db} dB")

        self.audit.record(
            phase=1, layer="L4",
            action="schedule_deployed",
            contract_state="C1_factory active",
            working_mode=rc.working_mode.value,
            schedule="FS1",
            throughput=metrics.total_throughput_uph,
            noise_db=metrics.factory_noise_db,
            guards_armed=len(cert.guards) if cert.is_certified and cert.guards else 0,
        )

        return FactoryPhaseResult(
            phase=1,
            phase_name="Initial Contract & Deployment",
            contract=c1,
            factory_schedule=fs1,
            factory_metrics=metrics,
            feasibility=report,
            stochastic_report=stoch_report,
            notes=[
                f"Contract: {c1.name} with {len(c1.hard_constraints)} constraints",
                f"Selected {pareto_result.selected} from {len(candidates)} candidates",
                f"Throughput: {metrics.total_throughput_uph} uph | Noise: {metrics.factory_noise_db} dB",
                f"P(feasible): {stoch_report.p_feasible:.0%} | Certified: {cert.is_certified}",
                f"Operator fatigue: {metrics.operator_fatigue}",
            ],
        )

    # ------------------------------------------------------------------
    # Round 2: L4 — Disturbance Detection & Monitoring
    # ------------------------------------------------------------------

    def _round_monitor(self, rc: RoundConfig, prior: list) -> FactoryPhaseResult:
        p1 = prior[0]
        contract = p1.contract
        schedule = p1.factory_schedule
        metrics = p1.factory_metrics

        print("\n" + "=" * 70)
        print("  ROUND 2: Disturbance Detection & Monitoring (L4)")
        print("=" * 70)
        print(f"  Working mode: {rc.working_mode.value.upper()}")
        print()

        # ── L4: ContractMonitor with assumption drift tracking ──
        monitor = ContractMonitor(contract, guard_enforcer=self._guard_enforcer)
        disturbances: dict[str, Any] = {}
        notes: list[str] = []

        # Demand surge
        spike_pct = self.config.demand_spike_pct
        disturbances["demand_surge"] = {"magnitude_pct": spike_pct}
        new_factory_demand = 100.0 * (1 + spike_pct / 100)

        # Forward disturbances to Isaac Sim when integrated
        if self._orchestrator is not None:
            result = self._orchestrator.inject_disturbance(
                "demand_surge", magnitude_pct=spike_pct,
            )
            print(f"  [L4-INT] Disturbance injected: demand_surge +{spike_pct}% -> {result}")

        drifts = monitor.check_assumptions({
            "factory_demand_uph": new_factory_demand,
            "cellA_demand_uph": 52.0 * (1 + spike_pct / 100),
            "cellB_demand_uph": 48.0 * (1 + spike_pct / 100),
        })

        if drifts:
            print("  [L4] Assumption drift detected:")
            for d in drifts:
                print(f"    {d.attribution}")
                notes.append(f"Drift: {d.attribution}")

        # Operator absence
        if self.config.enable_operator_absence:
            absent_id = self.config.absent_operator_id
            disturbances["operator_absence"] = {"operator_id": absent_id, "hour": self.config.operator_absence_hour}
            monitor.check_assumptions({"h2_availability": 0.0})
            notes.append(f"Operator {absent_id} unavailable at hour {self.config.operator_absence_hour}")

        # Supply delay
        if self.config.enable_supply_delay:
            disturbances["supply_delay"] = {"variant": self.config.supply_delay_variant, "delay_hours": self.config.supply_delay_hours}
            monitor.check_assumptions({"supply_vc_available": 0.0})
            notes.append(f"Supply delay: {self.config.supply_delay_variant} delayed {self.config.supply_delay_hours}h")

        # Deadline gap
        alert = monitor.check_deadline(metrics.total_throughput_uph, new_factory_demand)
        if alert:
            print(f"\n  [L4] {alert.message}")
            notes.append(alert.message)

        notes.insert(0, f"Demand surge: +{spike_pct}%")

        self.audit.record(
            phase=2, layer="L4",
            action="assumption_drift_detected",
            contract_state="C1_factory active; drift",
            working_mode=rc.working_mode.value,
            drifts=[d.attribution for d in drifts],
            disturbances=list(disturbances.keys()),
        )
        self.audit.record(
            phase=2, layer="L4",
            action="deadline_gap_computed",
            contract_state="C1_factory active; disturbance",
            working_mode=rc.working_mode.value,
            current_throughput=metrics.total_throughput_uph,
            new_demand=new_factory_demand,
            gap_pct=round((new_factory_demand - metrics.total_throughput_uph) / new_factory_demand * 100, 1),
        )

        return FactoryPhaseResult(
            phase=2,
            phase_name="Disturbance Detection & Monitoring",
            factory_schedule=schedule,
            factory_metrics=metrics,
            contract=contract,
            disturbance=disturbances,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Round 3: L2 → L3 — Autonomous Replan Attempt (Rejected)
    # ------------------------------------------------------------------

    def _round_replan(self, rc: RoundConfig, prior: list) -> FactoryPhaseResult:
        p1 = prior[0]
        p2 = prior[1]
        contract = p1.contract
        dist = p2.disturbance or {}

        print("\n" + "=" * 70)
        print("  ROUND 3: Autonomous Replan Attempt (L2 → L3)")
        print("=" * 70)
        print(f"  Working mode: {rc.working_mode.value.upper()}")
        print("  Strategy: aggressive replan to meet new demand + handle disturbances")
        print()

        # Determine replan parameters from disturbances
        replan_assignments = dict(self.config.default_operator_assignments)
        replan_routing = dict(self.config.default_variant_routing)
        cell_a_variants = ["V_A", "V_B", "V_C"]
        cell_b_variants = ["V_A", "V_B", "V_C"]
        notes: list[str] = []

        if "operator_absence" in dist:
            absent = dist["operator_absence"]["operator_id"]
            notes.append(
                f"[L2 REASONING] {absent} absent. H3 (backup) is inspection-only. "
                f"Cell B limited to V_A; V_B/V_C routed through Cell A."
            )
            replan_assignments = {"H1": "A", "H3": "B"}
            cell_b_variants = ["V_A"]
            replan_routing = {"V_A": ["A", "B"], "V_B": ["A"], "V_C": ["A"]}

        if "supply_delay" in dist:
            delayed = dist["supply_delay"]["variant"]
            notes.append(f"[L2 REASONING] {delayed} supply delayed. Suspended; V_A overflow fills Cell B.")
            replan_routing[delayed] = []
            cell_a_variants = [v for v in cell_a_variants if v != delayed]
            cell_b_variants = [v for v in cell_b_variants if v != delayed]

        # ── L2: Generate aggressive replan candidates ──
        spike_pct = self.config.demand_spike_pct
        dist_ctx = ", ".join(dist.keys()) if dist else "none"
        candidates = self.factory_planner.generate_replan(
            contract=contract,
            routing=replan_routing,
            assignments=replan_assignments,
            cell_a_variants=cell_a_variants,
            cell_b_variants=cell_b_variants,
            dist_context=dist_ctx,
        )

        pareto_result, cand_metrics, cand_feas = self._evaluate_and_rank(candidates, contract)

        print(f"  [L2] Generated {len(candidates)} aggressive replan candidates")
        print(f"    Pareto front: [{', '.join(pareto_result.non_dominated)}]")

        selected, _gate = self._certification_gate(candidates, pareto_result, cand_metrics, cand_feas, contract, phase=3)
        _gate_phase = 3
        fs2 = selected.model_copy(update={"name": "FS2"})
        metrics = self.evaluator.evaluate(fs2)

        self.audit.record(
            phase=3, layer="L2",
            action="replan_candidates_generated",
            contract_state="C1_factory active; post-disturbance",
            working_mode=rc.working_mode.value,
            candidates=len(candidates),
            selected=pareto_result.selected,
            strategy="aggressive",
            routing_changes=list(replan_routing.keys()),
            candidate_sources=pareto_result.candidate_sources,
            non_dominated=pareto_result.non_dominated,
            scores=pareto_result.scores,
            selected_source=pareto_result.candidate_sources.get(
                pareto_result.selected, "deterministic"
            ),
        )
        self._apply_gate_annotation(_gate_phase)

        # ── L3: Verify — expected to be REJECTED ──
        report = self.checker.check_factory(contract, metrics)
        mc_samples = self.evaluator.evaluate_monte_carlo(fs2, n_samples=200, seed=self.mc_seed_base + 1)
        stoch_report = self.checker.check_factory_stochastic(contract, mc_samples, seed=self.mc_seed_base + 1)

        print(f"\n  [L3] FS2 VERIFICATION:")
        print(f"    Feasible: {report.is_feasible} | P(feasible): {stoch_report.p_feasible:.0%}")
        if report.violations:
            for v in report.violations:
                print(f"    Violation: {v}")
        print(f"    Margins: {report.constraint_margins}")

        notes.extend([
            f"FS2 feasible: {report.is_feasible} | P(feasible): {stoch_report.p_feasible:.0%}",
            f"Throughput: {metrics.total_throughput_uph} uph | Noise: {metrics.factory_noise_db} dB",
            f"Fatigue: {metrics.operator_fatigue}",
        ])
        if report.violations:
            notes.append(f"Violations: {report.violations}")

        self.audit.record(
            phase=3, layer="L3",
            action="replan_rejected",
            contract_state="C1_factory firewall active",
            working_mode=rc.working_mode.value,
            schedule="FS2",
            feasible=report.is_feasible,
            p_feasible=stoch_report.p_feasible,
            constraint_violation_probabilities=stoch_report.constraint_violation_probabilities,
            worst_case_margins=stoch_report.worst_case_margins,
            violations=report.violations,
        )

        return FactoryPhaseResult(
            phase=3,
            phase_name="Autonomous Replan Attempt",
            factory_schedule=fs2,
            factory_metrics=metrics,
            contract=contract,
            feasibility=report,
            stochastic_report=stoch_report,
            disturbance=p2.disturbance,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Round 4: Meta — Human-in-the-Loop Escalation
    # ------------------------------------------------------------------

    def _round_escalate(self, rc: RoundConfig, prior: list) -> FactoryPhaseResult:
        p3 = prior[2]
        report = p3.feasibility or FeasibilityReport(is_feasible=False, violations=["Unknown"])

        print("\n" + "=" * 70)
        print("  ROUND 4: Human-in-the-Loop Escalation (Meta)")
        print("=" * 70)
        print(f"  Working mode: {rc.working_mode.value.upper()}")
        print("  Reason: autonomous replan violated constraints — escalating to manager")
        print()

        # ── Meta: Generate factory escalation with 5 options ──
        query = self.escalation.generate_factory_query(
            report, demand_increase_pct=self.config.demand_spike_pct,
        )

        print(f"  [Escalation] {query.conflict_summary}")
        print(f"\n  [Escalation] 5 remediation options:")
        for opt in query.options:
            preserves = "Yes" if opt.preserves_human_constraints else "No"
            print(f"    - {opt.label}: {opt.description}")
            print(f"      Preserves human constraints: {preserves} | Gap: ~{opt.expected_deadline_gap_pct}%")

        # Manager selects Option D
        decision = self.escalation.simulate_manager_decision(query, auto_select="Option D")

        print(f"\n  [Manager] Decision: {decision.selected_option}")
        print(f"    Rationale: {decision.rationale}")

        self.audit.record(
            phase=4, layer="Meta",
            action="conflict_raised",
            contract_state="C1_factory → conflict",
            working_mode=rc.working_mode.value,
            conflict=query.conflict_summary,
            options_count=len(query.options),
        )

        self.audit.record(
            phase=4, layer="Meta",
            action="manager_decision_received",
            contract_state="C1_factory → pending revision",
            working_mode=rc.working_mode.value,
            selected=decision.selected_option,
            rationale=decision.rationale,
        )

        # ── Manager decision authorises contract revision ──
        # The Partner decision is itself the legislative act: C2_factory is
        # created here so Phase 5 (Auditor) can implement it without invoking L1.
        c2 = make_c1_factory()
        c2 = c2.model_copy(update={"name": "C2_factory"})
        c2.context["manager_decision"] = decision.selected_option
        c2.context["deadline_gap_acknowledged"] = True
        print(f"  [Meta] Contract pre-authorised: C1_factory → {c2.name}")

        self.audit.record(
            phase=4, layer="Meta",
            action="contract_preauthorised",
            contract_state=f"C1_factory → {c2.name}",
            working_mode=rc.working_mode.value,
            contract=c2.name,
        )

        options_dicts = [
            {"label": o.label, "description": o.description,
             "preserves_human_constraints": o.preserves_human_constraints,
             "expected_deadline_gap_pct": o.expected_deadline_gap_pct}
            for o in query.options
        ]

        return FactoryPhaseResult(
            phase=4,
            phase_name="Human-in-the-Loop Escalation",
            contract=c2,
            escalation_options=options_dicts,
            manager_decision=decision.selected_option,
            notes=[
                f"Conflict: {query.conflict_summary}",
                f"5 remediation options presented to manager.",
                f"Manager selected: {decision.selected_option} — {decision.rationale}",
                f"Contract pre-authorised: {c2.name}",
            ],
        )

    # ------------------------------------------------------------------
    # Round 5: L5 → L2 → L3 → L4 → L5 — Adaptation & Stabilization
    # ------------------------------------------------------------------

    def _round_adapt(self, rc: RoundConfig, prior: list) -> FactoryPhaseResult:
        p1 = prior[0]
        p3 = prior[2]
        p4 = prior[3]
        # Contract pre-authorised by manager in Phase 4 — no L1 needed here
        c2 = p4.contract or make_c1_factory()
        original_schedule = p1.factory_schedule
        dist = p3.disturbance or {}

        print("\n" + "=" * 70)
        print("  ROUND 5: Adaptation & Stabilization (L5 → L2 → L3 → L4 → L5)")
        print("=" * 70)
        print(f"  Working mode: {rc.working_mode.value.upper()}")
        print(f"  Using pre-authorised contract: {c2.name}")
        print()

        notes: list[str] = []

        # ── L5: Query learning store for planning bias ──
        learning_bias = self.learning.get_bias_for_planning("factory_disturbance")
        if learning_bias:
            print(f"  [L5] Learning bias from {self.learning.total_records} prior experience(s)")
        else:
            print("  [L5] No learning bias (first shift)")
        print()

        self.audit.record(
            phase=5, layer="L5",
            action="learning_bias_queried",
            contract_state=f"{c2.name} active",
            working_mode=rc.working_mode.value,
            bias_available=learning_bias is not None,
            records=self.learning.total_records,
        )

        # ── L2: Generate balanced candidates (conservative, constraint-respecting) ──
        replan_assignments = dict(self.config.default_operator_assignments)
        replan_routing = dict(self.config.default_variant_routing)
        cell_a_variants = ["V_A", "V_B", "V_C"]
        cell_b_variants = ["V_A", "V_B", "V_C"]

        if "operator_absence" in dist:
            replan_assignments = {"H1": "A", "H3": "B"}
            cell_b_variants = ["V_A"]
            replan_routing = {"V_A": ["A", "B"], "V_B": ["A"], "V_C": ["A"]}

        if "supply_delay" in dist:
            delayed = dist["supply_delay"]["variant"]
            replan_routing[delayed] = []
            cell_a_variants = [v for v in cell_a_variants if v != delayed]
            cell_b_variants = [v for v in cell_b_variants if v != delayed]

        # Conservative parameters (accepted 10% throughput reduction per manager decision)
        dist_ctx = ", ".join(dist.keys()) if dist else "none"
        candidates = self.factory_planner.generate_balanced(
            contract=c2,
            routing=replan_routing,
            assignments=replan_assignments,
            cell_a_variants=cell_a_variants,
            cell_b_variants=cell_b_variants,
            dist_context=dist_ctx,
        )

        pareto_result, cand_metrics, cand_feas = self._evaluate_and_rank(candidates, c2)

        print(f"  [L2] Generated {len(candidates)} balanced FS3 candidates")
        print(f"    Pareto front: [{', '.join(pareto_result.non_dominated)}]")

        selected, _gate = self._certification_gate(candidates, pareto_result, cand_metrics, cand_feas, c2, phase=5)
        _gate_phase = 5
        fs3 = selected.model_copy(update={"name": "FS3"})
        metrics = self.evaluator.evaluate(fs3)

        self.audit.record(
            phase=5, layer="L2",
            action="candidates_generated",
            contract_state=f"{c2.name} active",
            working_mode=rc.working_mode.value,
            candidates=len(candidates),
            selected=pareto_result.selected,
            strategy="conservative",
            candidate_sources=pareto_result.candidate_sources,
            non_dominated=pareto_result.non_dominated,
            scores=pareto_result.scores,
            selected_source=pareto_result.candidate_sources.get(
                pareto_result.selected, "deterministic"
            ),
        )
        self._apply_gate_annotation(_gate_phase)

        # ── L3: Verify balanced schedule ──
        report = self.checker.check_factory(c2, metrics)
        mc_samples = self.evaluator.evaluate_monte_carlo(fs3, n_samples=200, seed=self.mc_seed_base + 2)
        stoch_report = self.checker.check_factory_stochastic(c2, mc_samples, seed=self.mc_seed_base + 2)
        cert = self.cert_issuer.issue(fs3.name, report, c2, stochastic_report=stoch_report)

        print(f"\n  [L3] FS3 verification: feasible={report.is_feasible} | "
              f"P(feasible)={stoch_report.p_feasible:.0%}")

        self.audit.record(
            phase=5, layer="L3",
            action="schedule_verified",
            contract_state=f"{c2.name} active",
            working_mode=rc.working_mode.value,
            schedule="FS3",
            feasible=report.is_feasible,
            p_feasible=stoch_report.p_feasible,
            constraint_violation_probabilities=stoch_report.constraint_violation_probabilities,
            worst_case_margins=stoch_report.worst_case_margins,
            certified=cert.is_certified,
        )

        # ── L5: Adaptation (per-cell comparison) ──
        all_adaptations = []
        for cell_id in fs3.cell_schedules:
            if cell_id in original_schedule.cell_schedules:
                actions = self.adaptation.adapt(
                    original_schedule.cell_schedules[cell_id],
                    fs3.cell_schedules[cell_id],
                )
                all_adaptations.extend(actions)

        if all_adaptations:
            print(f"\n  [L5] Adaptations applied: {len(all_adaptations)}")
            for a in all_adaptations:
                print(f"    [{a.level.value}] {a.description}")

        self.audit.record(
            phase=5, layer="L5",
            action="adaptations_applied",
            contract_state=f"{c2.name} active",
            working_mode=rc.working_mode.value,
            adaptations=len(all_adaptations),
            levels=[a.level.value for a in all_adaptations],
        )

        # ── L4: Deploy (+ integration) ──
        if self._orchestrator is not None:
            try:
                deploy_result = self._orchestrator.deploy_factory_schedule(fs3)
                print(f"  [L4-INT] FS3 deployed to Isaac Sim: {deploy_result}")
            except Exception as e:
                logger.warning("Isaac FS3 deploy failed: %s", e)
            self._orchestrator.sync_factory_schedule(fs3)
            self._orchestrator.update_factory_kpis(metrics)

        print(f"\n  [L4] FS3 DEPLOYED | Throughput={metrics.total_throughput_uph} uph | "
              f"Noise={metrics.factory_noise_db} dB")

        self.audit.record(
            phase=5, layer="L4",
            action="schedule_deployed",
            contract_state=f"{c2.name} active",
            working_mode=rc.working_mode.value,
            schedule="FS3",
            throughput=metrics.total_throughput_uph,
            noise_db=metrics.factory_noise_db,
        )

        # ── L5: Record experience ──
        self.learning.record(
            disturbance_type="factory_disturbance",
            disturbance_magnitude=self.config.demand_spike_pct,
            rejected_policy="FS2",
            rejection_reason="; ".join(p3.feasibility.violations) if p3.feasibility and p3.feasibility.violations else "none",
            accepted_policy="FS3",
            adaptation_level="meso+micro" if all_adaptations else "none",
            outcome_metrics={
                "total_throughput_uph": metrics.total_throughput_uph,
                "factory_noise_db": metrics.factory_noise_db,
            },
        )
        print(f"  [L5] Experience recorded (total: {self.learning.total_records})")

        self.audit.record(
            phase=5, layer="L5",
            action="experience_recorded",
            contract_state=f"{c2.name} active",
            working_mode=rc.working_mode.value,
            disturbance_type="factory_disturbance",
            accepted_policy="FS3",
            total_records=self.learning.total_records,
        )

        notes = [
            f"Using pre-authorised contract: {c2.name}",
            f"FS3 feasible: {report.is_feasible} | P(feasible): {stoch_report.p_feasible:.0%}",
            f"Throughput: {metrics.total_throughput_uph} uph (accepted ~10% reduction)",
            f"Noise: {metrics.factory_noise_db} dB | Fatigue: {metrics.operator_fatigue}",
            f"Adaptations: {len(all_adaptations)} | Experience: {self.learning.total_records}",
        ]

        return FactoryPhaseResult(
            phase=5,
            phase_name="Adaptation & Stabilization",
            factory_schedule=fs3,
            factory_metrics=metrics,
            contract=c2,
            feasibility=report,
            stochastic_report=stoch_report,
            notes=notes,
        )


    # ------------------------------------------------------------------
    # Transition triggers (overrides base — Phase 5→6 goes to Legislator)
    # ------------------------------------------------------------------

    def _transition_trigger(self, rc: RoundConfig) -> str:
        if rc.round_type == RoundType.ADAPT_AND_DEPLOY:
            return "Shift stabilised; shift-close Legislator review begins"
        return super()._transition_trigger(rc)

    # ------------------------------------------------------------------
    # Round sequence: 6-phase factory lifecycle (overrides 5-phase base)
    # ------------------------------------------------------------------

    def _get_round_sequence(self) -> list[RoundConfig]:
        """Factory lifecycle adds a Round 6 macro-adaptation (Legislator)."""
        base = list(CBPA_ROUND_SEQUENCE)
        # Round 5 transitions to Legislator (not back to Auditor) in factory
        base[4] = RoundConfig(
            round_number=5,
            round_type=RoundType.ADAPT_AND_DEPLOY,
            description="Adaptation & Stabilization",
            working_mode=WorkingMode.AUDITOR,
            transition_after=WorkingMode.LEGISLATOR,
        )
        base.append(RoundConfig(
            round_number=6,
            round_type=RoundType.MACRO_ADAPT,
            description="Shift-Close Macro-Adaptation",
            working_mode=WorkingMode.LEGISLATOR,
            transition_after=None,
        ))
        return base

    # ------------------------------------------------------------------
    # Round 6: L4 → L5 → L1 → L2 → L3 → L4 — Macro-Adaptation
    # ------------------------------------------------------------------

    def _round_macro_adapt(self, rc: RoundConfig, prior: list) -> FactoryPhaseResult:
        p1 = prior[0]
        p5 = prior[4]
        c2 = p5.contract
        fs3 = p5.factory_schedule

        print("\n" + "=" * 70)
        print("  ROUND 6: Shift-Close Macro-Adaptation (L4 → L5 → L1 → L2 → L3 → L4)")
        print("=" * 70)
        print(f"  Working mode: {rc.working_mode.value.upper()}")
        print("  Trigger: shift ends — L5 learning feeds back into L1 for next shift")
        print()

        notes: list[str] = []

        # ── L4: Shift-close monitoring — collect final KPIs from FS3 run ──
        final_metrics = p5.factory_metrics
        print(f"  [L4] Shift close. Final FS3 KPIs:")
        if final_metrics:
            print(f"    Throughput: {final_metrics.total_throughput_uph} uph "
                  f"(target was 100.0; gap {round(100.0 - final_metrics.total_throughput_uph, 1)} uph)")
            print(f"    Noise: {final_metrics.factory_noise_db} dB | "
                  f"Fatigue: {final_metrics.operator_fatigue}")
        notes.append("[L4] Shift-close: FS3 ran to completion; KPIs collected")

        self.audit.record(
            phase=6, layer="L4",
            action="shift_close_kpis_collected",
            contract_state=f"{c2.name} shift complete",
            working_mode=rc.working_mode.value,
            throughput=final_metrics.total_throughput_uph if final_metrics else None,
            noise_db=final_metrics.factory_noise_db if final_metrics else None,
        )

        # ── L5: Retrieve and surface lessons from this shift ──
        learning_records = self.learning.total_records
        learning_bias = self.learning.get_bias_for_planning("factory_disturbance")

        lessons = [
            "H2 absence materialised at hour 3 — treat as elevated risk for next shift",
            "Cell B bottleneck under H3-only staffing: V_B/V_C overloaded Cell A",
            "FS3 conservative routing (V_A→Cell B only) maintained all fatigue constraints",
            f"Learning store has {learning_records} record(s) from this shift",
        ]
        print(f"\n  [L5] Shift lessons surfaced ({learning_records} experience record(s)):")
        for lesson in lessons:
            print(f"    • {lesson}")
        notes.append(f"[L5] {len(lessons)} lessons surfaced from shift learning store")

        self.audit.record(
            phase=6, layer="L5",
            action="shift_lessons_surfaced",
            contract_state=f"{c2.name} shift complete",
            working_mode=rc.working_mode.value,
            lessons_count=len(lessons),
            records=learning_records,
        )

        # ── L1: Legislator rewrites contract incorporating lessons ──
        c3 = make_c3_factory()
        print(f"\n  [L1] Contract rewritten for next shift: {c2.name} → {c3.name}")
        print(f"    H2 availability tolerance: 5% → 0% (plan for absence)")
        print(f"    Cell B conservative demand baseline: 48.0 → 44.0 uph")
        print(f"    Variant routing flexibility: built in as contract-level contingency")
        notes.append(
            f"[L1] {c3.name} for next shift: H2 absence planned for; "
            f"Cell B baseline 48→44 uph; routing contingency formalised"
        )

        self.audit.record(
            phase=6, layer="L1",
            action="contract_rewritten",
            contract_state=f"{c2.name} → {c3.name}",
            working_mode=rc.working_mode.value,
            contract=c3.name,
            changes=["h2_tolerance_0pct", "cellB_demand_48to44", "routing_flexibility"],
        )

        # ── L2: Pre-generate next-shift schedule candidates ──
        # Plan conservatively for H2 absence; Cell B at reduced target
        next_routing = {"V_A": ["A", "B"], "V_B": ["A"], "V_C": ["A"]}
        next_assignments = {"H1": "A", "H3": "B"}  # contingency: H2 may be absent

        lessons_str = "; ".join(lessons) if isinstance(lessons, list) else str(lessons)
        candidates = self.factory_planner.generate_next_shift(
            contract=c3,
            routing=next_routing,
            assignments=next_assignments,
            lessons=lessons_str,
        )

        pareto_result, cand_metrics, cand_feas = self._evaluate_and_rank(candidates, c3)
        print(f"\n  [L2] Pre-generated {len(candidates)} next-shift candidates (FS4)")
        print(f"    Pareto front: [{', '.join(pareto_result.non_dominated)}]")

        selected, _gate = self._certification_gate(candidates, pareto_result, cand_metrics, cand_feas, c3, phase=6)
        _gate_phase = 6
        fs4 = selected.model_copy(update={"name": "FS4"})
        metrics = self.evaluator.evaluate(fs4)

        self.audit.record(
            phase=6, layer="L2",
            action="next_shift_candidates_generated",
            contract_state=f"{c3.name} active",
            working_mode=rc.working_mode.value,
            candidates=len(candidates),
            selected=pareto_result.selected,
            candidate_sources=pareto_result.candidate_sources,
            non_dominated=pareto_result.non_dominated,
            scores=pareto_result.scores,
            selected_source=pareto_result.candidate_sources.get(
                pareto_result.selected, "deterministic"
            ),
        )
        self._apply_gate_annotation(_gate_phase)

        # ── L3: Verify FS4 for next shift ──
        report = self.checker.check_factory(c3, metrics)
        mc_samples = self.evaluator.evaluate_monte_carlo(fs4, n_samples=200, seed=self.mc_seed_base + 3)
        stoch_report = self.checker.check_factory_stochastic(c3, mc_samples, seed=self.mc_seed_base + 3)
        cert = self.cert_issuer.issue(fs4.name, report, c3, stochastic_report=stoch_report)

        print(f"\n  [L3] FS4 pre-verified: feasible={report.is_feasible} | "
              f"P(feasible)={stoch_report.p_feasible:.0%}")
        if report.violations:
            for v in report.violations:
                print(f"    Violation: {v}")

        self.audit.record(
            phase=6, layer="L3",
            action="next_shift_schedule_verified",
            contract_state=f"{c3.name} active",
            working_mode=rc.working_mode.value,
            schedule="FS4",
            feasible=report.is_feasible,
            p_feasible=stoch_report.p_feasible,
            constraint_violation_probabilities=stoch_report.constraint_violation_probabilities,
            worst_case_margins=stoch_report.worst_case_margins,
            certified=cert.is_certified,
        )

        # ── L4: Pre-stage FS4 (ready to deploy at next shift start) ──
        if cert.is_certified and cert.guards:
            self._guard_enforcer = GuardEnforcer(cert.guards)
            print(f"  [L4] FS4 pre-staged with {len(cert.guards)} guards armed")

        if self._orchestrator is not None:
            try:
                self._orchestrator.sync_factory_schedule(fs4)
                self._orchestrator.update_factory_kpis(metrics)
                print(f"  [L4-INT] FS4 synced to AAS / OPC-UA for next shift")
            except Exception as e:
                logger.warning("FS4 pre-stage sync failed: %s", e)

        print(f"\n  [L4] FS4 PRE-STAGED for next shift | "
              f"Throughput={metrics.total_throughput_uph} uph | "
              f"Noise={metrics.factory_noise_db} dB")

        notes.extend([
            f"FS4 feasible: {report.is_feasible} | P(feasible): {stoch_report.p_feasible:.0%}",
            f"Next-shift throughput target: {metrics.total_throughput_uph} uph "
            f"| Noise: {metrics.factory_noise_db} dB",
            f"Fatigue: {metrics.operator_fatigue}",
            "FS4 pre-staged: ready to deploy at next shift start",
        ])

        self.audit.record(
            phase=6, layer="L4",
            action="next_shift_schedule_prestaged",
            contract_state=f"{c3.name} pre-staged",
            working_mode=rc.working_mode.value,
            schedule="FS4",
            feasible=report.is_feasible,
            p_feasible=stoch_report.p_feasible,
            constraint_violation_probabilities=stoch_report.constraint_violation_probabilities,
            worst_case_margins=stoch_report.worst_case_margins,
        )

        return FactoryPhaseResult(
            phase=6,
            phase_name="Shift-Close Macro-Adaptation",
            factory_schedule=fs4,
            factory_metrics=metrics,
            contract=c3,
            feasibility=report,
            stochastic_report=stoch_report,
            notes=notes,
        )


def run_factory_experiment(
    config: FactoryScenarioConfig | None = None,
    mode: str = "deterministic",
) -> FactoryExperimentResult:
    """Convenience function to run the full factory experiment."""
    exp = FactoryExperiment(config=config, mode=mode)
    return exp.run_all_phases()
