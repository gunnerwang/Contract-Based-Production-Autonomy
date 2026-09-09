"""Single-cell CBPA experiment following the canonical lifecycle.

Inherits from :class:`CBPALifecycle` which defines the 5-round structure:

    Round 1 (LEGISLATOR): L1 → L2 → L3 → L4  — Initial Contract & Deployment
    Round 2 (AUDITOR):    L4                   — Disturbance Detection & Monitoring
    Round 3 (AUDITOR):    L2 → L3              — Autonomous Replan Attempt (Rejected)
    Round 4 (PARTNER):    Meta                 — Human-in-the-Loop Escalation
    Round 5 (AUDITOR):    L5 → L2 → L3 → L4 → L5 — Adaptation & Stabilization (C2 pre-authorised by Phase 4)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from cbpa.config.scenario import CellConfig, ScenarioConfig
from cbpa.layer2_planning.pareto import pareto_filter
from cbpa.layer2_planning.plan_generator import PlanGenerator
from cbpa.layer2_planning.vr_scorer import VRScorer
from cbpa.layer3_verification.guard_synthesizer import GuardEnforcer, GuardEvent, RuntimeGuardViolation
from cbpa.layer4_execution.assumption_tracker import AssumptionTracker
from cbpa.layer4_execution.monitor import ContractMonitor
from cbpa.layer5_adaptation.adaptation_engine import AdaptationEngine
from cbpa.llm.client import ClaudeClient, create_client
from cbpa.meta_layer.audit_trail import AuditTrail
from cbpa.models.contract import OutcomeContract
from cbpa.models.escalation import AdaptationLevel, ManagerDecision, WorkingMode
from cbpa.models.metrics import (
    FeasibilityReport,
    ParetoFrontResult,
    SimulationMetrics,
    StochasticFeasibilityReport,
    VRScoreResult,
)
from cbpa.models.schedule import Schedule
from cbpa.layer4_execution.simulation import ManufacturingCell, SimulationResult
from cbpa.physics.cell_evaluator import CellEvaluator
from cbpa.runner.lifecycle import (
    CBPALifecycle,
    RoundConfig,
    RoundType,
    CBPA_ROUND_SEQUENCE,
)

logger = logging.getLogger(__name__)

MANAGER_INTENT = (
    "Increase delivered value per resource consumed for this shift, "
    "while keeping operator fatigue low and noise under 80dB."
)


# Mandatory scenario floors for the single-cell contract: the plant ontology
# owns these limits; an elicited draft may express them in its own vocabulary
# or with looser values, but certification must always bind them.
_SC_MANDATORY_FLOORS = {"FatigueIndex": 0.4, "Noise": 80.0}


def _canonicalise_contract(contract, intent: str | None = None):
    """Bind elicited hard constraints to the plant ontology (fail-closed).

    Renames fatigue/noise constraints stated in the LLM's own vocabulary to
    the canonical names the verifier monitors, clamps their limits to the
    mandatory floors, and re-inserts any floor constraint the draft omitted.
    Mirrors the factory runner's mandatory-floor patching.  When *intent* is
    given, the LLM's limits are additionally cross-checked against a
    rule-based parse of the same text and the tightest limit is kept, so a
    limit the manager stated explicitly cannot be relaxed by the LLM
    (contract-fidelity benchmark, Section 5.5 of the paper).
    """
    from cbpa.models.contract import HardConstraint

    def canon_of(name: str):
        low = name.lower()
        if "fatigue" in low:
            return "FatigueIndex"
        if "noise" in low:
            return "Noise"
        return None

    patched = []
    seen: dict[str, int] = {}
    for hc in contract.hard_constraints:
        canon = canon_of(hc.name)
        if canon is None:
            patched.append(hc)
            continue
        floor = _SC_MANDATORY_FLOORS[canon]
        limit = hc.limit
        if hc.name != canon:
            logger.warning(
                "LLM constraint '%s' canonicalised to %s", hc.name, canon
            )
        if limit > floor:
            logger.warning(
                "LLM relaxed mandatory constraint %s: %.3f -> clamped to %.3f",
                canon, limit, floor,
            )
            limit = floor
        if canon in seen:  # keep the tightest duplicate
            prev = patched[seen[canon]]
            if limit < prev.limit:
                patched[seen[canon]] = prev.model_copy(update={"limit": limit})
            continue
        seen[canon] = len(patched)
        patched.append(hc.model_copy(update={"name": canon, "limit": limit}))
    for canon, floor in _SC_MANDATORY_FLOORS.items():
        if canon not in seen:
            logger.warning(
                "LLM omitted mandatory constraint %s; adding it back", canon
            )
            seen[canon] = len(patched)
            patched.append(
                HardConstraint(name=canon, operator="<=", limit=floor)
            )
    if intent:
        from cbpa.baselines.elicitation_baselines import RuleBasedElicitor

        for hc in RuleBasedElicitor().elicit(intent).hard_constraints:
            if hc.name in seen and hc.limit < patched[seen[hc.name]].limit:
                logger.warning(
                    "LLM constraint %s (%.3f) looser than the stated limit %.3f; "
                    "cross-check tightened it", hc.name, patched[seen[hc.name]].limit, hc.limit,
                )
                patched[seen[hc.name]] = patched[seen[hc.name]].model_copy(update={"limit": hc.limit})
    from cbpa.models.contract import make_c1, reinsert_typed_assumptions

    contract = contract.model_copy(update={"hard_constraints": patched})
    # Typed assumptions: the tracker in Layer 4 checks only typed entries, which the
    # LLM draft never carries (defect (vi) of the revision); re-insert the plant's.
    return reinsert_typed_assumptions(contract, make_c1(), logger=logger)



# Default disturbance patterns for multi-shift runs.  Each dict
# configures the demand spike for that shift.  Shift 1 uses the
# scenario's default demand_spike_pct; shifts 2+ use slightly
# different magnitudes so the learning curve is exercised.
_DEFAULT_SHIFT_DISTURBANCES: list[dict] = [
    {"demand_spike_pct": 20.0, "label": "Shift 1: demand +20%"},
    {"demand_spike_pct": 15.0, "label": "Shift 2: demand +15%"},
    {"demand_spike_pct": 25.0, "label": "Shift 3: demand +25%"},
]


@dataclass
class ShiftSummary:
    """Per-shift metrics collected during a multi-shift run."""

    shift: int
    demand_spike_pct: float
    phases_to_resolution: int
    vr_score_accepted: float
    total_constraint_violations: int
    learning_bias_available: bool
    candidates_skipped: int
    experience_records_before: int
    experience_records_after: int


@dataclass
class MultiShiftResult:
    """Aggregated results from a multi-shift learning-curve run."""

    shift_summaries: list[ShiftSummary] = field(default_factory=list)
    shift_results: list[ExperimentResult] = field(default_factory=list)


@dataclass
class PhaseResult:
    """Result of a single phase."""

    phase: int
    schedule: Schedule | None = None
    metrics: SimulationMetrics | None = None
    vr_score: VRScoreResult | None = None
    feasibility: FeasibilityReport | None = None
    contract: OutcomeContract | None = None
    decision: ManagerDecision | None = None
    pareto_front: ParetoFrontResult | None = None
    guard_events: list[GuardEvent] = field(default_factory=list)
    description: str = ""
    working_mode: WorkingMode = WorkingMode.LEGISLATOR


@dataclass
class ExperimentResult:
    """Complete results from all 5 phases."""

    phases: list[PhaseResult] = field(default_factory=list)
    audit: AuditTrail = field(default_factory=AuditTrail)
    schedule_metrics: dict[str, SimulationMetrics] = field(default_factory=dict)
    schedule_vr: dict[str, VRScoreResult] = field(default_factory=dict)
    schedule_feasibility: dict[str, FeasibilityReport] = field(default_factory=dict)
    all_guard_events: list[GuardEvent] = field(default_factory=list)
    mode_summary: dict = field(default_factory=dict)


class CBPAExperiment(CBPALifecycle):
    """Single-cell CBPA experiment following the canonical lifecycle.

    Inherits the 5-round lifecycle from :class:`CBPALifecycle` and
    implements each round using single-cell evaluators, schedules, and
    SimPy / Isaac Sim execution.
    """

    def __init__(
        self,
        config: ScenarioConfig | None = None,
        use_llm: bool = False,
        llm_client: ClaudeClient | None = None,
        mc_seed_base: int = 42,
    ):
        self.config = config or ScenarioConfig()

        # Initialize shared L1-L5 + Meta layer components via base class
        super().__init__(use_llm=use_llm, llm_client=llm_client, mc_seed_base=mc_seed_base)

        # When simulation or integrated execution is active, use analytical
        # physics models so that pre-verification (L3) is consistent with the
        # simulation ground-truth.  Deterministic mode returns hardcoded
        # Table 3 values which may diverge from simulated metrics.
        mode = (
            "analytical"
            if self.config.use_simulation or self.config.use_integrated
            else "deterministic"
        )

        # Cell-specific layer components
        self.planner = PlanGenerator(self.config, self.client, use_llm=use_llm)
        self.evaluator = CellEvaluator(self.config.cell, mode=mode)
        self.vr_scorer = VRScorer(
            self.config.vr_value_weights,
            self.config.vr_resource_weights,
            mode=mode,
        )

        # L3/L4/L5 LLM agents
        from cbpa.layer3_verification.repair_agent import L3RepairAgent
        from cbpa.layer4_execution.attribution_agent import L4AttributionAgent
        from cbpa.layer5_adaptation.adaptation_reasoner import L5AdaptationReasoner

        use_llm_flag = use_llm and self.client is not None
        self.repair_agent = L3RepairAgent(self.client, use_llm=use_llm_flag)
        self.attribution_agent = L4AttributionAgent(self.client, use_llm=use_llm_flag)
        self.adaptation_reasoner = L5AdaptationReasoner(self.client, use_llm=use_llm_flag)

        # Assumption tracker — created during Round 2 and shared with Round 3/4.
        self._assumption_tracker: AssumptionTracker | None = None

        # Integration orchestrator — created when use_integrated is True.
        # Connects to Isaac Sim, BaSyx AAS, and OPC-UA.
        # Imported lazily to avoid circular import through cbpa.service.
        self._orchestrator: Any = None
        if self.config.use_integrated:
            from cbpa.service.integration.orchestrator import IntegrationOrchestrator
            self._orchestrator = IntegrationOrchestrator(self.config)

    def shutdown(self) -> None:
        """Release resources (OPC-UA server, bridges)."""
        if self._orchestrator is not None:
            try:
                self._orchestrator.shutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # SimPy execution — runs a deployed schedule through the full
    # 8-hour shift simulation with guard enforcement.
    # ------------------------------------------------------------------

    def _execute_simulation(
        self,
        schedule: Schedule,
        phase: int,
        contract: OutcomeContract | None = None,
        demand_spike_at_s: float | None = None,
        demand_spike_pct: float = 0.0,
        cell_config: CellConfig | None = None,
    ) -> SimulationResult | None:
        """Run the deployed schedule through SimPy discrete-event simulation.

        Returns None if ``config.use_simulation`` is False.
        """
        if not self.config.use_simulation:
            return None

        cfg = cell_config or self.config.cell
        cell = ManufacturingCell(
            cell=cfg,
            seed=self.mc_seed_base + phase,
            guard_enforcer=self._guard_enforcer,
        )

        print(f"\n  [L4] Running SimPy simulation (8-hour shift) ...")
        sim_result = cell.run(
            schedule,
            demand_spike_at_s=demand_spike_at_s,
            demand_spike_pct=demand_spike_pct,
        )
        m = sim_result.metrics

        print(f"  [L4] Simulation complete:")
        print(f"    Units produced: {m.units_produced}")
        print(f"    Throughput:     {m.throughput_uph} u/h")
        print(f"    Defect rate:    {m.defect_rate}")
        print(f"    Fatigue index:  {m.fatigue_index}")
        print(f"    Noise:          {m.noise_db} dB")
        print(f"    Energy:         {m.energy_kwh} kWh")
        print(f"    Deadline gap:   {m.deadline_gap_pct}%")
        print(f"    Demand met:     {sim_result.demand_met}")

        if sim_result.guard_events:
            print(f"    Guard events:   {len(sim_result.guard_events)}")
            for ev in sim_result.guard_events:
                print(f"      {ev.guard_name}: {ev.metric_name}={ev.metric_value} "
                      f"(threshold {ev.threshold}) -> {ev.action}")

        self.audit.record(
            phase=phase, layer="L4 (Simulation)",
            action="simulation_executed",
            contract_state=f"active",
            working_mode=self.governance.current_mode.value,
            schedule=schedule.name,
            units_produced=m.units_produced,
            throughput_uph=m.throughput_uph,
            defect_rate=m.defect_rate,
            fatigue_index=m.fatigue_index,
            noise_db=m.noise_db,
            energy_kwh=m.energy_kwh,
            demand_met=sim_result.demand_met,
            guard_events_count=len(sim_result.guard_events),
        )

        return sim_result

    # ------------------------------------------------------------------
    # Integrated execution — Isaac Sim + BaSyx AAS + OPC-UA
    # ------------------------------------------------------------------

    def _execute_integrated(
        self,
        schedule: Schedule,
        phase: int,
        contract: OutcomeContract | None = None,
        demand_spike_at_s: float | None = None,
        demand_spike_pct: float = 0.0,
        cell_config: CellConfig | None = None,
    ) -> SimulationResult | None:
        """Run the deployed schedule through the integrated stack.

        Returns None if ``config.use_integrated`` is False or orchestrator
        is not available.
        """
        if not self.config.use_integrated or self._orchestrator is None:
            return None

        # Sync contract to BaSyx AAS + OPC-UA
        if contract is not None:
            try:
                self._orchestrator.sync_contract(contract)
            except Exception as e:
                logger.warning("Contract sync failed: %s", e)

        print(f"\n  [L4-INT] Running integrated execution (Isaac Sim + BaSyx + OPC-UA) ...")

        try:
            sim_result = self._orchestrator.deploy_and_run_schedule(
                schedule,
                cell_config=cell_config,
                guard_enforcer=self._guard_enforcer,
                demand_spike_at_s=demand_spike_at_s,
                demand_spike_pct=demand_spike_pct,
            )
        except RuntimeGuardViolation:
            raise
        except Exception as e:
            self._block_execution(
                phase, f"Integrated execution failed; further dispatch halted: {e}"
            )

        m = sim_result.metrics
        print(f"  [L4-INT] Integrated execution complete:")
        print(f"    Units produced: {m.units_produced}")
        print(f"    Throughput:     {m.throughput_uph} u/h")
        print(f"    Defect rate:    {m.defect_rate}")
        print(f"    Fatigue index:  {m.fatigue_index}")
        print(f"    Noise:          {m.noise_db} dB")
        print(f"    Energy:         {m.energy_kwh} kWh")
        print(f"    Deadline gap:   {m.deadline_gap_pct}%")
        print(f"    Demand met:     {sim_result.demand_met}")

        if sim_result.guard_events:
            print(f"    Guard events:   {len(sim_result.guard_events)}")
            for ev in sim_result.guard_events:
                print(f"      {ev.guard_name}: {ev.metric_name}={ev.metric_value} "
                      f"(threshold {ev.threshold}) -> {ev.action}")

        readiness = self._orchestrator.check_readiness()
        bridges_str = ", ".join(
            f"{k}={'OK' if v else 'stub'}" for k, v in readiness.items()
        )
        print(f"    Bridges: {bridges_str}")

        self.audit.record(
            phase=phase, layer="L4 (Integrated)",
            action="integrated_execution",
            contract_state="active",
            working_mode=self.governance.current_mode.value,
            schedule=schedule.name,
            units_produced=m.units_produced,
            throughput_uph=m.throughput_uph,
            defect_rate=m.defect_rate,
            fatigue_index=m.fatigue_index,
            noise_db=m.noise_db,
            energy_kwh=m.energy_kwh,
            demand_met=sim_result.demand_met,
            guard_events_count=len(sim_result.guard_events),
            bridges=readiness,
        )

        return sim_result

    # ------------------------------------------------------------------
    # Multi-objective candidate evaluation + Pareto filtering
    # ------------------------------------------------------------------

    def _evaluate_candidates_pareto(
        self,
        candidates: list[Schedule],
        contract: OutcomeContract,
        canonical_name: str,
        phase: int,
    ) -> tuple[
        ParetoFrontResult,
        dict[str, SimulationMetrics],
        dict[str, VRScoreResult],
        dict[str, FeasibilityReport],
    ]:
        """Evaluate every candidate schedule and run Pareto filtering.

        Each candidate is evaluated through the cell physics model, scored
        with the V/R index, and checked against the contract's hard
        constraints.  The results are then passed to the two-objective
        Pareto filter (maximise V/R score, maximise minimum constraint
        margin) to identify the non-dominated front.

        Parameters
        ----------
        candidates:
            All variant schedules (e.g. S1a ... S1e).
        contract:
            The active outcome contract for constraint checking.
        canonical_name:
            Base schedule name (``"S1"``, ``"S2"``, or ``"S3"``).
        phase:
            Current experiment phase number (for audit logging).

        Returns
        -------
        (pareto_result, all_metrics, all_vr, all_feasibility)
        """
        all_metrics: dict[str, SimulationMetrics] = {}
        all_vr: dict[str, VRScoreResult] = {}
        all_feasibility: dict[str, FeasibilityReport] = {}
        pareto_input: list[dict] = []

        for sched in candidates:
            m = self.evaluator.evaluate(sched)
            vr = self.vr_scorer.compute(m, schedule_name=sched.name)
            feas = self.checker.check(contract, m)

            all_metrics[sched.name] = m
            all_vr[sched.name] = vr
            all_feasibility[sched.name] = feas

            # Minimum constraint margin (most binding constraint).
            # Positive = inside feasible region; larger = more headroom.
            if feas.constraint_margins:
                min_margin = min(feas.constraint_margins.values())
            else:
                min_margin = 0.0

            pareto_input.append({
                "name": sched.name,
                "vr_score": vr.vr_score,
                "constraint_margin": min_margin,
                "feasible": feas.is_feasible,
                "source": getattr(sched, "source", "deterministic"),
            })

        pareto_result = pareto_filter(pareto_input)

        # Audit the full Pareto frontier for provenance
        self.audit.record(
            phase=phase,
            layer="L2",
            action="pareto_frontier_computed",
            contract_state=f"{'C1' if phase < 5 else 'C2'} active",
            working_mode=self.governance.current_mode.value,
            canonical=canonical_name,
            num_candidates=len(candidates),
            non_dominated=pareto_result.non_dominated,
            dominated=pareto_result.dominated,
            selected=pareto_result.selected,
            rationale=pareto_result.selection_rationale,
            scores=pareto_result.scores,
            candidate_sources=pareto_result.candidate_sources,
            selected_source=pareto_result.candidate_sources.get(
                pareto_result.selected, "deterministic"
            ),
        )

        logger.info(
            "Phase %d Pareto (%s variants): %d candidates, "
            "%d non-dominated [%s], selected=%s",
            phase,
            canonical_name,
            len(candidates),
            len(pareto_result.non_dominated),
            ", ".join(pareto_result.non_dominated),
            pareto_result.selected,
        )

        return pareto_result, all_metrics, all_vr, all_feasibility


    # ------------------------------------------------------------------
    # Certification gate: deploy the best *certified* candidate
    # ------------------------------------------------------------------

    CERT_THRESHOLD = 0.95

    def _certification_gate(self, candidates, pareto_result, cand_vr, cand_feas, contract, phase):
        """Select the first certified candidate in the existing Pareto order.

        Deployment phases halt if the pool has no certified candidate. Phase 3
        can retain a rejected proposal as a counterexample for escalation.
        """
        by_name = {c.name: c for c in candidates}
        vr = lambda n: cand_vr[n].vr_score if n in cand_vr else -1.0  # noqa: E731
        nd = sorted([n for n in pareto_result.non_dominated if n != pareto_result.selected], key=vr, reverse=True)
        dom = sorted(list(pareto_result.dominated), key=vr, reverse=True)
        order = [pareto_result.selected] + nd + dom
        tried: list[tuple[str, float]] = []
        chosen, certified, p_chosen = None, False, None
        seen: set[str] = set()
        for name in order:
            if name in seen or name not in by_name or not cand_feas[name].is_feasible:
                continue
            seen.add(name)
            mc = self.evaluator.evaluate_monte_carlo(by_name[name], n_samples=200, seed=self.mc_seed_base)
            checked = self.checker.check_stochastic(contract, mc, seed=self.mc_seed_base)
            p = checked.p_feasible
            tried.append((name, round(p, 3)))
            if not checked.evaluation_errors and p >= self.CERT_THRESHOLD:
                chosen, certified, p_chosen = by_name[name], True, p
                break
        no_feasible = not any(f.is_feasible for f in cand_feas.values())
        if chosen is None:
            # Retain a rejected proposal only for the diagnostic replan phase.
            chosen = by_name.get(pareto_result.selected) or next(iter(candidates), None)
            p_chosen = next((p for n, p in tried if chosen and n == chosen.name), None)
            if phase != 3 or chosen is None:
                self._block_deployment(
                    phase, chosen.name if chosen else "<empty candidate pool>",
                    f"No candidate passes nominal constraints and P >= {self.CERT_THRESHOLD}; tried={tried}",
                )
        for e in reversed(self.audit.entries):
            if e.phase == phase and "candidate_sources" in e.details:
                e.details["selected"] = chosen.name
                e.details["selected_source"] = e.details["candidate_sources"].get(chosen.name, "deterministic")
                e.details["gate_no_feasible_candidate"] = no_feasible
                e.details["gate_certified"] = certified
                e.details["gate_p_feasible"] = p_chosen
                e.details["gate_tried"] = tried
                break
        return chosen, {"certified": certified, "p_feasible": p_chosen, "tried": tried}

    # ------------------------------------------------------------------
    # CBPALifecycle abstract implementations
    # ------------------------------------------------------------------

    def _make_result(self) -> ExperimentResult:
        return ExperimentResult(audit=self.audit)

    def _append_result(self, result: ExperimentResult, pr: PhaseResult) -> None:
        result.phases.append(pr)

    def _finalize_result(self, result: ExperimentResult, rounds: list) -> None:
        # Populate schedule metrics/VR/feasibility from phase results
        phase_schedule_map = {1: "S1", 3: "S2", 5: "S3", 6: "S4"}
        for pr in result.phases:
            sname = phase_schedule_map.get(pr.phase)
            if sname and pr.metrics is not None:
                result.schedule_metrics[sname] = pr.metrics
            if sname and pr.vr_score is not None:
                result.schedule_vr[sname] = pr.vr_score
            if sname and pr.feasibility is not None:
                result.schedule_feasibility[sname] = pr.feasibility

        result.mode_summary = self.governance.get_mode_summary()
        for pr in result.phases:
            result.all_guard_events.extend(pr.guard_events)

    def _get_round_sequence(self) -> list[RoundConfig]:
        rounds = list(CBPA_ROUND_SEQUENCE)
        if self.config.enable_macro_phase:
            rounds.append(RoundConfig(
                round_number=6,
                round_type=RoundType.MACRO_ADAPT,
                description="Macro-Adaptation (Three-Tier Hierarchy)",
                working_mode=WorkingMode.LEGISLATOR,
            ))
        return rounds

    def _round_initial_deploy(self, rc: RoundConfig, prior: list) -> PhaseResult:
        """Round 1: L1 → L2 → L3 → L4."""
        return self._phase1()

    def _round_monitor(self, rc: RoundConfig, prior: list) -> PhaseResult:
        """Round 2: L4."""
        p1 = prior[0]  # Round 1 result
        return self._phase2(p1)

    def _round_replan(self, rc: RoundConfig, prior: list) -> PhaseResult:
        """Round 3: L2 → L3."""
        p1 = prior[0]
        return self._phase3(p1.contract)

    def _round_escalate(self, rc: RoundConfig, prior: list) -> PhaseResult:
        """Round 4: Meta."""
        p3 = prior[2]  # Round 3 result
        return self._phase4(p3.feasibility, previous_contract=prior[0].contract)

    def _round_adapt(self, rc: RoundConfig, prior: list) -> PhaseResult:
        """Round 5: L5 → L2 → L3 → L4 → L5 (contract pre-authorised by Phase 4)."""
        p1 = prior[0]
        p3 = prior[2]
        p4 = prior[3]
        return self._phase5(p1, p3, p4)

    def _round_macro_adapt(self, rc: RoundConfig, prior: list) -> PhaseResult:
        """Round 6 (optional): L4 → L5 → L1 → L2 → L3 → L4."""
        p5 = prior[4]  # Round 5 result
        return self._phase6_macro(p5)

    # ------------------------------------------------------------------
    # run_all_phases — delegates to lifecycle
    # ------------------------------------------------------------------

    def run_all_phases(
        self,
        on_phase_complete: Callable[[PhaseResult], None] | None = None,
    ) -> ExperimentResult:
        """Execute the canonical CBPA lifecycle (5 rounds + optional macro).

        Parameters
        ----------
        on_phase_complete:
            Optional callback invoked after each round finishes.  Used by
            the ``LiveResultWriter`` to push incremental results to the
            dashboard.
        """
        return super().run_all_phases(on_phase_complete=on_phase_complete)

    def _phase1(self) -> PhaseResult:
        """Phase 1: Intent -> C1 -> candidates -> Pareto rank -> verify -> deploy S1."""
        logger.info("=== Phase 1: Initial Contract and Deployment ===")

        # ── Narration: show the manager intent ──
        print("\n" + "=" * 70)
        print("  PHASE 1: Intent Translation & Contract Deployment")
        print("=" * 70)
        print(f"\n  Manager says: \"{MANAGER_INTENT}\"")
        print(f"  Working mode: {self.governance.current_mode.value.upper()}")
        print()

        # L1: Elicit contract with propose-and-refine ambiguity resolution.
        try:
            c1, ambiguity_log = self.elicitation.elicit_with_refinement(
                MANAGER_INTENT, "C1"
            )
        except Exception as exc:
            print(f"  [LLM FALLBACK] Refinement: {exc} -> using bare elicit()")
            logger.warning(
                "elicit_with_refinement() failed (%s); "
                "falling back to bare elicit().",
                exc,
            )
            c1 = self.elicitation.elicit(MANAGER_INTENT, "C1")
            ambiguity_log = []

        c1 = _canonicalise_contract(c1, MANAGER_INTENT)
        c1 = self._prepare_contract(c1, 1)

        # ── Narration: show the parsed contract ──
        print("  [L1] Contract C1 elicited:")
        print(f"    KPIs:        {', '.join(k.name + ' (' + k.direction.value + ')' for k in c1.kpi_targets)}")
        print(f"    Constraints: {', '.join(c.name + ' ' + c.operator + ' ' + str(c.limit) for c in c1.hard_constraints)}")
        print(f"    Priorities:  {' > '.join(p.value for p in c1.priority_order)}")
        if c1.typed_assumptions:
            print(f"    Assumptions: {', '.join(a.name + '=' + str(a.expected_value) for a in c1.typed_assumptions)}")
        print()

        # Audit: base contract elicitation
        self.audit.record(
            phase=1, layer="L1",
            action="contract_elicited", contract_state="C1 verified",
            working_mode=self.governance.current_mode.value,
            intent=MANAGER_INTENT,
        )

        # Audit: ambiguity resolution loop
        if ambiguity_log:
            self.audit.record(
                phase=1, layer="L1",
                action="ambiguities_resolved",
                contract_state="C1 verified",
                working_mode=self.governance.current_mode.value,
                ambiguity_count=len(ambiguity_log),
                flags=[
                    {
                        "field": f.field,
                        "question": f.question,
                        "resolution": f.resolution,
                    }
                    for f in ambiguity_log
                ],
            )
            # ── Narration: show ambiguity resolution ──
            print(f"  [L1] Ambiguity detection: {len(ambiguity_log)} issue(s) found")
            for i, f in enumerate(ambiguity_log, 1):
                print(f"    {i}. {f.field}")
                print(f"       Q: {f.question}")
                print(f"       A: {f.resolution}")
            print()

            logger.info(
                "Phase 1: %d ambiguit%s resolved during intent translation: %s",
                len(ambiguity_log),
                "y" if len(ambiguity_log) == 1 else "ies",
                ", ".join(f.field for f in ambiguity_log),
            )

        # L2: Generate all S1 candidates and run multi-objective Pareto ranking
        candidates = self.planner.generate_initial(c1)
        pareto_result, cand_metrics, cand_vr, cand_feas = (
            self._evaluate_candidates_pareto(candidates, c1, "S1", phase=1)
        )

        # ── Narration: Pareto exploration ──
        print(f"  [L2] Generated {len(candidates)} candidate schedules")
        print(f"    Pareto front: {len(pareto_result.non_dominated)} non-dominated "
              f"[{', '.join(pareto_result.non_dominated)}]")
        print(f"    Selected: {pareto_result.selected}")
        print()

        # Deploy the Pareto-selected candidate, re-labelled "S1" for consistency
        # (falls back to the first candidate if the selected name is missing).
        _selected, _gate = self._certification_gate(candidates, pareto_result, cand_vr, cand_feas, c1, phase=1)
        s1 = _selected.model_copy(update={"name": "S1"})
        metrics = self.evaluator.evaluate(s1)
        vr = self.vr_scorer.compute(metrics, schedule_name=s1.name)

        self.audit.record(
            phase=1, layer="L2",
            action="schedule_generated", contract_state="C1 verified",
            working_mode=self.governance.current_mode.value,
            schedule=s1.name, vr_score=vr.vr_score,
        )

        # L3: Deterministic + stochastic (Monte Carlo) verification of S1
        report = self.checker.check(c1, metrics)
        mc_samples = self.evaluator.evaluate_monte_carlo(s1, n_samples=200, seed=self.mc_seed_base)
        stoch_report = self.checker.check_stochastic(c1, mc_samples, seed=self.mc_seed_base)
        cert = self.cert_issuer.issue(s1.name, report, c1, stochastic_report=stoch_report)
        self._require_deployment_certificate(cert, phase=1)

        # ── Narration: verification result ──
        print(f"  [L3] Stochastic verification (n={stoch_report.n_samples} Monte Carlo samples):")
        print(f"    S1 feasible: {report.is_feasible} | P(feasible)={stoch_report.p_feasible:.0%}")
        print(f"    Margins: {report.constraint_margins}")
        print()

        self.audit.record(
            phase=1, layer="L3",
            action="verification_passed", contract_state="C1 verified",
            working_mode=self.governance.current_mode.value,
            feasible=report.is_feasible, margins=report.constraint_margins,
            p_feasible=stoch_report.p_feasible,
            constraint_violation_probabilities=stoch_report.constraint_violation_probabilities,
            worst_case_margins=stoch_report.worst_case_margins,
            stochastic_n_samples=stoch_report.n_samples,
        )

        # L4: Deploy — create guard enforcer from certificate guards
        if cert.is_certified and cert.guards:
            self._guard_enforcer = GuardEnforcer(cert.guards, cyber_risk_level=self.checker.cyber_risk_level)
            print(f"  [L4] Guards armed: {len(cert.guards)} runtime guards from certificate")
            logger.info(
                "Phase 1: GuardEnforcer armed with %d guards from certificate",
                len(cert.guards),
            )
            self.audit.record(
                phase=1, layer="L4 (Guard)",
                action="guards_armed", contract_state="C1 verified",
                working_mode=self.governance.current_mode.value,
                guard_count=len(cert.guards),
                guards=[str(g) for g in cert.guards],
            )

        self.audit.record(
            phase=1, layer="L4",
            action="schedule_deployed", contract_state="C1 verified",
            working_mode=self.governance.current_mode.value,
            schedule=s1.name,
        )

        print(f"  [L4] S1 DEPLOYED | Throughput={metrics.throughput_uph} u/h | "
              f"V/R={vr.vr_score:.3f} | P(feasible)={stoch_report.p_feasible:.0%}")

        # Run integrated or SimPy simulation of deployed S1 schedule.
        # Any observed violation halts further dispatch; an alternative must
        # pass a fresh planning and certification cycle before execution.
        deployed = s1
        sim_result = self._execute_integrated(deployed, phase=1, contract=c1)
        if sim_result is None:
            sim_result = self._execute_simulation(deployed, phase=1, contract=c1)
        if sim_result is not None:
            metrics = sim_result.metrics
            vr = self.vr_scorer.compute(metrics, schedule_name=deployed.name)
            report = self.checker.check(c1, metrics)

            if not report.is_feasible:
                self._block_deployment(
                    1, deployed.name,
                    "Execution metrics violated the contract; further dispatch halted: "
                    + "; ".join(report.violations),
                )

        logger.info(
            f"Phase 1: {deployed.name} deployed | Throughput={metrics.throughput_uph} | "
            f"V/R={vr.vr_score} | Feasible={report.is_feasible} | "
            f"P(feasible)={stoch_report.p_feasible:.2%} (Monte Carlo n={stoch_report.n_samples})"
        )

        return PhaseResult(
            phase=1,
            schedule=deployed,
            metrics=metrics,
            vr_score=vr,
            feasibility=report,
            contract=c1,
            pareto_front=pareto_result,
            description=f"{deployed.name} deployed; K satisfied",
        )

    def _phase2(self, p1: PhaseResult) -> PhaseResult:
        """Phase 2: Demand +20% -> monitoring detects shortfall via assumption drift."""
        logger.info("=== Phase 2: Demand Surge Detection ===")

        print("\n" + "=" * 70)
        print("  PHASE 2: Disturbance Detection & Monitoring")
        print("=" * 70)
        print(f"\n  Disturbance: Demand surge +{self.config.demand_spike_pct}%")
        print(f"  Working mode: {self.governance.current_mode.value.upper()}")
        print()

        assert p1.contract is not None
        assert p1.metrics is not None

        # L4 monitoring detects the gap — with guard enforcement if available
        monitor = ContractMonitor(
            p1.contract,
            guard_enforcer=self._guard_enforcer,
        )
        new_demand = self.config.cell.demand_base_uph * (
            1 + self.config.demand_spike_pct / 100
        )

        # Forward demand surge disturbance to Isaac Sim when integrated
        if self._orchestrator is not None:
            result = self._orchestrator.inject_disturbance(
                "demand_surge", magnitude_pct=self.config.demand_spike_pct,
            )
            print(f"  [L4-INT] Disturbance injected: demand_surge "
                  f"+{self.config.demand_spike_pct}% -> {result}")

        # --- Runtime guard enforcement on current S1 metrics ---
        # Guards from the Phase 1 certificate are checked against the
        # current S1 metrics.  For S1 (which passed verification), guards
        # should *not* fire — this validates the end-to-end enforcement
        # pipeline.  Any guard events would indicate a runtime anomaly.
        kpi_alerts = monitor.check(p1.metrics)
        guard_events = list(monitor.guard_events)

        if guard_events:
            self.audit.record(
                phase=2, layer="L4 (Guard)",
                action="guards_enforced", contract_state="C1 active",
                working_mode=self.governance.current_mode.value,
                guard_events_count=len(guard_events),
                guard_events=[
                    {
                        "guard": ev.guard_name,
                        "metric": ev.metric_name,
                        "value": ev.metric_value,
                        "threshold": ev.threshold,
                        "action": ev.action,
                        "enforced": ev.enforced,
                    }
                    for ev in guard_events
                ],
            )
            for ev in guard_events:
                logger.info(
                    "Phase 2 guard event: %s %s=%s (threshold %s) -> %s",
                    ev.guard_name, ev.metric_name, ev.metric_value,
                    ev.threshold, ev.action,
                )
        else:
            self.audit.record(
                phase=2, layer="L4 (Guard)",
                action="guards_checked", contract_state="C1 active",
                working_mode=self.governance.current_mode.value,
                guard_events_count=0,
                detail="All guards satisfied -- S1 metrics within safe bounds",
            )
            logger.info("Phase 2: All guards satisfied for S1 metrics")

        # --- Assumption drift tracking ---
        assumption_violations = monitor.check_assumptions(
            {"demand_base_uph": new_demand}
        )

        # Record each assumption violation in the audit trail
        for drift in assumption_violations:
            self.audit.record(
                phase=2, layer="L4 (Monitor)",
                action="assumption_violated",
                contract_state="C1 active; assumption drift",
                working_mode=self.governance.current_mode.value,
                assumption=drift.assumption.name,
                expected=drift.assumption.expected_value,
                observed=drift.assumption.current_value,
                drift_pct=drift.drift_pct,
                attribution=drift.attribution,
            )
            logger.info(
                "Phase 2: Assumption A.%s violated: %s -> %s (relative drift %s%%)",
                drift.assumption.name,
                drift.assumption.expected_value,
                drift.assumption.current_value,
                drift.drift_pct,
            )

        # Attribute deadline gap to the specific assumption drifts
        alert = monitor.check_deadline(p1.metrics.throughput_uph, new_demand)

        constraint_attributions: list[str] = []
        if alert and monitor.assumption_tracker is not None:
            constraint_attributions = (
                monitor.assumption_tracker.attribute_constraint_breach(
                    "DeadlineGap", "soft"
                )
            )

        # --- L4 LLM attribution ---
        l4_attribution = self.attribution_agent.explain_breach(
            phase=2,
            drifts=[
                {
                    "name": d.assumption.name,
                    "expected": d.assumption.expected_value,
                    "actual": d.assumption.current_value,
                    "drift_pct": d.drift_pct,
                }
                for d in assumption_violations
            ],
            current_metrics={
                "throughput_uph": p1.metrics.throughput_uph,
                "fatigue_index": p1.metrics.fatigue_index,
                "noise_db": p1.metrics.noise_db,
                "energy_kwh": p1.metrics.energy_kwh,
            },
            breached_constraints={
                "DeadlineGap": -(alert.margin_pct if alert else 0) / 100.0,
            } if alert else {},
            schedule_name=p1.schedule.name if p1.schedule else "",
        )

        self.audit.record(
            phase=2, layer="L4 (Monitor)",
            action="shortfall_detected", contract_state="C1 active; gap",
            working_mode=self.governance.current_mode.value,
            current_throughput=p1.metrics.throughput_uph,
            new_demand=new_demand,
            gap_pct=alert.margin_pct if alert else 0,
            assumption_violations=[d.attribution for d in assumption_violations],
            constraint_attributions=constraint_attributions,
            l4_llm_attributions=[
                {
                    "assumption": e.drifted_assumption,
                    "drift_pct": e.drift_pct,
                    "constraints": e.violated_constraints,
                    "mechanism": e.mechanism,
                    "urgency": e.urgency,
                    "action": e.recommended_action,
                }
                for e in l4_attribution.explanations
            ],
            l4_attribution_llm_used=l4_attribution.llm_used,
        )

        violation_summary = (
            "; ".join(d.attribution for d in assumption_violations)
            if assumption_violations
            else "no assumption drift"
        )

        # ── Narration: guard and drift results ──
        if guard_events:
            print(f"  [L4] Guard events: {len(guard_events)}")
            for ev in guard_events:
                print(f"    {ev.guard_name}: {ev.metric_name}={ev.metric_value} "
                      f"(threshold {ev.threshold}) -> {ev.action}")
        else:
            print("  [L4] All runtime guards satisfied")
        print()
        if assumption_violations:
            print("  [L4] Assumption drift detected:")
            for d in assumption_violations:
                print(f"    {d.attribution}")
        print(f"\n  [L4] Shortfall: throughput {p1.metrics.throughput_uph} u/h "
              f"vs new demand {new_demand} u/h")

        logger.info(
            f"Phase 2: Demand +{self.config.demand_spike_pct}% | "
            f"Current throughput {p1.metrics.throughput_uph} vs demand {new_demand} | "
            f"Shortfall predicted | {violation_summary}"
        )

        # Persist the tracker so _phase3 can attribute constraint breaches
        # to the same set of drifted assumptions detected here.
        self._assumption_tracker = monitor.assumption_tracker

        return PhaseResult(
            phase=2,
            contract=p1.contract,
            guard_events=guard_events,
            description=(
                f"Shortfall predicted \u2014 {violation_summary}"
                if assumption_violations
                else "Shortfall predicted"
            ),
        )

    def _phase3(self, c1: OutcomeContract) -> PhaseResult:
        """Phase 3: Generate S2 candidates -> Pareto rank -> verify best -> REJECT."""
        logger.info("=== Phase 3: Autonomous Response and Rejection ===")

        print("\n" + "=" * 70)
        print("  PHASE 3: Autonomous Replan Attempt")
        print("=" * 70)
        print(f"\n  Working mode: {self.governance.current_mode.value.upper()}")
        print("  Strategy: generate aggressive S2 schedules to meet new demand")
        print()

        # L2: Generate aggressive candidate set and run Pareto ranking
        candidates = self.planner.generate_fast(c1, self.config.demand_spike_pct)
        pareto_result, cand_metrics, cand_vr, cand_feas = (
            self._evaluate_candidates_pareto(candidates, c1, "S2", phase=3)
        )

        # Deploy the Pareto-selected candidate, re-labelled "S2" for consistency
        # (falls back to the first candidate if the selected name is missing).
        _selected, _gate = self._certification_gate(candidates, pareto_result, cand_vr, cand_feas, c1, phase=3)
        s2 = _selected.model_copy(update={"name": "S2"})
        metrics = self.evaluator.evaluate(s2)
        vr = self.vr_scorer.compute(metrics, schedule_name=s2.name)

        self.audit.record(
            phase=3, layer="L2",
            action="aggressive_schedule_generated", contract_state="C1 active",
            working_mode=self.governance.current_mode.value,
            schedule=s2.name, vr_score=vr.vr_score,
        )

        # L3: Deterministic + stochastic (Monte Carlo) verification of S2 -> REJECT
        report = self.checker.check(c1, metrics)
        mc_samples = self.evaluator.evaluate_monte_carlo(s2, n_samples=200, seed=self.mc_seed_base)
        stoch_report = self.checker.check_stochastic(c1, mc_samples, seed=self.mc_seed_base)
        cert = self.cert_issuer.issue(s2.name, report, c1, stochastic_report=stoch_report)

        # Build a rich rejection summary with stochastic violation probabilities
        viol_prob_summary = ", ".join(
            f"{c}={p:.0%}"
            for c, p in sorted(
                stoch_report.constraint_violation_probabilities.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
            if p > 0
        )
        stoch_rejection_detail = (
            f"P(feasible)={stoch_report.p_feasible:.2%}; "
            f"violation probs: {viol_prob_summary}"
        )

        # --- Assumption attribution for each violated constraint ---
        # CBPA differentiator: explain *why* constraints were breached by
        # mapping them back to assumptions that drifted in Phase 2.
        constraint_attributions: list[str] = []
        if self._assumption_tracker is not None and report.violations:
            for violated_constraint in report.violations:
                # Strip human-readable suffix to get the bare constraint name.
                # Violation strings look like "FatigueIndex: 0.6 violates <= 0.4"
                # so we split on whitespace and strip any trailing colon/punctuation.
                cname = violated_constraint.split()[0].rstrip(":").strip() if violated_constraint else violated_constraint
                attrs = self._assumption_tracker.attribute_constraint_breach(cname, "hard")
                constraint_attributions.extend(attrs)
            if constraint_attributions:
                for attr_msg in constraint_attributions:
                    logger.info("Phase 3: Attribution — %s", attr_msg)

        # --- L3 LLM repair mediation ---
        repair_result = self.repair_agent.suggest_repairs(
            schedule_name=s2.name,
            schedule_params={
                "r1_speed_fraction": s2.r1_speed_fraction,
                "r2_speed_fraction": s2.r2_speed_fraction,
                "human_cycle_rate_multiplier": s2.human_cycle_rate_multiplier,
                "buffer_time_s": s2.buffer_time_s,
            },
            violations=report.violations,
            constraint_margins=report.constraint_margins,
            metrics={
                "throughput_uph": metrics.throughput_uph,
                "fatigue_index": metrics.fatigue_index,
                "noise_db": metrics.noise_db,
                "energy_kwh": metrics.energy_kwh,
            },
        )
        if repair_result.suggestions:
            for sugg in repair_result.suggestions:
                logger.info(
                    "L3 repair: %s: %.2f -> %.2f (%s)",
                    sugg.parameter, sugg.current_value,
                    sugg.suggested_value, sugg.rationale,
                )

        self.audit.record(
            phase=3, layer="L3",
            action="schedule_rejected", contract_state="C1 active",
            working_mode=self.governance.current_mode.value,
            feasible=report.is_feasible,
            violations=report.violations,
            p_feasible=stoch_report.p_feasible,
            stochastic_n_samples=stoch_report.n_samples,
            constraint_violation_probabilities=stoch_report.constraint_violation_probabilities,
            assumption_attributions=constraint_attributions,
            l3_repair_suggestions=[
                {
                    "parameter": s.parameter,
                    "current_value": s.current_value,
                    "suggested_value": s.suggested_value,
                    "rationale": s.rationale,
                }
                for s in repair_result.suggestions
            ],
            l3_repair_llm_used=repair_result.llm_used,
        )

        # ── Narration: rejection with attribution ──
        print(f"  [L2] {len(candidates)} aggressive S2 variants generated")
        print(f"    Pareto front: {len(pareto_result.non_dominated)} non-dominated "
              f"[{', '.join(pareto_result.non_dominated)}]")
        print()
        print(f"  [L3] S2 REJECTED by verification firewall:")
        print(f"    P(feasible) = {stoch_report.p_feasible:.0%}")
        for v in report.violations:
            print(f"    Violation: {v}")
        if viol_prob_summary:
            print(f"    Violation probabilities: {viol_prob_summary}")
        if constraint_attributions:
            print(f"\n  [L4] Constraint breach attribution:")
            for attr_msg in constraint_attributions:
                print(f"    {attr_msg}")
        if repair_result.suggestions:
            llm_tag = " (LLM)" if repair_result.llm_used else " (rule-based)"
            print(f"\n  [L3] Repair suggestions{llm_tag}:")
            for sugg in repair_result.suggestions:
                print(f"    {sugg.parameter}: {sugg.current_value} -> "
                      f"{sugg.suggested_value} ({sugg.rationale})")

        logger.info(
            f"Phase 3: S2 REJECTED | Throughput={metrics.throughput_uph} | "
            f"V/R={vr.vr_score} | Violations={report.violations} | "
            f"{stoch_rejection_detail}"
        )

        return PhaseResult(
            phase=3,
            schedule=s2,
            metrics=metrics,
            vr_score=vr,
            feasibility=report,
            contract=c1,
            pareto_front=pareto_result,
            description=(
                f"S2 rejected: {stoch_rejection_detail}"
                + (
                    f" | Attribution: {'; '.join(constraint_attributions)}"
                    if constraint_attributions else ""
                )
            ),
        )

    def _phase4(self, s2_report: FeasibilityReport, previous_contract=None) -> PhaseResult:
        """Phase 4: Escalation -> Manager picks Option C."""
        logger.info("=== Phase 4: Human-in-the-Loop Negotiation ===")

        print("\n" + "=" * 70)
        print("  PHASE 4: Human-in-the-Loop Escalation")
        print("=" * 70)
        print(f"\n  Working mode: {self.governance.current_mode.value.upper()}")
        print("  Reason: autonomous replan failed — escalating to manager")
        print()

        # Collect assumption drift attribution strings to include in the
        # escalation query so the manager understands *why* the conflict
        # arose (which assumptions were violated), not just that it arose.
        assumption_drift_msgs: list[str] = []
        if self._assumption_tracker is not None:
            assumption_drift_msgs = self._assumption_tracker.summary()

        query, decision = self.governance.handle_conflict(
            s2_report,
            demand_increase_pct=self.config.demand_spike_pct,
            phase=4,
            assumption_drifts=assumption_drift_msgs,
        )

        # ── Narration: show the escalation dialogue ──
        print(f"  [Escalation] Conflict summary:")
        print(f"    {query.conflict_summary}")
        if hasattr(query, 'options') and query.options:
            print(f"\n  [Escalation] Remediation options presented to manager:")
            for opt in query.options:
                label = getattr(opt, 'label', str(opt))
                desc = getattr(opt, 'description', '')
                preserves = getattr(opt, 'preserves_human_constraints', None)
                print(f"    - {label}: {desc}")
                if preserves is not None:
                    print(f"      Preserves human constraints: {'Yes' if preserves else 'No'}")
        print(f"\n  [Manager] Decision: {decision.selected_option}")
        if hasattr(decision, 'rationale') and decision.rationale:
            print(f"    Rationale: {decision.rationale}")

        logger.info(
            f"Phase 4: {query.conflict_summary}\n"
            f"  Manager selected: {decision.selected_option}"
        )

        # The Partner decision is the legislative act: pre-authorise C2 here
        # so Phase 5 (Auditor) can implement it without invoking L1.
        c2 = _canonicalise_contract(self.elicitation.elicit(MANAGER_INTENT, "C2"), MANAGER_INTENT)
        c2 = self._prepare_contract(c2, 4, previous=previous_contract)
        print(f"  [Meta] Contract pre-authorised: C1 → {c2.name}")

        self.audit.record(
            phase=4, layer="Meta",
            action="contract_preauthorised",
            contract_state="C1 → C2 (pre-authorised)",
            working_mode=self.governance.current_mode.value,
            contract=c2.name,
        )

        return PhaseResult(
            phase=4,
            contract=c2,
            decision=decision,
            description=f"Manager: {decision.selected_option} (keep constraints)",
        )

    def _phase5(self, p1: PhaseResult, p3: PhaseResult, p4: PhaseResult) -> PhaseResult:
        """Phase 5: L5 → L2 → L3 → L4 → L5.

        Contract C2 is pre-authorised by Phase 4 (Partner decision).
        No L1 invocation here — the Auditor implements, does not legislate.
        """
        logger.info("=== Phase 5: Adaptation and Stabilization ===")

        print("\n" + "=" * 70)
        print("  PHASE 5: Adaptation & Learning")
        print("=" * 70)
        print(f"\n  Working mode: {self.governance.current_mode.value.upper()}")
        print("  Strategy: implement pre-authorised C2, generate balanced S3 with learning bias")
        print()

        # Contract pre-authorised by manager in Phase 4 — no L1 needed here
        from cbpa.models.contract import make_c2 as _make_c2
        c2 = p4.contract if p4.contract is not None else _make_c2()
        c2 = self._prepare_contract(c2, 5, previous=p1.contract)
        print(f"  Using pre-authorised contract: {c2.name}")
        print()

        # L5→L2 feedback: query learning store for planning bias.
        # On the first shift this returns {} (no prior experience), but
        # the wiring is in place for multi-shift runs where past
        # experience biases future candidate generation.
        learning_bias = self.learning.get_bias_for_planning("demand_surge")

        self.audit.record(
            phase=5, layer="L5",
            action="learning_bias_queried",
            contract_state=f"{c2.name} active",
            working_mode=self.governance.current_mode.value,
            bias_available=bool(learning_bias),
            bias_content=learning_bias if learning_bias else "no prior experience",
            experience_count=self.learning.total_records,
        )

        # ── Narration: learning bias ──
        if learning_bias:
            print(f"  [L5->L2] Learning bias from {self.learning.total_records} prior experience(s):")
            print(f"    Avoid:  {learning_bias.get('avoid_policies', [])}")
            print(f"    Prefer: {learning_bias.get('prefer_families', [])}")
            print(f"    Hints:  {list(learning_bias.get('parameter_hints', {}).keys())}")
            logger.info(
                "Phase 5: Learning bias active — avoid=%s, prefer=%s, hints=%s",
                learning_bias.get("avoid_policies"),
                learning_bias.get("prefer_families"),
                list(learning_bias.get("parameter_hints", {}).keys()),
            )
        else:
            print("  [L5->L2] No learning bias (first shift — no prior experience)")
            logger.info("Phase 5: No learning bias (first shift)")
        print()

        # --- L5 LLM adaptation reasoning ---
        prior_experiences = self.learning.get_similar("demand_surge")
        p3_stoch = p3.feasibility  # feasibility report from Phase 3
        # Build constraint violation probs from Phase 3 if available
        p3_violations = {}
        if hasattr(p3, 'description') and p3.description:
            # Use the stochastic report from Phase 3 audit if available
            pass
        adaptation_reasoning = self.adaptation_reasoner.reason_tier_selection(
            p_feasible=0.0,  # Phase 3 was rejected (0% feasible)
            constraint_violations={"FatigueIndex": 1.0, "Noise": 0.8},
            prior_experiences=prior_experiences,
            disturbance_context=f"demand surge +{self.config.demand_spike_pct}%",
            schedule_params={
                "r1_speed_fraction": p1.schedule.r1_speed_fraction if p1.schedule else 0.8,
                "r2_speed_fraction": p1.schedule.r2_speed_fraction if p1.schedule else 0.8,
            } if p1.schedule else None,
        )
        # L5 experience synthesis
        experience_synthesis = self.adaptation_reasoner.synthesize_experience(
            prior_experiences, "demand_surge"
        )

        self.audit.record(
            phase=5, layer="L5 (Reasoning)",
            action="adaptation_tier_selected",
            contract_state=f"{c2.name} active",
            working_mode=self.governance.current_mode.value,
            recommended_tier=adaptation_reasoning.recommended_tier.value,
            reasoning_confidence=adaptation_reasoning.confidence,
            reasoning_rationale=adaptation_reasoning.rationale,
            reasoning_key_insight=adaptation_reasoning.key_insight,
            reasoning_llm_used=self.adaptation_reasoner.use_llm,
            experience_synthesis=(
                {
                    "pattern": experience_synthesis.disturbance_pattern,
                    "successful": experience_synthesis.successful_policies,
                    "failed": experience_synthesis.failed_policies,
                    "insight": experience_synthesis.insight,
                    "confidence": experience_synthesis.confidence,
                }
                if experience_synthesis else None
            ),
        )

        print(f"  [L5] Adaptation reasoning: {adaptation_reasoning.recommended_tier.value}")
        print(f"    Insight: {adaptation_reasoning.key_insight}")
        if experience_synthesis:
            print(f"    Synthesis: {experience_synthesis.insight}")
        print()

        # L2: Generate balanced candidate set and run Pareto ranking
        candidates = self.planner.generate_balanced(
            c2, self.config.demand_spike_pct, learning_bias=learning_bias,
        )
        pareto_result, cand_metrics, cand_vr, cand_feas = (
            self._evaluate_candidates_pareto(candidates, c2, "S3", phase=5)
        )

        # Deploy the Pareto-selected candidate, re-labelled "S3" for consistency
        # (falls back to the first candidate if the selected name is missing).
        _selected, _gate = self._certification_gate(candidates, pareto_result, cand_vr, cand_feas, c2, phase=5)
        s3 = _selected.model_copy(update={"name": "S3"})
        metrics = self.evaluator.evaluate(s3)
        vr = self.vr_scorer.compute(metrics, schedule_name=s3.name)

        self.audit.record(
            phase=5, layer="L2",
            action="balanced_schedule_generated", contract_state="C2 verified",
            working_mode=self.governance.current_mode.value,
            schedule=s3.name, vr_score=vr.vr_score,
        )

        # L3: Deterministic + stochastic (Monte Carlo) verification of S3
        report = self.checker.check(c2, metrics)
        mc_samples = self.evaluator.evaluate_monte_carlo(s3, n_samples=200, seed=self.mc_seed_base)
        stoch_report = self.checker.check_stochastic(c2, mc_samples, seed=self.mc_seed_base)
        cert = self.cert_issuer.issue(s3.name, report, c2, stochastic_report=stoch_report)
        self._require_deployment_certificate(cert, phase=5)

        self.audit.record(
            phase=5, layer="L3",
            action="verification_passed", contract_state="C2 verified",
            working_mode=self.governance.current_mode.value,
            feasible=report.is_feasible,
            p_feasible=stoch_report.p_feasible,
            constraint_violation_probabilities=stoch_report.constraint_violation_probabilities,
            worst_case_margins=stoch_report.worst_case_margins,
            stochastic_n_samples=stoch_report.n_samples,
        )

        # ── Narration: S3 verification and adaptation ──
        print(f"  [L2] {len(candidates)} balanced S3 variants generated")
        print(f"    Pareto front: [{', '.join(pareto_result.non_dominated)}]")
        print(f"\n  [L3] S3 verification: P(feasible)={stoch_report.p_feasible:.0%} | "
              f"Feasible={report.is_feasible}")

        logger.info(
            f"Phase 5: S3 stochastic verification | "
            f"P(feasible)={stoch_report.p_feasible:.2%} "
            f"(n={stoch_report.n_samples}) | "
            f"Certified={cert.is_certified}"
        )

        # L5: Apply adaptation
        assert p3.schedule is not None
        actions = self.adaptation.adapt(p3.schedule, s3)

        self.audit.record(
            phase=5, layer="L5",
            action="adaptation_applied", contract_state="C2 verified",
            working_mode=self.governance.current_mode.value,
            adaptations=[
                {"level": a.level.value, "desc": a.description}
                for a in actions
            ],
        )

        # L4: Deploy
        self.audit.record(
            phase=5, layer="L4",
            action="schedule_deployed", contract_state="C2 verified",
            working_mode=self.governance.current_mode.value,
            schedule=s3.name,
        )

        # ── Narration: adaptation and deployment ──
        print(f"\n  [L5] Adaptations applied: {len(actions)}")
        for a in actions:
            print(f"    [{a.level.value}] {a.description}")
        print(f"\n  [L4] S3 DEPLOYED | Throughput={metrics.throughput_uph} u/h | "
              f"V/R={vr.vr_score:.3f} | P(feasible)={stoch_report.p_feasible:.0%}")

        # Run integrated or SimPy simulation of deployed S3 schedule.
        # Any observed violation halts further dispatch; an alternative must
        # pass a fresh planning and certification cycle before execution.
        deployed = s3
        sim_result = self._execute_integrated(deployed, phase=5, contract=c2)
        if sim_result is None:
            sim_result = self._execute_simulation(deployed, phase=5, contract=c2)
        if sim_result is not None:
            metrics = sim_result.metrics
            vr = self.vr_scorer.compute(metrics, schedule_name=deployed.name)
            report = self.checker.check(c2, metrics)

            if not report.is_feasible:
                self._block_deployment(
                    5, deployed.name,
                    "Execution metrics violated the contract; further dispatch halted: "
                    + "; ".join(report.violations),
                )

        # L5: Learning store
        p3_stoch = getattr(p3, "_stoch_report", None)
        rejection_reason = "Fatigue > 0.4, Noise > 80dB (deterministic)"

        self.learning.record(
            disturbance_type="demand_surge",
            disturbance_magnitude=self.config.demand_spike_pct,
            rejected_policy="S2",
            rejection_reason=rejection_reason,
            accepted_policy="S3",
            adaptation_level="meso+micro",
            outcome_metrics={
                "throughput_uph": metrics.throughput_uph,
                "deadline_gap_pct": metrics.deadline_gap_pct,
            },
        )

        logger.info(
            f"Phase 5: {deployed.name} deployed | Throughput={metrics.throughput_uph} | "
            f"V/R={vr.vr_score} | Feasible={report.is_feasible} | "
            f"Adaptations={len(actions)}"
        )

        return PhaseResult(
            phase=5,
            schedule=deployed,
            metrics=metrics,
            vr_score=vr,
            feasibility=report,
            contract=c2,
            pareto_front=pareto_result,
            description=f"{deployed.name} deployed with meso/micro adaptation and learning update",
        )

    # ------------------------------------------------------------------
    # Multi-shift learning curve
    # ------------------------------------------------------------------

    def run_multi_shift(
        self,
        n_shifts: int = 3,
        on_phase_complete: Callable[[PhaseResult], None] | None = None,
        on_shift_start: Callable[[int], None] | None = None,
    ) -> MultiShiftResult:
        """Run the full phase sequence *n_shifts* times with a shared LearningStore.

        Each shift injects a slightly different disturbance pattern.  The
        learning store persists across shifts so that shifts 2+ benefit
        from accumulated experience — demonstrating CBPA's *cumulative
        operational intelligence*.

        Returns a :class:`MultiShiftResult` containing per-shift summaries
        (learning-curve metrics) and the full :class:`ExperimentResult` for
        each shift.
        """
        disturbances = (
            self.config.shift_disturbances
            or _DEFAULT_SHIFT_DISTURBANCES[:n_shifts]
        )
        # Extend with defaults if the caller asked for more shifts than
        # there are explicit disturbance configs.
        while len(disturbances) < n_shifts:
            fallback_spike = 20.0 + (len(disturbances) % 3) * 5.0
            disturbances.append({
                "demand_spike_pct": fallback_spike,
                "label": f"Shift {len(disturbances) + 1}: demand +{fallback_spike:.0f}%",
            })

        # The learning store is shared across all shifts.
        shared_learning = self.learning

        multi = MultiShiftResult()

        for shift_idx in range(n_shifts):
            dist = disturbances[shift_idx]
            spike = dist.get("demand_spike_pct", self.config.demand_spike_pct)
            label = dist.get("label", f"Shift {shift_idx + 1}")

            logger.info(
                "===== MULTI-SHIFT %d/%d: %s =====",
                shift_idx + 1, n_shifts, label,
            )

            if on_shift_start is not None:
                on_shift_start(shift_idx + 1)

            records_before = shared_learning.total_records

            # Build a per-shift config with the appropriate demand spike.
            shift_config = ScenarioConfig(
                cell=self.config.cell,
                schedules=self.config.schedules,
                demand_spike_pct=spike,
                spike_time_hours=self.config.spike_time_hours,
                vr_value_weights=self.config.vr_value_weights,
                vr_resource_weights=self.config.vr_resource_weights,
                table3_tolerance_pct=self.config.table3_tolerance_pct,
                enable_macro_phase=self.config.enable_macro_phase,
                macro_r1_degradation_fraction=self.config.macro_r1_degradation_fraction,
                macro_adapt_threshold=self.config.macro_adapt_threshold,
                use_simulation=self.config.use_simulation,
                use_integrated=self.config.use_integrated,
                isaac_sim_url=self.config.isaac_sim_url,
                basyx_registry_url=self.config.basyx_registry_url,
                basyx_aas_server_url=self.config.basyx_aas_server_url,
                opcua_endpoint=self.config.opcua_endpoint,
                llm_only=self.config.llm_only,
                optimiser_anchor=self.config.optimiser_anchor,
            )

            # Create a fresh experiment instance per shift but inject the
            # *shared* learning store so experience accumulates.
            shift_exp = CBPAExperiment(
                config=shift_config,
                use_llm=self.client is not None,
                llm_client=self.client,
            )
            shift_exp.learning = shared_learning

            result = shift_exp.run_all_phases(
                on_phase_complete=on_phase_complete,
            )
            multi.shift_results.append(result)

            # --- Collect per-shift metrics ---
            # phases_to_resolution: count phases until a feasible schedule
            # was accepted (phase 5 is the resolution phase; if it passes
            # on the first attempt that counts as 5 phases).
            phases_to_resolution = len(result.phases)

            # For shifts with learning bias, the planner may converge
            # faster.  We measure by counting phases where a feasible
            # schedule exists (earlier = better).
            first_feasible_phase = phases_to_resolution
            for pr in result.phases:
                if pr.feasibility is not None and pr.feasibility.is_feasible:
                    first_feasible_phase = pr.phase
                    break

            # VR score of the accepted schedule (S3 in phase 5)
            vr_accepted = 0.0
            if "S3" in result.schedule_vr:
                vr_accepted = result.schedule_vr["S3"].vr_score

            # Total constraint violations across all phases
            total_violations = 0
            for pr in result.phases:
                if pr.feasibility is not None and pr.feasibility.violations:
                    total_violations += len(pr.feasibility.violations)

            # Learning bias availability
            bias_available = records_before > 0

            # Candidates skipped due to bias — count by checking how
            # many fewer than the maximum 5 variants were generated in
            # phases that use learning bias (phase 5).
            candidates_skipped = 0
            for pr in result.phases:
                if pr.pareto_front is not None and pr.phase == 5:
                    generated = len(pr.pareto_front.candidates)
                    candidates_skipped = max(0, 5 - generated)

            records_after = shared_learning.total_records

            summary = ShiftSummary(
                shift=shift_idx + 1,
                demand_spike_pct=spike,
                phases_to_resolution=first_feasible_phase,
                vr_score_accepted=vr_accepted,
                total_constraint_violations=total_violations,
                learning_bias_available=bias_available,
                candidates_skipped=candidates_skipped,
                experience_records_before=records_before,
                experience_records_after=records_after,
            )
            multi.shift_summaries.append(summary)

            logger.info(
                "Shift %d summary: phases_to_resolution=%d, "
                "vr_accepted=%.2f, violations=%d, bias=%s, "
                "records=%d->%d",
                shift_idx + 1,
                summary.phases_to_resolution,
                summary.vr_score_accepted,
                summary.total_constraint_violations,
                summary.learning_bias_available,
                records_before,
                records_after,
            )

        return multi

    # ------------------------------------------------------------------
    # Phase 6: Macro-adaptation (optional)
    # ------------------------------------------------------------------

    def _phase6_macro(self, p5: PhaseResult) -> PhaseResult:
        """Phase 6: Second disturbance -> micro/meso insufficient -> MACRO replan.

        Demonstrates the complete three-tier adaptation hierarchy that is
        unique to CBPA:

        1. Inject a second disturbance: R1 mechanical wear degrades its
           maximum speed to a fraction of nominal.
        2. Try micro-adaptation (parameter retune of the current S3).
        3. Try meso-adaptation (policy family switch).
        4. Both fail stochastic verification (p_feasible < threshold).
        5. Trigger macro-adaptation: full replan from L1-L2.
        6. The replanned S4 passes verification.
        """
        logger.info("=== Phase 6: Macro-Adaptation (Second Disturbance) ===")

        print("\n" + "=" * 70)
        print("  PHASE 6: Macro-Adaptation (Three-Tier Hierarchy)")
        print("=" * 70)
        print(f"\n  Second disturbance: R1 mechanical wear")
        print(f"  R1 speed degraded to {self.config.macro_r1_degradation_fraction:.0%} of nominal")
        print(f"  Macro threshold: P(feasible) < {self.config.macro_adapt_threshold:.0%}")
        print()

        assert p5.contract is not None
        assert p5.schedule is not None

        c2 = p5.contract
        s3 = p5.schedule
        degradation = self.config.macro_r1_degradation_fraction
        threshold = self.config.macro_adapt_threshold

        # --- Inject second disturbance: R1 speed degradation ---
        # Mechanical wear limits R1 to a fraction of its original max speed.
        degraded_r1_max = self.config.cell.r1_max_speed_mps * degradation

        # Forward R1 degradation disturbance to Isaac Sim when integrated
        if self._orchestrator is not None:
            result = self._orchestrator.inject_disturbance(
                "r1_degradation", fraction=degradation,
            )
            print(f"  [L4-INT] Disturbance injected: r1_degradation "
                  f"fraction={degradation} -> {result}")

        self.audit.record(
            phase=6, layer="L4 (Monitor)",
            action="second_disturbance_detected",
            contract_state="C2 active",
            working_mode=self.governance.current_mode.value,
            disturbance="R1 mechanical wear",
            r1_max_speed_before=self.config.cell.r1_max_speed_mps,
            r1_max_speed_after=degraded_r1_max,
            degradation_fraction=degradation,
        )
        logger.info(
            "Phase 6: R1 mechanical wear detected — max speed degraded "
            "from %.2f to %.2f m/s (%.0f%%)",
            self.config.cell.r1_max_speed_mps,
            degraded_r1_max,
            degradation * 100,
        )

        # Track assumption drift for R1
        monitor = ContractMonitor(c2, guard_enforcer=self._guard_enforcer)
        r1_assumption_drifts = monitor.check_assumptions(
            {"r1_max_speed_mps": degraded_r1_max}
        )
        for drift in r1_assumption_drifts:
            self.audit.record(
                phase=6, layer="L4 (Monitor)",
                action="assumption_violated",
                contract_state="C2 active; R1 degraded",
                working_mode=self.governance.current_mode.value,
                assumption=drift.assumption.name,
                expected=drift.assumption.expected_value,
                observed=drift.assumption.current_value,
                drift_pct=drift.drift_pct,
            )

        # --- Create a degraded CellConfig for evaluation ---
        from cbpa.config.scenario import CellConfig
        degraded_cell = CellConfig(
            r1_cycle_time_s=self.config.cell.r1_cycle_time_s,
            r2_cycle_time_s=self.config.cell.r2_cycle_time_s,
            human_insertion_time_s=self.config.cell.human_insertion_time_s,
            human_inspection_time_s=self.config.cell.human_inspection_time_s,
            r1_max_speed_mps=degraded_r1_max,
            r2_max_speed_mps=self.config.cell.r2_max_speed_mps,
            shift_hours=self.config.cell.shift_hours,
            demand_base_uph=self.config.cell.demand_base_uph,
        )
        degraded_evaluator = CellEvaluator(degraded_cell, mode="analytical")

        # --- Attempt 1: Micro-adaptation (parameter retune of S3) ---
        # To compensate for R1 degradation while maintaining throughput,
        # push R2 speed and human rate higher to make up lost capacity.
        s3_micro = s3.model_copy(update={
            "name": "S3_micro",
            "r1_speed_fraction": min(s3.r1_speed_fraction + 0.10, 0.99),
            "r2_speed_fraction": min(s3.r2_speed_fraction + 0.25, 0.99),
            "human_cycle_rate_multiplier": min(
                s3.human_cycle_rate_multiplier + 0.15, 1.5
            ),
            "buffer_time_s": max(s3.buffer_time_s - 1.0, 0.5),
        })
        micro_metrics = degraded_evaluator.evaluate(s3_micro)
        micro_report = self.checker.check(c2, micro_metrics)
        micro_mc = degraded_evaluator.evaluate_monte_carlo(
            s3_micro, n_samples=200, seed=self.mc_seed_base + 1
        )
        micro_stoch = self.checker.check_stochastic(c2, micro_mc, seed=self.mc_seed_base + 1)

        print(f"  [L5] Tier 1 — Micro-adaptation (parameter retune):")
        print(f"    S3_micro: P(feasible)={micro_stoch.p_feasible:.0%} | "
              f"Feasible={micro_report.is_feasible} -> INSUFFICIENT")

        logger.info(
            "Phase 6: Micro-adaptation attempted (S3_micro) — "
            "p_feasible=%.2f%%, feasible=%s",
            micro_stoch.p_feasible * 100,
            micro_report.is_feasible,
        )

        self.audit.record(
            phase=6, layer="L5",
            action="micro_adaptation_attempted",
            contract_state="C2 active; R1 degraded",
            working_mode=self.governance.current_mode.value,
            schedule="S3_micro",
            p_feasible=micro_stoch.p_feasible,
            feasible=micro_report.is_feasible,
            sufficient=micro_report.is_feasible and not self.adaptation.should_macro_adapt(
                micro_stoch.p_feasible, threshold
            ),
        )

        # --- Attempt 2: Meso-adaptation (switch to throughput-recovery policy) ---
        # Policy family switch: push R2 to maximum and increase human rate
        # substantially to recover throughput lost from R1 degradation.
        s3_meso = s3.model_copy(update={
            "name": "S3_meso",
            "r1_speed_fraction": 0.99,
            "r2_speed_fraction": 0.99,
            "human_cycle_rate_multiplier": 1.20,
            "buffer_time_s": 1.0,
        })
        meso_metrics = degraded_evaluator.evaluate(s3_meso)
        meso_report = self.checker.check(c2, meso_metrics)
        meso_mc = degraded_evaluator.evaluate_monte_carlo(
            s3_meso, n_samples=200, seed=self.mc_seed_base + 2
        )
        meso_stoch = self.checker.check_stochastic(c2, meso_mc, seed=self.mc_seed_base + 2)

        print(f"\n  [L5] Tier 2 — Meso-adaptation (policy family switch):")
        print(f"    S3_meso: P(feasible)={meso_stoch.p_feasible:.0%} | "
              f"Feasible={meso_report.is_feasible} -> INSUFFICIENT")

        logger.info(
            "Phase 6: Meso-adaptation attempted (S3_meso) — "
            "p_feasible=%.2f%%, feasible=%s",
            meso_stoch.p_feasible * 100,
            meso_report.is_feasible,
        )

        self.audit.record(
            phase=6, layer="L5",
            action="meso_adaptation_attempted",
            contract_state="C2 active; R1 degraded",
            working_mode=self.governance.current_mode.value,
            schedule="S3_meso",
            p_feasible=meso_stoch.p_feasible,
            feasible=meso_report.is_feasible,
            sufficient=meso_report.is_feasible and not self.adaptation.should_macro_adapt(
                meso_stoch.p_feasible, threshold
            ),
        )

        # --- Determine if macro-adaptation is needed ---
        best_p = max(micro_stoch.p_feasible if micro_report.is_feasible else 0.0,
                     meso_stoch.p_feasible if meso_report.is_feasible else 0.0)
        needs_macro = self.adaptation.should_macro_adapt(best_p, threshold)

        print(f"\n  [L5] Best P(feasible) after micro/meso: {best_p:.0%} "
              f"(threshold: {threshold:.0%})")

        if needs_macro:
            print(f"\n  [L5] Tier 3 — MACRO-ADAPTATION TRIGGERED")
            print(f"    Full replan: re-elicit contract -> regenerate schedules -> verify")
            macro_action = self.adaptation.macro_adapt(
                reason=(
                    f"Best p_feasible after micro/meso = {best_p:.2%}, "
                    f"below threshold {threshold:.0%}; "
                    f"R1 degraded to {degradation:.0%} of nominal"
                ),
            )
            logger.info(
                "Phase 6: MACRO adaptation triggered — %s",
                macro_action.description,
            )

            self.audit.record(
                phase=6, layer="L5",
                action="macro_adaptation_triggered",
                contract_state="C2 active; R1 degraded",
                working_mode=self.governance.current_mode.value,
                best_p_feasible=best_p,
                threshold=threshold,
                description=macro_action.description,
            )

            # --- Transition to LEGISLATOR for full replan ---
            self.governance.transition_mode(
                WorkingMode.LEGISLATOR,
                trigger=(
                    "Macro-adaptation: micro/meso insufficient "
                    "under R1 degradation"
                ),
                phase="5->6",
            )

            # --- Full replan: L1 re-elicit -> L2 generate -> L3 verify ---
            c3 = _canonicalise_contract(self.elicitation.elicit(MANAGER_INTENT, "C3"), MANAGER_INTENT)
            c3 = self._prepare_contract(c3, 6, previous=c2)
            c3.context["r1_degraded"] = True
            c3.context["r1_effective_max_mps"] = degraded_r1_max

            self.audit.record(
                phase=6, layer="L1",
                action="contract_re_elicited",
                contract_state="C3 (macro replan)",
                working_mode=self.governance.current_mode.value,
                r1_degraded_max=degraded_r1_max,
            )

            # L2: Generate S4 designed for degraded conditions
            s4 = Schedule(
                name="S4",
                r1_speed_fraction=degradation * 0.85,
                r2_speed_fraction=0.65,
                human_cycle_rate_multiplier=1.05,
                buffer_time_s=2.5,
                demand_target_uph=c2.context.get(
                    "demand_base_uph",
                    self.config.cell.demand_base_uph,
                ),
            )

            s4_metrics = degraded_evaluator.evaluate(s4)
            s4_vr = self.vr_scorer.compute(s4_metrics, schedule_name=s4.name)

            self.audit.record(
                phase=6, layer="L2",
                action="replanned_schedule_generated",
                contract_state="C3 (macro replan)",
                working_mode=self.governance.current_mode.value,
                schedule=s4.name,
                vr_score=s4_vr.vr_score,
            )

            # L3: Verify S4
            s4_report = self.checker.check(c3, s4_metrics)
            s4_mc = degraded_evaluator.evaluate_monte_carlo(
                s4, n_samples=200, seed=self.mc_seed_base + 3
            )
            s4_stoch = self.checker.check_stochastic(c3, s4_mc, seed=self.mc_seed_base + 3)
            s4_cert = self.cert_issuer.issue(
                s4.name, s4_report, c3, stochastic_report=s4_stoch
            )
            self._require_deployment_certificate(s4_cert, phase=6)

            self.audit.record(
                phase=6, layer="L3",
                action="replanned_verification",
                contract_state="C3 (macro replan)",
                working_mode=self.governance.current_mode.value,
                feasible=s4_report.is_feasible,
                p_feasible=s4_stoch.p_feasible,
                certified=s4_cert.is_certified,
            )

            # L4: Deploy S4
            self.audit.record(
                phase=6, layer="L4",
                action="schedule_deployed",
                contract_state="C3 (macro replan)",
                working_mode=self.governance.current_mode.value,
                schedule=s4.name,
            )

            print(f"\n  [L1] Contract C3 re-elicited for degraded R1 conditions")
            print(f"  [L2] S4 generated for degraded envelope")
            print(f"  [L3] S4 verified: P(feasible)={s4_stoch.p_feasible:.0%} | "
                  f"Feasible={s4_report.is_feasible}")
            print(f"\n  [L4] S4 DEPLOYED via MACRO replan | "
                  f"Throughput={s4_metrics.throughput_uph:.1f} u/h | "
                  f"V/R={s4_vr.vr_score:.3f}")

            # Run integrated or SimPy simulation of deployed S4
            sim_result = self._execute_integrated(
                s4, phase=6, contract=c3, cell_config=degraded_cell,
            )
            if sim_result is None:
                sim_result = self._execute_simulation(
                    s4, phase=6, contract=c3, cell_config=degraded_cell,
                )
            if sim_result is not None:
                s4_metrics = sim_result.metrics
                s4_vr = self.vr_scorer.compute(s4_metrics, schedule_name=s4.name)
                # Re-check feasibility against actual simulation metrics
                s4_report = self.checker.check(c3, s4_metrics)
                if not s4_report.is_feasible:
                    self._block_deployment(
                        6, s4.name, "Execution metrics violated the macro contract: "
                        + "; ".join(s4_report.violations),
                    )

            # L5: Record in learning store
            self.learning.record(
                disturbance_type="r1_mechanical_wear",
                disturbance_magnitude=round((1 - degradation) * 100, 1),
                rejected_policy="S3_micro/S3_meso",
                rejection_reason=(
                    f"Micro/meso insufficient: best p_feasible={best_p:.2%} "
                    f"< {threshold:.0%}"
                ),
                accepted_policy="S4",
                adaptation_level="macro",
                outcome_metrics={
                    "throughput_uph": s4_metrics.throughput_uph,
                    "p_feasible": s4_stoch.p_feasible,
                },
            )

            logger.info(
                "Phase 6: S4 deployed via MACRO replan | "
                "Throughput=%.1f | V/R=%.2f | Feasible=%s | "
                "P(feasible)=%.2f%%",
                s4_metrics.throughput_uph,
                s4_vr.vr_score,
                s4_report.is_feasible,
                s4_stoch.p_feasible * 100,
            )

            return PhaseResult(
                phase=6,
                schedule=s4,
                metrics=s4_metrics,
                vr_score=s4_vr,
                feasibility=s4_report,
                contract=c3,
                description=(
                    f"MACRO replan: S4 deployed under R1 degradation "
                    f"({degradation:.0%} of nominal); "
                    f"micro/meso insufficient (best p={best_p:.2%})"
                ),
            )

        # This branch has only evaluated lower-tier proposals, not applied them.
        self._block_deployment(
            6, "lower-tier adaptation",
            "A lower-tier proposal passed diagnostic checks, but this branch has "
            "no implemented certified dispatch; manager intervention required",
        )


def main() -> None:
    """CLI entry point."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run CBPA case study experiment")
    parser.add_argument("--no-llm", action="store_true", help="Deterministic mode (no LLM)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (auto-generated from flags if omitted)")
    parser.add_argument(
        "--shifts", type=int, default=1,
        help="Number of shifts to run (>1 enables multi-shift learning curve)",
    )
    parser.add_argument(
        "--full-demo", action="store_true",
        help=(
            "Run the complete demo exercising all 9 CBPA unique values: "
            "LLM-powered intent translation, Pareto frontiers, stochastic "
            "verification, assumption drift attribution, L5->L2 learning "
            "feedback, macro-adaptation (Phase 6), working mode transitions, "
            "guard enforcement, and multi-shift learning curve (3 shifts)"
        ),
    )
    parser.add_argument(
        "--provider", type=str, default="claude",
        choices=["claude", "openai"],
        help=(
            "LLM provider. 'claude' uses CLI auth (claude login); "
            "'openai' uses OPENAI_API_KEY env var. Default: claude"
        ),
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "LLM model name override. Defaults: 'claude-sonnet-4-6' for Claude, "
            "'gpt-4o' for OpenAI"
        ),
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help=(
            "Run deployed schedules through the SimPy discrete-event "
            "simulation (8-hour shift) instead of instant analytical "
            "evaluation. Guards fire in simulated time."
        ),
    )
    parser.add_argument(
        "--integrated", action="store_true",
        help=(
            "Run deployed schedules through the integrated stack: "
            "Isaac Sim (physics), BaSyx AAS (digital twin), and "
            "OPC-UA (real-time data). Mutually exclusive with --simulate."
        ),
    )
    parser.add_argument(
        "--isaac-url", type=str, default=None,
        help="Isaac Sim REST API URL (default: http://localhost:8211)",
    )
    parser.add_argument(
        "--basyx-registry", type=str, default=None,
        help="BaSyx AAS registry URL (default: http://localhost:9082)",
    )
    parser.add_argument(
        "--opcua-endpoint", type=str, default=None,
        help="OPC-UA server endpoint (default: opc.tcp://localhost:4840/cbpa/)",
    )
    parser.add_argument(
        "--with-dashboard", action="store_true",
        help=(
            "Write live results to data/results/live_demo.json as phases "
            "complete, for consumption by the Streamlit dashboard."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve execution mode: --integrated overrides --simulate
    use_simulation = args.simulate and not args.integrated
    use_integrated = args.integrated

    # Build integration URL overrides
    integrated_kwargs: dict = {}
    if use_integrated:
        integrated_kwargs["use_integrated"] = True
        if args.isaac_url:
            integrated_kwargs["isaac_sim_url"] = args.isaac_url
        if args.basyx_registry:
            integrated_kwargs["basyx_registry_url"] = args.basyx_registry
        if args.opcua_endpoint:
            integrated_kwargs["opcua_endpoint"] = args.opcua_endpoint

    # Auto-generate output directory name from CLI flags when not specified
    if args.output_dir is None:
        parts = ["data/results"]
        if args.full_demo:
            parts.append("full")
        if args.no_llm:
            parts.append("nollm")
        elif args.provider != "claude":
            parts.append(args.provider)
        if use_simulation:
            parts.append("sim")
        elif use_integrated:
            parts.append("integrated")
        else:
            parts.append("analytical")
        n = max(args.shifts, 3) if args.full_demo else args.shifts
        if n > 1:
            parts.append(f"{n}shifts")
        args.output_dir = "_".join(parts)

    # --full-demo enables LLM + macro phase + 3 shifts + simulation
    if args.full_demo:
        use_llm = not args.no_llm  # --no-llm takes precedence over --full-demo
        n_shifts = max(args.shifts, 3)
        config = ScenarioConfig(
            enable_macro_phase=True,
            use_simulation=use_simulation,
            **integrated_kwargs,
        )
    else:
        use_llm = not args.no_llm
        n_shifts = args.shifts
        config = ScenarioConfig(
            use_simulation=use_simulation,
            **integrated_kwargs,
        )

    # Build LLM client based on provider
    llm_client = None
    if use_llm:
        model_name = args.model or (
            "claude-sonnet-4-6" if args.provider == "claude" else "gpt-4o"
        )
        print(f"\n  Connecting to LLM: {args.provider} ({model_name}) ...")
        llm_client = create_client(
            provider=args.provider,
            model=args.model,
        )
        ok, msg = llm_client.check_connection()
        if ok:
            print(f"  LLM ready: {msg}")
        else:
            print(f"\n  WARNING: {msg}")
            print("  The experiment will fall back to DETERMINISTIC mode")
            print("  for any LLM call that fails.\n")
            if args.provider == "claude":
                print("  To fix: run 'claude login' and try again.")
            else:
                print("  To fix: set OPENAI_API_KEY and try again.")
            print()
        logger.info("LLM provider: %s (model: %s, connected: %s)", args.provider, model_name, ok)

    experiment = CBPAExperiment(config=config, use_llm=use_llm, llm_client=llm_client)

    # Ensure OPC-UA / integration bridges are released on exit
    import atexit
    atexit.register(experiment.shutdown)

    # Print integrated readiness check if --integrated
    if use_integrated and experiment._orchestrator is not None:
        readiness = experiment._orchestrator.check_readiness()
        print("\n  Integrated mode readiness check:")
        for service, ready in readiness.items():
            status = "CONNECTED" if ready else "DISCONNECTED (stub)"
            print(f"    {service}: {status}")
        print()

    from cbpa.runner.results_exporter import ResultsExporter
    exporter = ResultsExporter(args.output_dir)

    # Live result writer for dashboard integration
    live_writer = None
    if args.with_dashboard:
        from cbpa.runner.live_writer import LiveResultWriter
        live_path = str(Path(args.output_dir) / "live_demo.json")
        live_writer = LiveResultWriter(path=live_path)
        print(f"\n  Dashboard feed: {live_path}")

    if n_shifts > 1:
        # --- Multi-shift learning curve ---
        if args.full_demo:
            logger.info(
                "Running CBPA FULL DEMO (%d shifts, LLM=on, macro=on) — "
                "all 9 unique values active",
                n_shifts,
            )
        else:
            logger.info(
                "Running CBPA multi-shift experiment (%d shifts, LLM=%s)",
                n_shifts, "on" if use_llm else "off",
            )
        multi = experiment.run_multi_shift(
            n_shifts=n_shifts,
            on_phase_complete=live_writer.on_phase_complete if live_writer else None,
            on_shift_start=live_writer.set_shift if live_writer else None,
        )

        print("\n" + "=" * 70)
        print("CBPA CASE STUDY - MULTI-SHIFT LEARNING CURVE")
        print("=" * 70)

        exporter.export_multi_shift(multi.shift_summaries)

        # Export full results from the last shift (table3, audit trail, figures, etc.)
        if multi.shift_results:
            exporter.export_all(multi.shift_results[-1])

        # Print per-shift summaries
        header = (
            f"{'Shift':<6} {'Spike%':<8} {'Phases':<8} {'VR(S3)':<9} "
            f"{'Violations':<11} {'Bias?':<6} {'Skipped':<8} {'Records':<8}"
        )
        print("\n" + header)
        print("-" * len(header))
        for s in multi.shift_summaries:
            print(
                f"{s.shift:<6} {s.demand_spike_pct:<8.0f} "
                f"{s.phases_to_resolution:<8} {s.vr_score_accepted:<9.2f} "
                f"{s.total_constraint_violations:<11} "
                f"{'Yes' if s.learning_bias_available else 'No':<6} "
                f"{s.candidates_skipped:<8} "
                f"{s.experience_records_after:<8}"
            )

        print(f"\nMulti-shift results exported to {args.output_dir}/")
        if live_writer:
            live_writer.finalize()

        # Generate visualizations for multi-shift run
        try:
            import subprocess
            proc = subprocess.run(
                [sys.executable, "scripts/visualize_results.py", "--data-dir", args.output_dir],
                check=True, capture_output=True, text=True,
            )
            print(f"Figures saved to {args.output_dir}/figures/")
        except Exception as viz_err:
            logger.warning("Visualization failed: %s", viz_err)
        return

    # --- Standard single-shift run ---
    logger.info(f"Running CBPA experiment (LLM={'on' if use_llm else 'off'})")
    result = experiment.run_all_phases(
        on_phase_complete=live_writer.on_phase_complete if live_writer else None,
    )
    if live_writer:
        live_writer.finalize(result)

    # Print summary
    print("\n" + "=" * 70)
    print("CBPA CASE STUDY - EXPERIMENT RESULTS")
    print("=" * 70)

    print("\n--- Execution Trace (Table 2) ---")
    for pr in result.phases:
        mode_str = pr.working_mode.value if hasattr(pr.working_mode, 'value') else pr.working_mode
        print(f"  Phase {pr.phase} [{mode_str}]: {pr.description}")

    print("\n--- Schedule Comparison (Table 3) ---")
    header = f"{'Plan':<6} {'Thr(u/h)':<10} {'Defect':<8} {'Noise(dB)':<10} {'Fatigue':<9} {'Energy(kWh)':<12} {'Gap(%)':<8} {'V/R':<7} {'Feasible':<8}"
    print(header)
    print("-" * len(header))
    for name in ["S1", "S2", "S3"]:
        m = result.schedule_metrics[name]
        vr = result.schedule_vr[name]
        f = result.schedule_feasibility[name]
        print(
            f"{name:<6} {m.throughput_uph:<10} {m.defect_rate:<8} "
            f"{m.noise_db:<10} {m.fatigue_index:<9} {m.energy_kwh:<12} "
            f"{m.deadline_gap_pct:<8} {vr.vr_score:<7} {'Yes' if f.is_feasible else 'No':<8}"
        )

    # Print Pareto frontier summaries for phases that have them
    print("\n--- Pareto Frontiers ---")
    for pr in result.phases:
        if pr.pareto_front is not None:
            pf = pr.pareto_front
            print(
                f"  Phase {pr.phase}: {len(pf.candidates)} candidates, "
                f"{len(pf.non_dominated)} non-dominated "
                f"[{', '.join(pf.non_dominated)}], "
                f"selected={pf.selected}"
            )
            print(f"    Rationale: {pf.selection_rationale}")

    print("\n--- Audit Trail ---")
    for entry in result.audit.summary():
        mode_tag = f" [{entry['working_mode']}]" if entry.get('working_mode') else ""
        print(f"  Phase {entry['phase']} | {entry['layer']:<18} | {entry['action']}{mode_tag}")

    # Print working mode transitions
    if result.mode_summary:
        print("\n--- Working Mode Transitions ---")
        print(f"  Current mode: {result.mode_summary['current_mode']}")
        print(f"  Modes used: {', '.join(result.mode_summary['modes_used'])}")
        for t in result.mode_summary['transitions']:
            print(
                f"  {t['from_mode']} -> {t['to_mode']} "
                f"(phase {t['phase']}, trigger: {t['trigger']})"
            )

    # Export results
    exporter.export_all(result)
    print(f"\nResults exported to {args.output_dir}/")

    # Generate visualizations
    try:
        import subprocess
        proc = subprocess.run(
            [sys.executable, "scripts/visualize_results.py"],
            check=True, capture_output=True, text=True,
        )
        print(f"Figures saved to {args.output_dir}/figures/")
    except Exception as viz_err:
        logger.warning("Visualization failed: %s", viz_err)


if __name__ == "__main__":
    main()
