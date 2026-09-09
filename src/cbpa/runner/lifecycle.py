"""CBPA lifecycle base class: the canonical layer-stack execution pattern.

Both single-cell and factory experiments inherit from this base class,
which defines the 5-round CBPA lifecycle:

    Round 1 (LEGISLATOR): L1 → L2 → L3 → L4
        Elicit contract → generate candidates → verify → deploy

    Round 2 (AUDITOR):    L4
        Monitor → detect disturbance via assumption drift

    Round 3 (AUDITOR):    L2 → L3
        Autonomous replan attempt → rejected by verification firewall

    Round 4 (PARTNER):    Meta
        Escalation → manager selects remediation option

    Round 5 (AUDITOR):    L1 → L5 → L2 → L3 → L4 → L5
        Revise contract → learning bias → balanced replan → verify → deploy → record

Each round always invokes layers in the same canonical order.  Subclasses
override *what* each layer does, never *when* it runs.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from cbpa.layer1_outcome.contract_validator import ContractValidator
from cbpa.layer1_outcome.elicitation_agent import ElicitationAgent
from cbpa.layer2_planning.pareto import pareto_filter
from cbpa.layer2_planning.vr_scorer import VRScorer
from cbpa.layer3_verification.constraint_checker import ConstraintChecker
from cbpa.layer3_verification.feasibility_cert import CertificateIssuer
from cbpa.layer3_verification.guard_synthesizer import GuardEnforcer, RuntimeGuardViolation
from cbpa.layer4_execution.monitor import ContractMonitor
from cbpa.layer5_adaptation.adaptation_engine import AdaptationEngine
from cbpa.layer5_adaptation.learning_store import LearningStore
from cbpa.llm.client import ClaudeClient
from cbpa.meta_layer.audit_trail import AuditTrail
from cbpa.meta_layer.escalation_agent import EscalationAgent
from cbpa.meta_layer.governance import GovernanceOrchestrator
from cbpa.models.escalation import WorkingMode

logger = logging.getLogger(__name__)


class DeploymentBlocked(RuntimeError):
    """The lifecycle halted before dispatch; a manager must resolve rejection."""


class RoundType(str, Enum):
    """Canonical CBPA lifecycle round types."""

    INITIAL_DEPLOY = "initial_deploy"       # L1 → L2 → L3 → L4
    MONITOR = "monitor"                     # L4
    REPLAN_ATTEMPT = "replan_attempt"        # L2 → L3
    ESCALATE = "escalate"                   # Meta
    ADAPT_AND_DEPLOY = "adapt_and_deploy"   # L1 → L5 → L2 → L3 → L4 → L5
    MACRO_ADAPT = "macro_adapt"             # L4 → L5 → L1 → L2 → L3 → L4


@dataclass
class RoundConfig:
    """Configuration for a single lifecycle round."""

    round_number: int
    round_type: RoundType
    description: str
    working_mode: WorkingMode
    transition_after: WorkingMode | None = None


# The canonical CBPA lifecycle: 5 rounds that both single-cell and factory
# experiments share.  Each round maps to a specific subset of the L1-L5 stack.
CBPA_ROUND_SEQUENCE: list[RoundConfig] = [
    RoundConfig(
        round_number=1,
        round_type=RoundType.INITIAL_DEPLOY,
        description="Initial Contract & Deployment",
        working_mode=WorkingMode.LEGISLATOR,
        transition_after=WorkingMode.AUDITOR,
    ),
    RoundConfig(
        round_number=2,
        round_type=RoundType.MONITOR,
        description="Disturbance Detection & Monitoring",
        working_mode=WorkingMode.AUDITOR,
    ),
    RoundConfig(
        round_number=3,
        round_type=RoundType.REPLAN_ATTEMPT,
        description="Autonomous Replan Attempt",
        working_mode=WorkingMode.AUDITOR,
        transition_after=WorkingMode.PARTNER,
    ),
    RoundConfig(
        round_number=4,
        round_type=RoundType.ESCALATE,
        description="Human-in-the-Loop Escalation",
        working_mode=WorkingMode.PARTNER,
        transition_after=WorkingMode.AUDITOR,
    ),
    RoundConfig(
        round_number=5,
        round_type=RoundType.ADAPT_AND_DEPLOY,
        description="Adaptation & Stabilization",
        working_mode=WorkingMode.AUDITOR,
    ),
]


class CBPALifecycle(ABC):
    """Abstract base for CBPA lifecycle-aligned experiments.

    Provides shared layer initialization and the canonical 5-round
    execution pattern.  Subclasses implement the per-round logic for
    their specific schedule, metrics, and evaluator types.
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_client: ClaudeClient | None = None,
        mc_seed_base: int = 42,
    ):
        # Base seed for Monte Carlo verification / simulation.  Each phase
        # derives its own seed as ``mc_seed_base + k`` so that (a) the default
        # of 42 reproduces the published single-run results exactly and (b)
        # repeated-run experiments can vary the verification randomness
        # independently of the LLM sampling randomness.
        self.mc_seed_base = int(mc_seed_base)

        # LLM client
        if llm_client is not None:
            self.client = llm_client
        elif use_llm:
            self.client = ClaudeClient()
        else:
            self.client = None

        use_llm_flag = use_llm and self.client is not None

        # L1: Outcome Elicitation
        self.elicitation = ElicitationAgent(self.client, use_llm=use_llm_flag)
        self.validator = ContractValidator()

        # L2: Planning (VR scorer; candidate generation is subclass-specific)
        self.vr_scorer = VRScorer(mode="analytical")

        # L3: Verification Firewall
        self.checker = ConstraintChecker()
        self.cert_issuer = CertificateIssuer()

        # L4: Guard enforcer (armed during Round 1 verification)
        self._guard_enforcer: GuardEnforcer | None = None

        # L5: Adaptation & Learning
        self.adaptation = AdaptationEngine()
        self.learning = LearningStore()

        # Meta: Governance & Audit
        self.audit = AuditTrail()
        self.escalation = EscalationAgent(self.client, use_llm=use_llm_flag)
        self.governance = GovernanceOrchestrator(self.escalation, self.audit)

    # ------------------------------------------------------------------
    # Lifecycle orchestration
    # ------------------------------------------------------------------

    def _block_deployment(self, phase: int, schedule: str, reason: str) -> None:
        """Fail closed in the runner; do not claim a physical emergency stop."""
        self._guard_enforcer = None
        self.audit.record(
            phase=phase, layer="L3", action="deployment_blocked",
            contract_state="rejected; manager intervention required",
            working_mode=self.governance.current_mode.value,
            schedule=schedule, reason=reason, deployment_authorized=False,
        )
        raise DeploymentBlocked(f"Phase {phase}: {schedule}: {reason}")

    def _block_execution(self, phase: int, reason: str, events=()) -> None:
        """Retain a runtime failure and prevent another execution path."""
        self.audit.record(
            phase=phase, layer="L4", action="runtime_execution_blocked",
            contract_state="runtime failure; manager intervention required",
            working_mode=self.governance.current_mode.value,
            guard_events=[event.model_dump() for event in events], reason=reason,
        )
        self._guard_enforcer = None
        raise DeploymentBlocked(reason)

    def _prepare_contract(self, contract, phase: int, *, factory=False, previous=None):
        """Validate each deployment contract and retain prior encoded obligations.

        These prototype transitions revise context, not permission to relax K.
        Incompatible comparison changes require explicit redesign and halt here.
        """
        errors = self.validator.validate(contract, factory=factory).errors
        if errors:
            self._block_deployment(phase, contract.name, "; ".join(errors))
        contract = contract.model_copy(deep=True)
        if previous is not None:
            constraints = {c.name: c for c in contract.hard_constraints}
            for old in previous.hard_constraints:
                new = constraints.get(old.name)
                if new is None:
                    constraints[old.name] = old.model_copy(deep=True)
                elif new.operator in {"<=", "<"} and old.operator in {"<=", "<"}:
                    if old.limit < new.limit or (old.limit == new.limit and old.operator == "<"):
                        constraints[old.name] = old.model_copy(deep=True)
                elif new.operator in {">=", ">"} and old.operator in {">=", ">"}:
                    if old.limit > new.limit or (old.limit == new.limit and old.operator == ">"):
                        constraints[old.name] = old.model_copy(deep=True)
                elif (new.operator, new.limit) != (old.operator, old.limit):
                    self._block_deployment(phase, contract.name, f"Incompatible revision of {old.name}")
            contract.hard_constraints = list(constraints.values())
        ceilings = ({"FatigueIndex_H1": .4, "FatigueIndex_H2": .35,
                     "FatigueIndex_H3": .4, "FactoryNoise": 82.0} if factory
                    else {"FatigueIndex": .4, "Noise": 80.0})
        errors = self.validator.validate(contract, factory=factory).errors
        for name, limit in ceilings.items():
            c = contract.get_constraint(name)
            if c is None or c.operator not in {"<=", "<", "=="} or c.limit > limit:
                errors.append(f"{name}: contract does not enforce the mandatory upper bound {limit}")
        if errors:
            self._block_deployment(phase, contract.name, "; ".join(errors))
        return contract

    def _require_deployment_certificate(self, certificate, phase: int) -> None:
        """Require nominal AND sampled acceptance before dispatch or pre-staging.

        A missing stochastic report is not permission to use the deterministic
        demonstration path. Successful authorization also refreshes the guards.
        """
        p = certificate.p_feasible
        stochastic = certificate.stochastic_report
        if not (
            certificate.is_certified
            and certificate.feasibility_report.is_feasible
            and not certificate.feasibility_report.evaluation_errors
            and p is not None
            and self.cert_issuer.confidence_threshold <= p <= 1.0
            and stochastic is not None
            and not stochastic.evaluation_errors
            and stochastic.n_samples == 200
            and p == stochastic.p_feasible
            and self.cert_issuer.confidence_threshold <= stochastic.p_feasible <= 1.0
        ):
            self._block_deployment(phase, certificate.schedule_name, certificate.notes)
        self._guard_enforcer = GuardEnforcer(certificate.guards, cyber_risk_level=self.checker.cyber_risk_level)

    def run_all_phases(
        self,
        on_phase_complete: Callable | None = None,
    ) -> Any:
        """Execute the canonical CBPA lifecycle.

        Runs through the 5-round sequence, with working mode transitions
        between rounds.  Each round invokes its layer subset in the
        canonical order.

        Returns the experiment-specific result type.
        """
        result = self._make_result()
        round_results: list[Any] = []

        for rc in self._get_round_sequence():
            logger.info(
                "=== Round %d: %s [%s] ===",
                rc.round_number,
                rc.description,
                rc.working_mode.value.upper(),
            )

            pr = self._execute_round(rc, round_results)
            pr.working_mode = rc.working_mode.value
            round_results.append(pr)
            self._append_result(result, pr)

            if on_phase_complete is not None:
                on_phase_complete(pr)

            # Working mode transition after round
            if rc.transition_after is not None:
                self.governance.transition_mode(
                    rc.transition_after,
                    trigger=self._transition_trigger(rc),
                    phase=f"{rc.round_number}->{rc.round_number + 1}",
                )

        self._finalize_result(result, round_results)
        return result

    @property
    def total_phases(self) -> int:
        """Total number of rounds in this lifecycle."""
        return len(self._get_round_sequence())

    def run_single_phase(self, phase_num: int, prior: list | None = None) -> Any:
        """Execute a single phase by number, given prior phase results.

        This allows step-by-step execution from the UI.
        """
        prior = prior or []
        sequence = self._get_round_sequence()
        if phase_num < 1 or phase_num > len(sequence):
            raise ValueError(f"Phase {phase_num} out of range [1, {len(sequence)}]")

        rc = sequence[phase_num - 1]
        logger.info(
            "=== Round %d: %s [%s] ===",
            rc.round_number, rc.description, rc.working_mode.value.upper(),
        )

        pr = self._execute_round(rc, prior)
        pr.working_mode = rc.working_mode.value

        # Working mode transition
        if rc.transition_after is not None:
            self.governance.transition_mode(
                rc.transition_after,
                trigger=self._transition_trigger(rc),
                phase=f"{rc.round_number}->{rc.round_number + 1}",
            )

        return pr

    def build_result(self, phases: list) -> Any:
        """Build an experiment result from a list of phase results."""
        result = self._make_result()
        for pr in phases:
            self._append_result(result, pr)
        self._finalize_result(result, phases)
        return result

    def _execute_round(self, rc: RoundConfig, prior: list) -> Any:
        """Dispatch to the appropriate round implementation."""
        dispatch = {
            RoundType.INITIAL_DEPLOY: self._round_initial_deploy,
            RoundType.MONITOR: self._round_monitor,
            RoundType.REPLAN_ATTEMPT: self._round_replan,
            RoundType.ESCALATE: self._round_escalate,
            RoundType.ADAPT_AND_DEPLOY: self._round_adapt,
            RoundType.MACRO_ADAPT: self._round_macro_adapt,
        }
        handler = dispatch.get(rc.round_type)
        if handler is None:
            raise ValueError(f"Unknown round type: {rc.round_type}")
        try:
            return handler(rc, prior)
        except RuntimeGuardViolation as exc:
            self._block_execution(
                rc.round_number, f"Runtime guard stopped phase {rc.round_number}: {exc}", exc.events
            )

    def _transition_trigger(self, rc: RoundConfig) -> str:
        """Human-readable trigger for a working mode transition."""
        triggers = {
            RoundType.INITIAL_DEPLOY: "Contract deployed; system enters autonomous monitoring",
            RoundType.REPLAN_ATTEMPT: "Autonomous replan failed; escalation to human manager",
            RoundType.ESCALATE: "Manager decision received; system resumes autonomous execution",
            RoundType.ADAPT_AND_DEPLOY: "Adaptation complete; returning to monitoring",
        }
        return triggers.get(rc.round_type, f"Round {rc.round_number} complete")

    # ------------------------------------------------------------------
    # Round sequence (overridable for experiments with extra rounds)
    # ------------------------------------------------------------------

    def _get_round_sequence(self) -> list[RoundConfig]:
        """Return the round sequence for this experiment.

        Default is the canonical 5-round CBPA lifecycle.  Subclasses can
        override to add rounds (e.g., Phase 6 macro-adaptation).
        """
        return list(CBPA_ROUND_SEQUENCE)

    # ------------------------------------------------------------------
    # Abstract: per-round implementations (subclass-specific)
    # ------------------------------------------------------------------

    @abstractmethod
    def _round_initial_deploy(self, rc: RoundConfig, prior: list) -> Any:
        """Round 1: L1 → L2 → L3 → L4.

        Elicit contract from manager intent, generate candidate schedules,
        verify with stochastic firewall, deploy best with guards armed.
        """

    @abstractmethod
    def _round_monitor(self, rc: RoundConfig, prior: list) -> Any:
        """Round 2: L4.

        Monitor deployed schedule, inject disturbance, detect assumption
        drift and shortfall via ContractMonitor.
        """

    @abstractmethod
    def _round_replan(self, rc: RoundConfig, prior: list) -> Any:
        """Round 3: L2 → L3.

        Generate aggressive replan candidates to meet new demand,
        verify — expected to be REJECTED by L3 firewall.
        """

    @abstractmethod
    def _round_escalate(self, rc: RoundConfig, prior: list) -> Any:
        """Round 4: Meta.

        Escalate constraint conflict to manager with remediation options.
        Manager selects option (deterministic or interactive).
        """

    @abstractmethod
    def _round_adapt(self, rc: RoundConfig, prior: list) -> Any:
        """Round 5: L1 → L5 → L2 → L3 → L4 → L5.

        Revise contract, query learning store for planning bias, generate
        balanced candidates, verify, deploy, record experience.
        """

    def _round_macro_adapt(self, rc: RoundConfig, prior: list) -> Any:
        """Round 6 (optional): L4 → L5 → L1 → L2 → L3 → L4.

        Second disturbance → micro/meso insufficient → macro replan.
        Default raises NotImplementedError; subclasses override if needed.
        """
        raise NotImplementedError("Macro-adaptation not implemented")

    # ------------------------------------------------------------------
    # Abstract: result construction (subclass-specific types)
    # ------------------------------------------------------------------

    @abstractmethod
    def _make_result(self) -> Any:
        """Create an empty experiment result container."""

    @abstractmethod
    def _append_result(self, result: Any, phase_result: Any) -> None:
        """Append a phase result to the experiment result."""

    @abstractmethod
    def _finalize_result(self, result: Any, rounds: list) -> None:
        """Finalize the experiment result after all rounds complete."""
