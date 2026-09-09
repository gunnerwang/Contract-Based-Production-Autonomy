"""Feasibility certificate issuance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from cbpa.layer3_verification.guard_synthesizer import GuardSynthesizer, RuntimeGuard
from cbpa.models.contract import OutcomeContract
from cbpa.layer3_verification.constraint_checker import ConstraintChecker
from cbpa.models.metrics import FeasibilityReport, StochasticFeasibilityReport

# Minimum P(feasible) required to issue a certificate when stochastic
# verification has been performed.
DEFAULT_CONFIDENCE_THRESHOLD = 0.95


@dataclass
class FeasibilityCertificate:
    """Certificate that a schedule has passed verification.

    Deterministic path
    ------------------
    ``is_certified`` mirrors ``feasibility_report.is_feasible``.  The
    stochastic fields are ``None``.

    Stochastic path
    ---------------
    When ``stochastic_report`` is provided, ``is_certified`` is ``True`` only
    if *both* the deterministic check passes *and*
    ``stochastic_report.p_feasible >= confidence_level``.
    """

    schedule_name: str
    is_certified: bool
    feasibility_report: FeasibilityReport
    guards: list[RuntimeGuard] = field(default_factory=list)
    issued_at: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""

    # --- Stochastic extensions (optional) ---
    p_feasible: float | None = None
    """Fraction of Monte Carlo samples satisfying all constraints."""

    confidence_level: float | None = None
    """Threshold that p_feasible must meet or exceed for certification."""

    stochastic_report: StochasticFeasibilityReport | None = None
    """Full Monte Carlo report, including violation probabilities and
    sensitivity rankings."""


class CertificateIssuer:
    """Issues feasibility certificates for verified schedules."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.guard_synth = GuardSynthesizer()
        self.confidence_threshold = confidence_threshold

    def issue(
        self,
        schedule_name: str,
        report: FeasibilityReport,
        contract: OutcomeContract,
        stochastic_report: StochasticFeasibilityReport | None = None,
    ) -> FeasibilityCertificate:
        """Issue a certificate for *schedule_name*.

        Parameters
        ----------
        schedule_name:
            Identifier for the candidate schedule.
        report:
            Deterministic feasibility report from ``ConstraintChecker.check()``.
        contract:
            The outcome contract (used for guard synthesis).
        stochastic_report:
            Optional stochastic report from
            ``ConstraintChecker.check_stochastic()``.  When supplied,
            certification additionally requires
            ``p_feasible >= self.confidence_threshold``.
        """
        # Determine certification outcome
        spec = ConstraintChecker.constraint_spec(contract)
        coverage_errors = ConstraintChecker.contract_errors(
            contract, factory=report.verification_scope == "factory"
        )
        if report.checked_constraints != spec or set(report.constraint_results) != {c.name for c in contract.hard_constraints}:
            coverage_errors.append("Nominal report does not cover the supplied contract constraints")
        det_pass = (report.is_feasible and not report.evaluation_errors
                    and not coverage_errors and all(report.constraint_results.values()))

        if stochastic_report is not None:
            if (stochastic_report.checked_constraints != spec
                    or stochastic_report.verification_scope != report.verification_scope):
                coverage_errors.append("Stochastic report does not cover the supplied contract constraints")
            stoch_pass = (not coverage_errors and not stochastic_report.evaluation_errors
                          and stochastic_report.n_samples > 0
                          and self.confidence_threshold <= stochastic_report.p_feasible <= 1.0)
            is_certified = det_pass and stoch_pass
        else:
            stoch_pass = None
            is_certified = det_pass

        guards = self.guard_synth.synthesize(contract, factory=report.verification_scope == "factory") if is_certified else []

        # Build human-readable notes
        if is_certified:
            if stochastic_report is not None:
                notes = (
                    f"All K constraints satisfied. "
                    f"P(feasible)={stochastic_report.p_feasible:.2%} "
                    f"≥ threshold {self.confidence_threshold:.0%}."
                )
            else:
                notes = "All K constraints satisfied"
        else:
            parts: list[str] = list(coverage_errors)
            if not det_pass:
                parts.append(f"deterministic violations: {'; '.join(report.violations)}")
            if stochastic_report is not None and not stoch_pass:
                viol_parts = [
                    f"{c}={p:.0%}"
                    for c, p in sorted(
                        stochastic_report.constraint_violation_probabilities.items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )
                    if p > 0
                ]
                parts.append(
                    f"P(feasible)={stochastic_report.p_feasible:.2%} "
                    f"did not qualify at threshold {self.confidence_threshold:.0%}; "
                    f"violation probs: {', '.join(viol_parts)}"
                )
            if stochastic_report is not None and stochastic_report.evaluation_errors:
                parts.append("incomplete evaluation: " + "; ".join(stochastic_report.evaluation_errors[:3]))
            notes = "REJECTED: " + "; ".join(parts)

        return FeasibilityCertificate(
            schedule_name=schedule_name,
            is_certified=is_certified,
            feasibility_report=report,
            guards=guards,
            notes=notes,
            p_feasible=(
                stochastic_report.p_feasible if stochastic_report is not None else None
            ),
            confidence_level=(
                self.confidence_threshold if stochastic_report is not None else None
            ),
            stochastic_report=stochastic_report,
        )
