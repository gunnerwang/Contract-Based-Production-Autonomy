"""Schema and coherence validation for outcome contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from cbpa.models.contract import OutcomeContract
from cbpa.layer3_verification.constraint_checker import ConstraintChecker


@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ContractValidator:
    """Validates an OutcomeContract for completeness and coherence."""

    REQUIRED_CONSTRAINTS = {"FatigueIndex", "Noise"}

    def validate(self, contract: OutcomeContract, *, factory: bool = False) -> ValidationResult:
        result = ValidationResult()
        result.errors.extend(ConstraintChecker.contract_errors(contract, factory=factory))
        result.is_valid = not result.errors

        # Must have at least one KPI
        if not contract.kpi_targets:
            result.errors.append("Contract must have at least one KPI target")
            result.is_valid = False

        # Must have hard constraints
        if not contract.hard_constraints:
            result.errors.append("Contract must have hard constraints (K)")
            result.is_valid = False

        # Check required constraints
        constraint_names = {c.name for c in contract.hard_constraints}
        missing = self.REQUIRED_CONSTRAINTS - constraint_names
        if missing:
            result.warnings.append(
                f"Missing recommended constraints: {missing}"
            )

        # Must have priority ordering
        if not contract.priority_order:
            result.warnings.append("No priority ordering specified")

        # Constraint limits must be positive
        for c in contract.hard_constraints:
            if c.limit < 0:
                result.errors.append(
                    f"Constraint {c.name} has negative limit: {c.limit}"
                )
                result.is_valid = False

        # Ambiguity resolution: if any ambiguity flags are present they must
        # all be resolved before the contract is forwarded to Layer 2.
        unresolved = [
            af for af in contract.ambiguities if not af.resolved
        ]
        if unresolved:
            fields = ", ".join(af.field for af in unresolved)
            result.errors.append(
                f"Contract has {len(unresolved)} unresolved ambiguit"
                f"{'y' if len(unresolved) == 1 else 'ies'} "
                f"that must be clarified before scheduling: {fields}"
            )
            result.is_valid = False

        return result
