"""Typed, explicitly confirmed clarification updates with atomic writeback.

A structurally valid LLM proposal is not evidence of semantic fidelity or human
approval. Callers must review the patch and supply confirmation separately.
"""
from __future__ import annotations
import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from cbpa.models.contract import HardConstraint, KPITarget, Assumption, PriorityLevel, OutcomeContract
from cbpa.layer1_outcome.contract_validator import ContractValidator

class ClarificationPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    constraints: list[HardConstraint] = Field(default_factory=list)
    kpis: list[KPITarget] = Field(default_factory=list)
    priorities: list[PriorityLevel] | None = None
    assumptions: list[Assumption] = Field(default_factory=list)
    context: dict[str, float | bool | str] = Field(default_factory=dict)
    disposition: Literal["update", "retain"] = "update"

class ClarificationAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    patch: ClarificationPatch


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def apply_answer(contract: OutcomeContract, field: str, answer: ClarificationAnswer,
                 *, confirmed: bool, factory: bool = False) -> OutcomeContract:
    """Apply one reviewed answer atomically; never relax an existing obligation.

    Context and assumptions record the stated specification; they are not new
    monitored constraints. Unsupported hard requirements must remain unresolved.
    """
    if not confirmed:
        raise ValueError("Clarification patch requires explicit confirmation")
    matches = [a for a in contract.ambiguities if a.field == field]
    if len(matches) != 1:
        raise ValueError("Clarification field must identify exactly one pending question")
    if matches[0].resolved:
        raise ValueError("Clarification is already resolved")
    patch = answer.patch
    changes = bool(patch.constraints or patch.kpis or patch.priorities is not None or patch.assumptions or patch.context)
    if (patch.disposition == "update") != changes:
        raise ValueError("Use update with changes or retain with no changes")
    result = contract.model_copy(deep=True)
    for group in (patch.constraints, patch.kpis, patch.assumptions):
        if len({v.name for v in group}) != len(group):
            raise ValueError("Duplicate update field")
    constraints = {c.name: c for c in result.hard_constraints}
    for c in patch.constraints:
        units = {"Energy": "kWh", "Noise": "dB", "FactoryNoise": "dB", "Throughput": "u/h"}
        if c.unit and c.unit != units.get(c.name, ""):
            raise ValueError(f"{c.name}: unsupported unit {c.unit}")
        old = constraints.get(c.name)
        if old:
            upper = old.operator in ("<=", "<") and c.operator in ("<=", "<")
            lower = old.operator in (">=", ">") and c.operator in (">=", ">")
            tighter = ((upper and (c.limit < old.limit or (c.limit == old.limit and (old.operator == "<=" or c.operator == "<"))))
                       or (lower and (c.limit > old.limit or (c.limit == old.limit and (old.operator == ">=" or c.operator == ">"))))
                       or (old.operator == c.operator and old.limit == c.limit))
            if not tighter:
                raise ValueError(f"{c.name}: relaxation or incompatible comparison requires contract revision")
        constraints[c.name] = c.model_copy(deep=True)
    result.hard_constraints = list(constraints.values())
    kpis = {k.name: k for k in result.kpi_targets}
    for k in patch.kpis:
        if k.name not in {"Throughput", "DefectRate", "ChangeoverTime", "Energy"}:
            raise ValueError(f"Unsupported KPI: {k.name}")
        if k.threshold is not None and (not _finite(k.threshold) or k.threshold < 0):
            raise ValueError("Invalid KPI target")
        if k.direction.value in {"upper_bound", "lower_bound"} and k.threshold is None:
            raise ValueError("Bounded KPI requires a target")
        kpis[k.name] = k.model_copy(deep=True)
    result.kpi_targets = list(kpis.values())
    if patch.priorities is not None:
        if not patch.priorities or len(set(patch.priorities)) != len(patch.priorities):
            raise ValueError("Priorities must be nonempty and unique")
        result.priority_order = list(patch.priorities)
    # The executable twin has a fixed shift horizon. Recording a different
    # horizon would not change execution, so refuse it rather than silently accept.
    for key, value in patch.context.items():
        if key == "overtime_allowed" and value is False:
            result.context[key] = value
        elif key == "shift_hours" and _finite(value) and value == contract.context.get(key, 8):
            result.context[key] = value
        else:
            raise ValueError(f"Unsupported context change: {key}")
    assumptions = {a.name: a for a in result.typed_assumptions}
    for a in patch.assumptions:
        if a.name not in assumptions:
            raise ValueError(f"Unsupported monitored assumption: {a.name}")
        if not _finite(a.expected_value) or not _finite(a.tolerance_pct) or a.tolerance_pct < 0:
            raise ValueError("Invalid assumption value or tolerance")
        assumptions[a.name] = a.model_copy(update={"current_value": None, "violated": False, "drift_pct": None})
    result.typed_assumptions = list(assumptions.values())
    for flag in result.ambiguities:
        if flag.field == field:
            flag.resolution = answer.text
            flag.resolved = True
            flag.applied_patch = patch.model_dump(mode="json")
            flag.resolution_source = "confirmed_patch"
    # Validate obligations atomically, while permitting other questions to remain pending.
    validation_copy = result.model_copy(deep=True)
    validation_copy.ambiguities = []
    errors = ContractValidator().validate(validation_copy, factory=factory).errors
    ceilings = ({"FatigueIndex_H1": .4, "FatigueIndex_H2": .35, "FatigueIndex_H3": .4, "FactoryNoise": 82.0}
                if factory else {"FatigueIndex": .4, "Noise": 80.0})
    for name, ceiling in ceilings.items():
        c = result.get_constraint(name)
        if c is None or c.operator not in {"<=", "<", "=="} or c.limit > ceiling:
            errors.append(f"{name}: mandatory plant bound missing or relaxed")
    if errors:
        raise ValueError("; ".join(errors))
    return result


def propose_answer_patch(client, contract, field, answer_text):
    """Extract a reviewable patch; this function does not approve or apply it."""
    import json
    output = client.query_structured(
        system=("Translate the supplied manager answer into a minimal typed contract patch. "
                "Preserve comparison operators, numbers, units, target magnitudes and priority order. "
                "Never treat quoted examples or negated requirements as requested changes. "
                "Use only supported contract fields. Do not invent approval. "
                "A retain disposition means explicit confirmation of existing fields with no changes."),
        user_message=json.dumps({"contract": contract.model_dump(mode="json"), "question_field": field,
                                 "manager_answer": answer_text}),
        tool_name="propose_clarification_patch", tool_schema=ClarificationPatch.model_json_schema())
    return ClarificationAnswer(text=answer_text, patch=ClarificationPatch.model_validate(output))
