"""Explicit disagreement gate for LLM/rule contract cross-checking.

This detects disagreement, not which parser is right. A pending decision blocks
planning; resolution requires a confirmed typed patch or explicit draft review.
"""
from cbpa.models.contract import AmbiguityFlag
from cbpa.models.units import normalize_contract_units
from cbpa.layer1_outcome.contract_validator import ContractValidator

def reconcile_for_review(llm_contract, rule_contract, *, factory=False):
    llm_contract, llm_units = normalize_contract_units(llm_contract)
    rule_contract, rule_units = normalize_contract_units(rule_contract)
    # Existing questions must remain pending, not prevent new disagreements
    # from being collected. Validate structure on copies without clearing flags.
    errors=[]
    for draft in [llm_contract, rule_contract]:
        structural = draft.model_copy(deep=True)
        structural.ambiguities=[]
        errors.extend(ContractValidator().validate(structural,factory=factory).errors)
    if errors:
        raise ValueError("; ".join(errors))
    result=llm_contract.model_copy(deep=True)
    result.context["_parser_review_units"] = {"llm": llm_units, "rules": rule_units}
    left={c.name:c for c in llm_contract.hard_constraints}
    right={c.name:c for c in rule_contract.hard_constraints}
    if len(left)!=len(llm_contract.hard_constraints) or len(right)!=len(rule_contract.hard_constraints):
        raise ValueError("Duplicate constraint names")
    prior_pairs = llm_contract.context.get("_parser_review_pairs", {})
    pairs = {}
    for name in sorted(left.keys()|right.keys()):
        a,b=left.get(name),right.get(name)
        signature=lambda c: [c.operator,c.limit,c.unit] if c else None
        if signature(a)==signature(b):
            continue
        field="crosscheck:"+name
        pairs[field] = {"llm": signature(a), "rules": signature(b)}
        existing = next((f for f in result.ambiguities if f.field == field), None)
        if (existing is not None and existing.resolved
                and existing.applied_patch is not None
                and existing.resolution_source in {"confirmed_patch", "scripted_manager_patch"}
                and prior_pairs.get(field) == pairs[field]):
            continue  # confirmation applies only to this exact normalized pair
        flag = AmbiguityFlag(field=field,
            question=f"Parsers disagree on {name}: LLM={a.model_dump() if a else None}; rules={b.model_dump() if b else None}. Confirm the intended requirement.",
            default_resolution="No automatic resolution; manager confirmation required.")
        if existing is None:
            result.ambiguities.append(flag)
        else:
            result.ambiguities[result.ambiguities.index(existing)] = flag
    result.context["_parser_review_pairs"] = pairs
    # Mandatory bounds may never be waived by parser agreement.
    floors=({"FatigueIndex_H1":.4,"FatigueIndex_H2":.35,"FatigueIndex_H3":.4,"FactoryNoise":82}
            if factory else {"FatigueIndex":.4,"Noise":80,"CyberRiskLevel":2})
    for name,limit in floors.items():
        c=left.get(name)
        if c is None or c.operator not in ("<","<=","==") or c.limit>limit:
            raise ValueError("Missing or relaxed mandatory bound: "+name)
    return result
