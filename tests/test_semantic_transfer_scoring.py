"""Regression checks for semantic scores that must not credit unit/priority errors."""
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import semantic_transfer as transfer
from cbpa.models.contract import make_c1, HardConstraint, PriorityLevel


def case_for(c):
    return {"id": "audit", "category": "audit", "reference": {
        "constraints": [h.model_dump(mode="json") for h in c.hard_constraints],
        "kpi": {"direction": "maximize", "threshold": None},
        "explicit_priority": None}}


def stub_evaluate(case, contract, method, seed, error=None):
    return {"constraints_exact": contract is not None,
            "kpi_exact": contract is not None, "error": error}


def test_wrong_energy_unit_is_not_silently_converted_or_sent_to_planner():
    reference = make_c1()
    reference.hard_constraints.append(HardConstraint(name="Energy", limit=375, unit="kWh"))
    predicted = reference.model_copy(deep=True)
    predicted.get_constraint("Energy").unit = "Wh"
    with patch.object(transfer, "evaluate", side_effect=stub_evaluate) as evaluator:
        row = transfer.score(case_for(reference), predicted, "llm", 42, 0)
    assert evaluator.call_args.args[1] is None
    assert row["constraints_exact"] and row["kpi_exact"]
    assert not row["units_correct"] and not row["exact_scored_semantics"]
    assert predicted.get_constraint("Energy").unit == "Wh"


def test_explicit_priority_error_cannot_receive_full_semantic_credit():
    c = make_c1()
    case = case_for(c)
    case["reference"]["explicit_priority"] = ["Safety", "HumanWellbeing", "Throughput", "Quality"]
    with patch.object(transfer, "evaluate", side_effect=stub_evaluate):
        wrong = transfer.score(case, c, "llm", 42, 0)
        c.priority_order = [PriorityLevel(v) for v in case["reference"]["explicit_priority"]]
        correct = transfer.score(case, c, "llm", 42, 0)
    assert wrong["priority_exact"] is False and not wrong["exact_scored_semantics"]
    assert correct["priority_exact"] and correct["exact_scored_semantics"]


def test_parse_failure_is_not_a_correct_semantic_refusal():
    with patch.object(transfer, "evaluate", side_effect=stub_evaluate):
        row = transfer.score(case_for(make_c1()), None, "llm", 42, 0, "parse failed")
    assert not row["units_correct"] and not row["exact_scored_semantics"]
    assert row["error"] == "parse failed"
