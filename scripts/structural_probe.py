"""Structural-decision probe (Section 5.5 of the revision).

The factory runner fixes variant routing and operator assignment under H2
absence by a deterministic Layer-2 rule before any candidate is generated.
This probe withholds that rule: the LLM receives the *default* routing and
assignments, is told that H2 is unavailable and that H3 is certified for
inspection only, and is asked for replan candidates.  The script counts how
often the LLM (i) reassigns H3 to Cell B, (ii) avoids scheduling H2, and
(iii) keeps V_B/V_C away from the cell H3 now staffs (the prototype's rule).

Usage:
    python scripts/structural_probe.py --provider claude --model claude-sonnet-4-6 --calls 5
    OPENAI_API_KEY=... OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
        python scripts/structural_probe.py --provider openai --model gemini-3.8-flash --calls 5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from cbpa.config.scenario import FactoryScenarioConfig
from cbpa.llm.client import create_client
from cbpa.models.contract import make_c1_factory
from cbpa.runner import factory_experiment as fe

CONTEXT = (
    "Aggressive replan to meet +20% demand surge under disturbances: operator_absence "
    "(operator H2 is unavailable for the rest of the shift; H3 is the backup and is certified "
    "for inspection only). Push parameters hard but stay within contract constraints."
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--provider", default="claude", choices=["claude", "openai"])
    p.add_argument("--model", default=None)
    p.add_argument("--calls", type=int, default=5)
    p.add_argument("--output", default="data/baselines/structural_probe.json")
    args = p.parse_args()
    logging.disable(logging.WARNING)

    planner_cls = [c for c in vars(fe).values() if isinstance(c, type) and hasattr(c, "generate_replan")][0]
    cfg = FactoryScenarioConfig()
    contract = make_c1_factory()
    client = create_client(provider=args.provider, model=args.model)
    planner = planner_cls(client=client, use_llm=True, config=cfg)
    rows = []
    for i in range(args.calls):
        cands = planner._generate_llm(
            contract, "replan", dict(cfg.default_variant_routing), dict(cfg.default_operator_assignments),
            context=CONTEXT, cell_a_variants=["V_A", "V_B", "V_C"], cell_b_variants=["V_A", "V_B", "V_C"],
            demand_a=62.4, demand_b=57.6,
        )
        for c in cands:
            b = c.cell_schedules["B"]
            rows.append({
                "call": i, "name": c.name, "routing": c.variant_routing, "assign": c.operator_assignments,
                "B_operator": b.assigned_operator, "B_variants": b.assigned_variants,
                "h3_to_B": c.operator_assignments.get("H3") == "B" or b.assigned_operator == "H3",
                "h2_unused": "B" != c.operator_assignments.get("H2") and b.assigned_operator != "H2",
                "vb_vc_off_B": any(c.variant_routing.get(v) == ["A"] for v in ("V_B", "V_C"))
                or ("V_B" not in b.assigned_variants and "V_C" not in b.assigned_variants),
            })
    model = client.model
    summary = {"model": model, "calls": args.calls, "candidates": len(rows),
               "h3_to_B": sum(r["h3_to_B"] for r in rows), "h2_unused": sum(r["h2_unused"] for r in rows),
               "vb_vc_off_B": sum(r["vb_vc_off_B"] for r in rows)}
    print(json.dumps(summary))
    out = Path(args.output)
    data = json.loads(out.read_text()) if out.exists() else {}
    data[model] = {"summary": summary, "candidates": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
