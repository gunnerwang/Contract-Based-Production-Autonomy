"""Layer-1 baselines: template-based and rule/ontology-based contract elicitation.

Both baselines are deliberately *non-generative*: they can only populate
fields a designer anticipated.

* :class:`TemplateElicitor` — a fixed form with plant defaults; the only
  thing it reads from the intent text is an explicit numeric limit attached
  to one of the two anticipated safety fields (fatigue, noise).
* :class:`RuleBasedElicitor` — a keyword-to-ontology parser with synonym
  lists for KPIs, hard constraints, priorities, context, and assumptions,
  plus numeric extraction.  It covers more vocabulary than the template but
  still only the vocabulary it was given.

Neither baseline detects ambiguities or asks clarification questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cbpa.models.contract import (
    Assumption,
    HardConstraint,
    KPIDirection,
    KPITarget,
    OutcomeContract,
    PriorityLevel,
)

# Plant ontology floors used by the ontology validator (single-cell plant)
PLANT_FLOORS = {"FatigueIndex": 0.4, "Noise": 80.0}
DEFAULT_PRIORITIES = [
    PriorityLevel.HUMAN_WELLBEING, PriorityLevel.SAFETY,
    PriorityLevel.QUALITY, PriorityLevel.THROUGHPUT,
]

_NUM = r"(\d+(?:\.\d+)?)"


def _find_number(text: str, patterns: list[str]) -> float | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


class TemplateElicitor:
    """Fixed-form contract: plant defaults plus explicit numeric overrides."""

    name = "template"

    def elicit(self, intent: str, contract_name: str = "C1") -> OutcomeContract:
        fatigue = _find_number(intent, [rf"fatigue[^.,;]*?(?:under|below|<=?|at most|no more than|max(?:imum)?)\s*{_NUM}"])
        noise = _find_number(intent, [rf"noise[^.,;]*?(?:under|below|<=?|at most|no more than|max(?:imum)?)\s*{_NUM}\s*d?b?"])
        return OutcomeContract(
            name=contract_name,
            kpi_targets=[
                KPITarget(name="Throughput", direction=KPIDirection.MAXIMIZE),
                KPITarget(name="DefectRate", direction=KPIDirection.UPPER_BOUND, threshold=0.001),
                KPITarget(name="ChangeoverTime", direction=KPIDirection.MINIMIZE),
            ],
            hard_constraints=[
                HardConstraint(name="FatigueIndex", operator="<=", limit=min(fatigue, 0.4) if fatigue else 0.4),
                HardConstraint(name="Noise", operator="<=", limit=min(noise, 80.0) if noise else 80.0, unit="dB"),
                HardConstraint(name="CyberRiskLevel", operator="<=", limit=2.0),
            ],
            priority_order=list(DEFAULT_PRIORITIES),
            context={"shift_hours": 8, "product_mix": ["V_A", "V_B"]},
        )


@dataclass
class _Rule:
    keywords: tuple[str, ...]
    canonical: str
    kind: str                      # "kpi" | "constraint" | "assumption"
    direction: KPIDirection | None = None
    default_limit: float | None = None
    unit: str = ""
    number_patterns: tuple[str, ...] = field(default_factory=tuple)


class RuleBasedElicitor:
    """Keyword-to-ontology parser with synonym lists and numeric extraction."""

    name = "rule_based"

    RULES: list[_Rule] = [
        _Rule(("throughput", "output", "units per hour", "productivity", "production rate", "delivered value"),
              "Throughput", "kpi", KPIDirection.MAXIMIZE),
        _Rule(("defect", "scrap", "quality", "first-pass yield", "rework"),
              "DefectRate", "kpi", KPIDirection.UPPER_BOUND, 0.001,
              number_patterns=(rf"(?:defect|scrap)[^.,;]*?(?:under|below|<=?|at most|no more than)\s*{_NUM}\s*%",)),
        _Rule(("changeover", "setup time", "reconfigur"), "ChangeoverTime", "kpi", KPIDirection.MINIMIZE),
        _Rule(("energy", "kwh", "electricity", "power consumption"), "Energy", "constraint", None, None, "kWh",
              number_patterns=(rf"(?:energy|kwh|electricity)[^.,;]*?(?:under|below|<=?|at most|no more than|cap(?:ped)? at)\s*{_NUM}",
                               rf"{_NUM}\s*kwh")),
        _Rule(("fatigue", "tired", "exhaust", "workload", "strain", "fresh", "rest"),
              "FatigueIndex", "constraint", None, 0.4,
              number_patterns=(rf"(?:fatigue|exhaust\w*|workload|strain)[^.,;]*?(?:under|below|<=?|at most|no more than|max(?:imum)?|within)\s*{_NUM}",)),
        _Rule(("noise", "decibel", "db", "acoustic", "loud", "sound level"),
              "Noise", "constraint", None, 80.0, "dB",
              number_patterns=(rf"(?:noise|acoustic|sound|loud\w*|decibel)[^.,;]*?(?:under|below|<=?|at most|no more than|not above|max(?:imum)?|within)\s*{_NUM}",
                               rf"{_NUM}\s*(?:db|decibels?)")),
        _Rule(("cyber", "security", "intrusion"), "CyberRiskLevel", "constraint", None, 2.0),
        _Rule(("robot speed", "arm speed", "m/s", "metres per second", "meters per second"),
              "R1.speed", "assumption", None, 1.5, "m/s",
              number_patterns=(rf"{_NUM}\s*m/s",)),
        _Rule(("latency", "network"), "NetworkLatency", "assumption", None, 20.0, "ms",
              number_patterns=(rf"{_NUM}\s*ms",)),
    ]
    PRIORITY_WORDS = {
        PriorityLevel.HUMAN_WELLBEING: ("well-being", "wellbeing", "operator", "fatigue", "comfort", "ergonom", "health"),
        PriorityLevel.SAFETY: ("safety", "safe"),
        PriorityLevel.QUALITY: ("quality", "defect"),
        PriorityLevel.THROUGHPUT: ("throughput", "output", "deadline", "delivery"),
        PriorityLevel.COST: ("cost", "energy", "budget"),
        PriorityLevel.FLEXIBILITY: ("flexib", "changeover", "variant"),
    }

    def elicit(self, intent: str, contract_name: str = "C1") -> OutcomeContract:
        low = intent.lower()
        kpis: list[KPITarget] = []
        constraints: list[HardConstraint] = []
        assumptions: list[Assumption] = []
        for rule in self.RULES:
            if not any(k in low for k in rule.keywords):
                continue
            number = _find_number(intent, list(rule.number_patterns)) if rule.number_patterns else None
            if rule.kind == "kpi":
                thr = number / 100 if (number is not None and rule.canonical == "DefectRate") else rule.default_limit
                kpis.append(KPITarget(name=rule.canonical, direction=rule.direction, threshold=thr))
            elif rule.kind == "constraint":
                limit = number if number is not None else rule.default_limit
                if limit is None:
                    continue
                if rule.canonical in PLANT_FLOORS:
                    limit = min(limit, PLANT_FLOORS[rule.canonical])
                constraints.append(HardConstraint(name=rule.canonical, operator="<=", limit=limit, unit=rule.unit))
            else:
                value = number if number is not None else rule.default_limit
                assumptions.append(Assumption(name=rule.canonical, description=f"{rule.canonical} bound",
                                              expected_value=value, unit=rule.unit))
        if not any(k.name == "Throughput" for k in kpis):
            kpis.insert(0, KPITarget(name="Throughput", direction=KPIDirection.MAXIMIZE))
        # mandatory plant floors always present (ontology validation)
        names = {c.name for c in constraints}
        for canon, floor in PLANT_FLOORS.items():
            if canon not in names:
                constraints.append(HardConstraint(name=canon, operator="<=", limit=floor))
        if "CyberRiskLevel" not in names:
            constraints.append(HardConstraint(name="CyberRiskLevel", operator="<=", limit=2.0))
        # priorities: explicit "X over Y" / "X first" / "even if it costs Y"; otherwise default order
        order = self._priorities(low)
        ctx: dict = {"shift_hours": 8}
        m = re.search(r"(\d+)\s*-?\s*hour", low)
        if m:
            ctx["shift_hours"] = int(m.group(1))
        if "overtime" in low and ("no overtime" in low or "without overtime" in low):
            ctx["overtime_allowed"] = False
        variants = re.findall(r"\bv_?[a-c]\b", low)
        if variants:
            ctx["product_mix"] = sorted({v.upper().replace("V", "V_").replace("__", "_") for v in variants})
        if any(k in low for k in ("surge", "spike", "rush order", "peak")):
            ctx["demand_spike_risk"] = "High"
        return OutcomeContract(name=contract_name, kpi_targets=kpis, hard_constraints=constraints,
                               priority_order=order, context=ctx, typed_assumptions=assumptions)

    def _priorities(self, low: str) -> list[PriorityLevel]:
        def level_in(segment: str) -> PriorityLevel | None:
            for lvl, words in self.PRIORITY_WORDS.items():
                if any(w in segment for w in words):
                    return lvl
            return None
        explicit: list[PriorityLevel] = []
        for a, b in re.findall(r"([\w\- ]{3,40}?)\s+(?:over|before|ahead of|rather than)\s+([\w\- ]{3,40})", low):
            la, lb = level_in(a), level_in(b)
            if la and lb and la != lb:
                explicit += [x for x in (la, lb) if x not in explicit]
        m = re.search(r"(?:prioriti[sz]e|put)\s+([\w\- ]{3,40}?)\s+(?:first|above all)", low)
        if m:
            la = level_in(m.group(1))
            if la and la not in explicit:
                explicit.insert(0, la)
        m = re.search(r"even if (?:it|that) (?:costs?|reduces?|means less)\s+([\w\- ]{3,30})", low)
        if m:
            lb = level_in(m.group(1))
            if lb and lb in explicit:
                explicit.remove(lb)
            if lb:
                explicit.append(lb)
        rest = [p for p in DEFAULT_PRIORITIES if p not in explicit]
        return explicit + rest
