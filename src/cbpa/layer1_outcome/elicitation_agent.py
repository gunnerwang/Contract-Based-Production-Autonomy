"""Layer 1: LLM-based intent → contract elicitation."""

from __future__ import annotations

import logging

from cbpa.llm.client import ClaudeClient
from cbpa.llm.prompts import (
    ELICITATION_SYSTEM,
    ELICITATION_TOOL_SCHEMA,
    REFINEMENT_SYSTEM,
    REFINEMENT_TOOL_SCHEMA,
)
from cbpa.models.contract import (
    AmbiguityFlag,
    HardConstraint,
    KPIDirection,
    KPITarget,
    OutcomeContract,
    PriorityLevel,
    make_c1,
    make_c1_factory,
    make_c2,
)

logger = logging.getLogger(__name__)



_PRIORITY_SYNONYMS = (
    (("wellbeing", "well-being", "well being", "human", "operator", "worker", "ergonom", "fatigue", "comfort", "health"), "HumanWellbeing"),
    (("safety", "safe"), "Safety"),
    (("quality", "defect", "scrap", "yield"), "Quality"),
    (("throughput", "output", "productiv", "delivery", "deadline", "volume", "units"), "Throughput"),
    (("cost", "energy", "budget", "expense"), "Cost"),
    (("flexib", "changeover", "variant", "agility"), "Flexibility"),
)


def _normalise_priority(raw: str, priority_map: dict) -> "PriorityLevel | None":
    """Map a free-text priority label to a PriorityLevel (None if unmappable)."""
    if raw in priority_map:
        return priority_map[raw]
    low = raw.lower()
    for keys, canon in _PRIORITY_SYNONYMS:
        if any(k in low for k in keys):
            return priority_map[canon]
    return None

class ElicitationAgent:
    """Parses manager intent into an OutcomeContract.

    LLM mode: uses Claude to parse natural language.
    Deterministic mode: returns the paper's exact C1/C2 contracts.
    """

    def __init__(
        self,
        client: ClaudeClient | None = None,
        use_llm: bool = False,
    ):
        self.client = client
        self.use_llm = use_llm and client is not None and client.is_available

    def elicit(
        self,
        manager_intent: str,
        contract_name: str = "C1",
    ) -> OutcomeContract:
        if self.use_llm:
            return self._elicit_llm(manager_intent, contract_name)
        return self._elicit_deterministic(contract_name)

    def _elicit_deterministic(self, contract_name: str) -> OutcomeContract:
        """Return the paper's exact contract."""
        if contract_name == "C2":
            return make_c2()
        if "factory" in contract_name.lower():
            return make_c1_factory()
        return make_c1()

    def _elicit_llm(
        self, manager_intent: str, contract_name: str
    ) -> OutcomeContract:
        """Use Claude to parse manager intent into a contract."""
        assert self.client is not None

        user_msg = (
            f"The production manager says:\n\"{manager_intent}\"\n\n"
            f"Parse this into a formal outcome contract named {contract_name}. "
            "Include KPI targets, hard constraints (especially fatigue ≤ 0.4 "
            "and noise ≤ 80 dB), priority ordering, operational context, "
            "and assumptions."
        )

        try:
            result = self.client.query_structured(
                system=ELICITATION_SYSTEM,
                user_message=user_msg,
                tool_name="create_contract",
                tool_schema=ELICITATION_TOOL_SCHEMA,
                tool_description="Create a formal outcome contract from manager intent",
            )
            return self._parse_llm_result(result, contract_name)
        except Exception as e:
            print(f"  [LLM FALLBACK] Elicitation: {e} -> using deterministic")
            logger.warning(f"LLM elicitation failed: {e}, falling back to deterministic")
            return self._elicit_deterministic(contract_name)

    # ------------------------------------------------------------------
    # Propose-and-refine: ambiguity detection and resolution
    # ------------------------------------------------------------------

    def elicit_with_refinement(
        self,
        manager_intent: str,
        contract_name: str = "C1",
    ) -> tuple[OutcomeContract, list[AmbiguityFlag]]:
        """Elicit a contract, detect ambiguities, resolve them, return the
        refined contract together with the full ambiguity log.

        This implements CBPA's *propose-and-refine intent translation* loop:

        1. Call :meth:`elicit` to obtain a draft contract.
        2. Call :meth:`_detect_ambiguities` to surface trade-off questions.
        3. Resolve every flag (via LLM dialogue or pre-set deterministic
           answers).
        4. Embed the resolved flags into ``contract.ambiguities`` and return.

        The method is intentionally non-destructive: it never mutates the
        original draft but returns a refined copy.  :meth:`elicit` continues
        to work unchanged for callers that do not need refinement.

        Returns
        -------
        tuple[OutcomeContract, list[AmbiguityFlag]]
            ``(refined_contract, ambiguity_log)`` — the log contains every
            flag that was raised and its resolution.
        """
        # Step 1 — draft contract
        draft = self.elicit(manager_intent, contract_name)
        logger.debug("Refinement loop: draft contract '%s' obtained.", contract_name)

        # Step 2 — detect ambiguities
        flags = self._detect_ambiguities(draft, manager_intent)
        logger.info(
            "Refinement loop: %d ambiguit%s detected in '%s'.",
            len(flags),
            "y" if len(flags) == 1 else "ies",
            contract_name,
        )

        if not flags:
            # Nothing to refine — return the draft as-is with an empty log.
            refined = draft.model_copy(deep=True)
            refined.ambiguities = []
            return refined, []

        # Step 3 — resolve flags
        resolved_flags = self._resolve_flags(flags, draft)

        # Step 4 — attach resolved flags and return
        refined = draft.model_copy(deep=True)
        refined.ambiguities = resolved_flags

        logger.info(
            "Refinement loop: all %d flag(s) resolved for '%s'.",
            len(resolved_flags),
            contract_name,
        )
        return refined, resolved_flags

    def _detect_ambiguities(
        self,
        draft: OutcomeContract,
        manager_intent: str = "",
    ) -> list[AmbiguityFlag]:
        """Identify ambiguities in *draft* that require manager clarification.

        In LLM mode the method asks Claude to perform the analysis.  In
        deterministic mode it applies three hard-coded checks that are
        specifically calibrated to the battery-cell production scenario
        described in the paper:

        1. **Energy cap** — a MAXIMIZE throughput KPI has no energy hard
           constraint, so unconstrained optimisation could increase energy
           consumption by ~40 %.
        2. **Shift extension** — ``shift_hours`` in the context is a single
           point value; it is ambiguous whether overtime is permissible when
           demand exceeds capacity.
        3. **Operator fatigue priority** — the ``FatigueIndex`` constraint
           exists but the manager has not stated whether it is a *hard*
           limit or a *soft* target that can be traded off for throughput.
        """
        if self.use_llm:
            return self._detect_ambiguities_llm(draft, manager_intent)
        return self._detect_ambiguities_deterministic(draft)

    def _detect_ambiguities_deterministic(
        self, draft: OutcomeContract
    ) -> list[AmbiguityFlag]:
        """Deterministic ambiguity checks for the battery-cell scenario."""
        flags: list[AmbiguityFlag] = []

        # --- Check 1: unbounded energy consumption ---
        # If there is a MAXIMIZE throughput KPI but no explicit energy cap
        # constraint, the optimiser may drive energy far above baseline.
        has_maximize_throughput = any(
            kpi.direction == KPIDirection.MAXIMIZE
            and "throughput" in kpi.name.lower()
            for kpi in draft.kpi_targets
        )
        has_energy_constraint = any(
            "energy" in c.name.lower() for c in draft.hard_constraints
        )
        if has_maximize_throughput and not has_energy_constraint:
            flags.append(
                AmbiguityFlag(
                    field="energy_kwh_cap",
                    question=(
                        "Maximising throughput may increase energy consumption "
                        "by ~40 % (from ~2.1 kWh to ~3.0 kWh per shift).  "
                        "Should energy be hard-capped at 3.0 kWh, or is "
                        "unconstrained energy use acceptable?"
                    ),
                    default_resolution=(
                        "Apply a soft energy guideline of 3.0 kWh; "
                        "no hard constraint added unless manager confirms."
                    ),
                )
            )

        # --- Check 2: shift extension policy ---
        # shift_hours is present in context but has no overflow policy.
        shift_hours = draft.context.get("shift_hours")
        demand_spike = draft.context.get("demand_spike_risk", "").lower()
        if shift_hours is not None and demand_spike in ("high", "medium"):
            flags.append(
                AmbiguityFlag(
                    field="shift_hours_overflow",
                    question=(
                        f"The contract sets shift duration at {shift_hours} h, "
                        "but demand spike risk is flagged as high.  "
                        "If demand exceeds capacity within the shift, should "
                        "shift duration extend beyond the planned window, or "
                        "should throughput targets be relaxed instead?"
                    ),
                    default_resolution=(
                        "Do not extend shift hours; relax throughput targets "
                        "and escalate to the manager via the governance layer."
                    ),
                )
            )

        # --- Check 3: fatigue constraint hardness ---
        # FatigueIndex exists — clarify whether it is truly hard or soft.
        fatigue_constraint = draft.get_constraint("FatigueIndex")
        human_wellbeing_priority = (
            draft.priority_order and
            draft.priority_order[0] == PriorityLevel.HUMAN_WELLBEING
        )
        if fatigue_constraint is not None:
            # Only flag if it is not already unambiguously the top priority.
            if not human_wellbeing_priority:
                flags.append(
                    AmbiguityFlag(
                        field="fatigue_constraint_hardness",
                        question=(
                            f"The contract includes FatigueIndex ≤ "
                            f"{fatigue_constraint.limit} as a hard constraint.  "
                            "Should this be treated as an absolute limit that "
                            "must never be violated (hard constraint), or as a "
                            "soft target that can be exceeded by ≤ 10 % if "
                            "throughput is critically short?"
                        ),
                        default_resolution=(
                            "Treat FatigueIndex ≤ "
                            f"{fatigue_constraint.limit} as a hard constraint "
                            "in line with CBPA's human-wellbeing priority."
                        ),
                    )
                )
        elif not human_wellbeing_priority:
            # No fatigue constraint at all and no clear human priority.
            flags.append(
                AmbiguityFlag(
                    field="fatigue_constraint_hardness",
                    question=(
                        "No operator fatigue limit is defined in the contract.  "
                        "Should fatigue be capped at ≤ 0.4 (CBPA default), "
                        "or is fatigue monitoring not required for this shift?"
                    ),
                    default_resolution=(
                        "Add FatigueIndex ≤ 0.4 as a hard constraint "
                        "(CBPA human-wellbeing default)."
                    ),
                )
            )

        return flags

    def _detect_ambiguities_llm(
        self,
        draft: OutcomeContract,
        manager_intent: str,
    ) -> list[AmbiguityFlag]:
        """Use Claude to identify trade-off ambiguities in the draft contract."""
        assert self.client is not None

        import json

        draft_json = draft.model_dump(
            exclude={"ambiguities"},
            mode="json",
        )
        user_msg = (
            f"Original manager intent:\n\"{manager_intent}\"\n\n"
            "Draft outcome contract (JSON):\n"
            f"{json.dumps(draft_json, indent=2)}\n\n"
            "Identify all trade-off ambiguities in this contract that a "
            "production manager should clarify before the schedule optimiser "
            "runs.  Return at most 5 ambiguities, ranked by severity."
        )

        try:
            result = self.client.query_structured(
                system=REFINEMENT_SYSTEM,
                user_message=user_msg,
                tool_name="detect_ambiguities",
                tool_schema=REFINEMENT_TOOL_SCHEMA,
                tool_description=(
                    "Detect trade-off ambiguities in a draft manufacturing "
                    "outcome contract"
                ),
            )
            return [
                AmbiguityFlag(
                    field=a["field"],
                    question=a["question"],
                    default_resolution=a["default_resolution"],
                )
                for a in result.get("ambiguities", [])
            ]
        except Exception as exc:
            print(f"  [LLM FALLBACK] Ambiguity detection: {exc} -> using deterministic")
            logger.warning(
                "LLM ambiguity detection failed (%s); "
                "falling back to deterministic checks.",
                exc,
            )
            return self._detect_ambiguities_deterministic(draft)

    def _resolve_flags(
        self,
        flags: list[AmbiguityFlag],
        draft: OutcomeContract,  # noqa: ARG002  (reserved for LLM context)
    ) -> list[AmbiguityFlag]:
        """Resolve each flag.

        In deterministic mode the resolution is pre-set per flag field so the
        experiment can run without human input.  In LLM mode the same
        pre-set answers are used — a real production system would integrate a
        UI here, but that is outside the scope of the case study.

        The pre-set answers represent the *paper's* intended interpretation of
        the C1 contract for the battery-cell scenario.
        """
        # Map of field → manager's answer (deterministic resolution table)
        _PRESET_RESOLUTIONS: dict[str, str] = {
            "energy_kwh_cap": (
                "Yes — cap energy at 3.0 kWh per shift.  "
                "Throughput should not be maximised at the expense of "
                "excessive energy draw."
            ),
            "shift_hours_overflow": (
                "Do not extend the shift.  If demand cannot be met within "
                "8 h while respecting constraints, escalate to the manager "
                "with the standard governance query."
            ),
            "fatigue_constraint_hardness": (
                "Hard constraint.  FatigueIndex must never exceed 0.4 "
                "regardless of throughput pressure."
            ),
        }

        resolved: list[AmbiguityFlag] = []
        for flag in flags:
            answer = _PRESET_RESOLUTIONS.get(flag.field, flag.default_resolution)
            resolved.append(
                flag.model_copy(
                    update={"resolution": answer, "resolved": True}
                )
            )
            logger.debug(
                "Ambiguity '%s' resolved: %s",
                flag.field,
                answer[:80] + ("…" if len(answer) > 80 else ""),
            )
        return resolved

    def _parse_llm_result(
        self, result: dict, contract_name: str
    ) -> OutcomeContract:
        """Parse LLM structured output into OutcomeContract."""
        direction_map = {
            "maximize": KPIDirection.MAXIMIZE,
            "minimize": KPIDirection.MINIMIZE,
            "upper_bound": KPIDirection.UPPER_BOUND,
            "lower_bound": KPIDirection.LOWER_BOUND,
        }
        priority_map = {v.value: v for v in PriorityLevel}

        kpis = [
            KPITarget(
                name=k["name"],
                direction=direction_map.get(k["direction"], KPIDirection.MAXIMIZE),
                threshold=k.get("threshold"),
                unit=k.get("unit", ""),
            )
            for k in result.get("kpi_targets", [])
        ]

        constraints = [
            HardConstraint(
                name=c["name"],
                operator=c.get("operator", "<="),
                limit=c["limit"],
                unit=c.get("unit", ""),
            )
            for c in result.get("hard_constraints", [])
        ]

        # Priority labels arrive in the LLM's own wording ("human well-being",
        # "operator_wellbeing", "Safety first", ...); normalise them to the
        # ontology and drop what cannot be mapped rather than defaulting.
        priorities: list[PriorityLevel] = []
        for raw in result.get("priority_order", []):
            lvl = _normalise_priority(str(raw), priority_map)
            if lvl is not None and lvl not in priorities:
                priorities.append(lvl)
        raw_priorities = [str(p) for p in result.get("priority_order", [])]

        return OutcomeContract(
            name=contract_name,
            kpi_targets=kpis,
            hard_constraints=constraints,
            priority_order=priorities,
            context={**(result.get("context", {}) or {}), "_raw_priority_order": raw_priorities},
            assumptions=result.get("assumptions", {}),
        )
