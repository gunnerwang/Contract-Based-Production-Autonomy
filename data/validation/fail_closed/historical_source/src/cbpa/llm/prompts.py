"""Prompt templates for all LLM agents."""

ELICITATION_SYSTEM = """\
You are an industrial contract elicitation agent for the CBPA framework.
Your role is to parse a production manager's natural language intent and
compile it into a formal outcome contract C^out = <KPI, K, P, X, A>.

Output the contract as structured JSON using the provided tool.
"""

ELICITATION_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "kpi_targets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "direction": {"type": "string", "enum": ["maximize", "minimize", "upper_bound", "lower_bound"]},
                    "threshold": {"type": "number"},
                    "unit": {"type": "string"},
                },
                "required": ["name", "direction"],
            },
        },
        "hard_constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "operator": {"type": "string"},
                    "limit": {"type": "number"},
                    "unit": {"type": "string"},
                },
                "required": ["name", "operator", "limit"],
            },
        },
        "priority_order": {
            "type": "array",
            "items": {"type": "string"},
        },
        "context": {"type": "object"},
        "assumptions": {"type": "object"},
    },
    "required": ["kpi_targets", "hard_constraints", "priority_order"],
}

REFINEMENT_SYSTEM = """\
You are an industrial contract refinement agent for the CBPA framework.
A draft outcome contract has been compiled from a manager's natural language
intent.  Your task is to identify genuine trade-off ambiguities that, if left
unresolved, could cause the optimiser to make a decision the manager did not
intend.

Focus on:
1. KPI objectives that lack an upper bound and could drive unsafe resource use
   (e.g. "maximise throughput" with no energy cap may increase power by ~40%).
2. Hard constraints that directly conflict with a KPI direction under realistic
   operating conditions (e.g. fatigue ≤ 0.4 vs a cycle-rate multiplier > 1.0).
3. Contextual gaps whose default value changes feasibility
   (e.g. shift duration, variant mix, buffer policy).

For each ambiguity produce a short, actionable question a production manager
can answer in one sentence.  Also state the assumption you would apply by
default if the manager is unavailable.

Output structured JSON using the provided tool.
"""

REFINEMENT_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "ambiguities": {
            "type": "array",
            "description": "List of detected ambiguities in the draft contract.",
            "items": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": (
                            "The contract field, KPI name, or constraint name "
                            "that is ambiguous."
                        ),
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "The trade-off question to ask the manager, "
                            "including a quantitative hint where possible."
                        ),
                    },
                    "default_resolution": {
                        "type": "string",
                        "description": (
                            "The assumption applied automatically if no "
                            "manager response is available."
                        ),
                    },
                },
                "required": ["field", "question", "default_resolution"],
            },
        },
    },
    "required": ["ambiguities"],
}

PLANNING_SYSTEM = """\
You are a production planning agent for the CBPA framework.
Given an outcome contract, generate candidate schedules that optimize
the V/R index while respecting constraints.

Propose schedule parameters: r1_speed_fraction, r2_speed_fraction,
human_cycle_rate_multiplier, buffer_time_s.
"""

PLANNING_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "r1_speed_fraction": {"type": "number"},
                    "r2_speed_fraction": {"type": "number"},
                    "human_cycle_rate_multiplier": {"type": "number"},
                    "buffer_time_s": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["name", "r1_speed_fraction", "r2_speed_fraction",
                             "human_cycle_rate_multiplier", "buffer_time_s"],
            },
        },
    },
    "required": ["candidates"],
}

ESCALATION_SYSTEM = """\
You are a governance escalation agent for the CBPA framework.
A constraint conflict has occurred: the system cannot meet demand
without violating human-centric constraints.

Generate a clear, structured explanation for the production manager
with ranked remediation options.
"""

ESCALATION_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "conflict_summary": {"type": "string"},
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "relaxes_constraint": {"type": "string"},
                    "new_limit": {"type": "number"},
                    "expected_deadline_gap_pct": {"type": "number"},
                    "preserves_human_constraints": {"type": "boolean"},
                },
                "required": ["label", "description", "preserves_human_constraints"],
            },
        },
    },
    "required": ["conflict_summary", "options"],
}


# ---------------------------------------------------------------------------
# Factory-level prompts (multi-cell extension)
# ---------------------------------------------------------------------------

FACTORY_PLANNING_SYSTEM = """\
You are a factory-level production planning agent for the CBPA framework.
You manage a two-cell manufacturing factory:

- Cell A (Assembly): R1 pick-and-place, R2 screwdriving, human insertion/inspection
- Cell B (Test & Pack): R3 functional testing, R4 packaging, human QC/labeling

Three product variants with different cycle-time profiles:
- V_A: standard (all operations at nominal speed)
- V_B: assembly-heavy (+15% R1 time, -10% R2 time, +10% human time)
- V_C: fastener-heavy (-20% R1 time, +30% R2 time, +25% human time)

Three operators with different certifications:
- H1: assembly, inspection (max fatigue 0.4)
- H2: testing, inspection, packaging (max fatigue 0.35)
- H3: inspection only (max fatigue 0.45, backup)

AGV transfers products between cells (capacity ~80 uph, 45s per transfer).

Given an outcome contract, decide:
1. Variant routing: which variants go through which cell sequence
2. Operator assignments: which operator works in which cell
3. Per-cell schedule parameters: speeds, buffer times

Constraints couple across cells (combined noise ≤ 82 dB, per-operator fatigue
limits, AGV capacity). Optimize the V/R index across the whole factory.
"""

FACTORY_PLANNING_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "factory_schedule": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "variant_routing": {
                    "type": "object",
                    "description": "variant -> [cell_id sequence]",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "operator_assignments": {
                    "type": "object",
                    "description": "operator_id -> cell_id",
                    "additionalProperties": {"type": "string"},
                },
                "cell_schedules": {
                    "type": "object",
                    "description": "cell_id -> schedule parameters",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "r1_speed_fraction": {"type": "number"},
                            "r2_speed_fraction": {"type": "number"},
                            "human_cycle_rate_multiplier": {"type": "number"},
                            "buffer_time_s": {"type": "number"},
                            "assigned_variants": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "r1_speed_fraction",
                            "r2_speed_fraction",
                            "human_cycle_rate_multiplier",
                            "buffer_time_s",
                        ],
                    },
                },
            },
            "required": [
                "name",
                "variant_routing",
                "operator_assignments",
                "cell_schedules",
            ],
        },
        "rationale": {"type": "string"},
    },
    "required": ["factory_schedule", "rationale"],
}

FACTORY_PLANNING_CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "description": "4-5 factory schedule candidates with diverse trade-offs",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "rationale": {"type": "string"},
                    "variant_routing": {
                        "type": "object",
                        "description": "variant -> [cell_id sequence], e.g. {\"V_A\": [\"A\", \"B\"]}",
                        "additionalProperties": {"type": "array", "items": {"type": "string"}},
                    },
                    "operator_assignments": {
                        "type": "object",
                        "description": "operator_id -> cell_id, e.g. {\"H1\": \"A\", \"H2\": \"B\"}",
                        "additionalProperties": {"type": "string"},
                    },
                    "cell_A": {
                        "type": "object",
                        "properties": {
                            "r1_speed_fraction": {"type": "number", "description": "0.0-1.0"},
                            "r2_speed_fraction": {"type": "number", "description": "0.0-1.0"},
                            "human_cycle_rate_multiplier": {"type": "number", "description": "0.5-2.0"},
                            "buffer_time_s": {"type": "number", "description": "seconds"},
                            "assigned_variants": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["r1_speed_fraction", "r2_speed_fraction",
                                     "human_cycle_rate_multiplier", "buffer_time_s"],
                    },
                    "cell_B": {
                        "type": "object",
                        "properties": {
                            "r1_speed_fraction": {"type": "number", "description": "0.0-1.0"},
                            "r2_speed_fraction": {"type": "number", "description": "0.0-1.0"},
                            "human_cycle_rate_multiplier": {"type": "number", "description": "0.5-2.0"},
                            "buffer_time_s": {"type": "number", "description": "seconds"},
                            "assigned_variants": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["r1_speed_fraction", "r2_speed_fraction",
                                     "human_cycle_rate_multiplier", "buffer_time_s"],
                    },
                },
                "required": ["name", "rationale", "variant_routing", "operator_assignments",
                             "cell_A", "cell_B"],
            },
        },
    },
    "required": ["candidates"],
}

# ---------------------------------------------------------------------------
# Layer 3: Verification repair mediation
# ---------------------------------------------------------------------------

L3_REPAIR_SYSTEM = """\
You are a manufacturing constraint repair agent for the CBPA Layer 3 \
verification firewall. A candidate schedule has been rejected because it \
violates hard constraints (fatigue, noise, energy, etc).

Suggest minimal parameter adjustments that would bring the schedule back \
into feasibility. Consider the physics:
- Increasing robot speeds increases throughput but also noise and energy
- Increasing human cycle rate increases throughput but directly increases fatigue
- Decreasing speeds reduces noise/energy but hurts throughput
- Buffer time adjustments affect pipeline utilization

Prioritize the smallest adjustment that fixes the binding constraint.
"""

L3_REPAIR_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parameter": {
                        "type": "string",
                        "enum": [
                            "r1_speed_fraction",
                            "r2_speed_fraction",
                            "human_cycle_rate_multiplier",
                            "buffer_time_s",
                        ],
                    },
                    "current_value": {"type": "number"},
                    "suggested_value": {"type": "number"},
                    "rationale": {"type": "string"},
                    "expected_improvement": {"type": "string"},
                },
                "required": ["parameter", "current_value", "suggested_value", "rationale"],
            },
        },
    },
    "required": ["suggestions"],
}

# ---------------------------------------------------------------------------
# Layer 4: Execution attribution
# ---------------------------------------------------------------------------

L4_ATTRIBUTION_SYSTEM = """\
You are an execution attribution agent for the CBPA Layer 4 monitoring layer.
The system is executing a deployed schedule, but real-world conditions have \
drifted from the assumptions the contract was written under.

For each breached constraint, explain which assumption drifts caused it, \
through what causal mechanism, and assess urgency:
- "low": can continue, monitor closely
- "medium": needs attention within minutes
- "high": must escalate immediately

Consider manufacturing physics:
- Demand surge -> higher cycle pressure -> more fatigue
- Robot speed increase -> higher noise and energy
- Operator absence -> workload redistribution -> fatigue spikes
"""

L4_ATTRIBUTION_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "explanations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "drifted_assumption": {"type": "string"},
                    "drift_pct": {"type": "number"},
                    "violated_constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "mechanism": {"type": "string"},
                    "urgency": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "recommended_action": {"type": "string"},
                },
                "required": [
                    "drifted_assumption",
                    "violated_constraints",
                    "mechanism",
                    "urgency",
                ],
            },
        },
    },
    "required": ["explanations"],
}

# ---------------------------------------------------------------------------
# Layer 5: Adaptation reasoning and experience synthesis
# ---------------------------------------------------------------------------

L5_REASONING_SYSTEM = """\
You are an adaptation strategy agent for the CBPA Layer 5 adaptation engine.
The system has detected sub-optimal feasibility. Decide which tier:

1. MICRO: Small parameter tweaks (speeds +/-5%, buffer, cycle rate +/-10%).
   Works when violations are small and isolated to one constraint.
2. MESO: Policy family switch (e.g., aggressive -> robust).
   Works when the current policy is fundamentally misaligned with the disturbance.
3. MACRO: Full replanning from scratch (L1-L2).
   Required when a constraint is intrinsically hard under current conditions.

Recommend the smallest tier that could realistically fix the problem.
Use prior experience if available.
"""

L5_REASONING_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "recommended_tier": {
            "type": "string",
            "enum": ["MICRO", "MESO", "MACRO"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "rationale": {"type": "string"},
        "key_insight": {"type": "string"},
    },
    "required": ["recommended_tier", "confidence", "rationale", "key_insight"],
}

L5_SYNTHESIS_SYSTEM = """\
You are an experience synthesis agent for the CBPA learning loop.
Given past adaptation experiences for similar disturbance types, \
synthesize actionable patterns for future planning.

Consider: which policies worked/failed? What parameter ranges succeeded? \
How stable is the pattern given the small sample size?
"""

L5_SYNTHESIS_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "disturbance_pattern": {"type": "string"},
        "successful_policies": {
            "type": "array",
            "items": {"type": "string"},
        },
        "failed_policies": {
            "type": "array",
            "items": {"type": "string"},
        },
        "parameter_hints": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "number"},
            },
        },
        "insight": {"type": "string"},
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": ["disturbance_pattern", "successful_policies", "confidence"],
}


# ---------------------------------------------------------------------------
# Factory-level prompts (multi-cell extension)
# ---------------------------------------------------------------------------

FACTORY_ESCALATION_SYSTEM = """\
You are a factory-level governance escalation agent for the CBPA framework.
A constraint conflict has occurred in a two-cell factory.  The system cannot
meet demand without violating human-centric or safety constraints.

The factory has cross-cell coupling:
- Combined noise from both cells must stay ≤ 82 dB
- Each operator has individual fatigue limits (H1 ≤ 0.4, H2 ≤ 0.35)
- Operators have certification constraints (H3 can only do inspection)
- AGV capacity limits inter-cell throughput

Generate a clear explanation for the production manager with 5 ranked
remediation options that span:
- Constraint relaxation (fatigue or noise limits)
- Product mix adjustment (drop a variant from this shift)
- Overtime / backup operator deployment
- Throughput reduction (accept deadline gap)
- Cross-cell redistribution (change variant routing or operator assignments)

Each option must state whether it preserves human-centric constraints,
its expected throughput impact, and which cells/operators are affected.
"""

FACTORY_ESCALATION_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "conflict_summary": {"type": "string"},
        "affected_cells": {
            "type": "array",
            "items": {"type": "string"},
        },
        "root_cause": {"type": "string"},
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "constraint_relaxation",
                            "product_mix_change",
                            "overtime_backup",
                            "throughput_reduction",
                            "cross_cell_redistribution",
                        ],
                    },
                    "affected_cells": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "affected_operators": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "expected_throughput_change_pct": {"type": "number"},
                    "expected_deadline_gap_pct": {"type": "number"},
                    "preserves_human_constraints": {"type": "boolean"},
                    "operator_reassignment": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "variant_rerouting": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "required": [
                    "label",
                    "description",
                    "category",
                    "preserves_human_constraints",
                ],
            },
        },
    },
    "required": ["conflict_summary", "root_cause", "options"],
}
