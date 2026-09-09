"""Pareto front filtering for candidate schedules.

Implements true multi-objective non-dominance over two objectives:
  - Objective 1: maximise V/R score (value delivered per resource consumed)
  - Objective 2: maximise minimum constraint margin
        (= min over hard constraints of (limit − actual), so that a larger
         value means the schedule sits further inside the feasible region)

Candidate A *dominates* candidate B when:
  A is at least as good as B on every objective, AND
  A is strictly better than B on at least one objective.

The non-dominated (Pareto-optimal) set contains every candidate that is
not dominated by any other.  From that set the best *feasible* candidate
is selected (highest V/R score among feasible; if none are feasible the
highest V/R overall is returned as a fallback).
"""

from __future__ import annotations

import logging

from cbpa.models.metrics import ParetoFrontResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core non-dominance logic
# ---------------------------------------------------------------------------

def _dominates(a: dict, b: dict) -> bool:
    """Return True if candidate *a* dominates candidate *b*.

    Objectives maximised: ``vr_score``, ``constraint_margin``.
    a dominates b iff:
      a.vr_score    >= b.vr_score    AND
      a.constraint_margin >= b.constraint_margin AND
      (a.vr_score > b.vr_score OR a.constraint_margin > b.constraint_margin)
    """
    vr_a = a["vr_score"]
    vr_b = b["vr_score"]
    cm_a = a["constraint_margin"]
    cm_b = b["constraint_margin"]

    at_least_as_good = vr_a >= vr_b and cm_a >= cm_b
    strictly_better = vr_a > vr_b or cm_a > cm_b
    return at_least_as_good and strictly_better


def _compute_non_dominated(candidates: list[dict]) -> tuple[list[str], list[str]]:
    """Partition candidate names into (non_dominated, dominated).

    Each candidate dict must contain keys: ``name``, ``vr_score``,
    ``constraint_margin``.
    """
    n = len(candidates)
    is_dominated_flag = [False] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(candidates[j], candidates[i]):
                is_dominated_flag[i] = True
                break

    non_dominated = [
        candidates[i]["name"]
        for i in range(n)
        if not is_dominated_flag[i]
    ]
    dominated = [
        candidates[i]["name"]
        for i in range(n)
        if is_dominated_flag[i]
    ]
    return non_dominated, dominated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pareto_filter(candidates: list[dict]) -> ParetoFrontResult:
    """Multi-objective Pareto filter over a set of candidate schedules.

    Parameters
    ----------
    candidates:
        List of dicts, each containing at minimum:
            name              – unique schedule identifier (str)
            vr_score          – V/R index (float, higher is better)
            constraint_margin – minimum hard-constraint margin across all
                                constraints (float, higher = safer)
            feasible          – bool, True if all hard constraints are met

        Any additional keys are forwarded verbatim to
        ``ParetoFrontResult.scores``.

    Returns
    -------
    ParetoFrontResult
        Full Pareto analysis including the selected schedule name and a
        plain-English rationale string.
    """
    if not candidates:
        raise ValueError("pareto_filter received an empty candidate list")

    all_names = [c["name"] for c in candidates]

    # Build compact score dict per candidate (drop non-numeric keys)
    _SKIP_KEYS = {"name", "source"}
    scores: dict[str, dict[str, float]] = {}
    for c in candidates:
        entry: dict[str, float] = {
            k: (float(v) if not isinstance(v, bool) else float(v))
            for k, v in c.items()
            if k not in _SKIP_KEYS
        }
        scores[c["name"]] = entry

    # Compute Pareto front
    non_dominated, dominated = _compute_non_dominated(candidates)

    # --- Selection: best feasible candidate from the non-dominated set ---
    nd_set = {c["name"]: c for c in candidates if c["name"] in non_dominated}

    feasible_nd = [c for c in nd_set.values() if c.get("feasible", False)]
    infeasible_nd = [c for c in nd_set.values() if not c.get("feasible", False)]

    if feasible_nd:
        # Among feasible non-dominated candidates: pick highest V/R score
        best = max(feasible_nd, key=lambda c: c["vr_score"])
        rationale = (
            f"Selected '{best['name']}' from the non-dominated front "
            f"({len(non_dominated)} of {len(candidates)} candidates) as the "
            f"feasible candidate with the highest V/R score "
            f"({best['vr_score']:.3f}) and constraint margin "
            f"({best['constraint_margin']:.4f})."
        )
    elif infeasible_nd:
        # All non-dominated candidates are infeasible — pick highest V/R
        # as the least-bad option and flag it
        best = max(infeasible_nd, key=lambda c: c["vr_score"])
        rationale = (
            f"No feasible candidate exists in the non-dominated front. "
            f"Returning '{best['name']}' (highest V/R = {best['vr_score']:.3f}) "
            f"as a fallback; manual review required before deployment."
        )
        logger.warning(
            "pareto_filter: no feasible candidate in non-dominated set – "
            "returning infeasible fallback '%s'",
            best["name"],
        )
    else:
        # Degenerate: non-dominated set is empty (shouldn't happen with >= 1 candidate)
        best = max(candidates, key=lambda c: c["vr_score"])
        rationale = (
            f"Degenerate Pareto result; selected '{best['name']}' by V/R score only."
        )
        logger.warning("pareto_filter: empty non-dominated set – falling back to VR-only selection")

    logger.info(
        "Pareto filter | candidates=%d | non-dominated=%d | dominated=%d | selected=%s",
        len(candidates),
        len(non_dominated),
        len(dominated),
        best["name"],
    )

    # Extract candidate provenance (llm vs deterministic)
    candidate_sources = {c["name"]: c.get("source", "deterministic") for c in candidates}

    return ParetoFrontResult(
        candidates=all_names,
        non_dominated=non_dominated,
        dominated=dominated,
        selected=best["name"],
        selection_rationale=rationale,
        scores=scores,
        candidate_sources=candidate_sources,
    )
