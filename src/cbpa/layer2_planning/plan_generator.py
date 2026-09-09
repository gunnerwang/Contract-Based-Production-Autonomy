"""Layer 2: Candidate schedule synthesis (LLM + deterministic).

Deterministic mode generates **five** variants per phase by applying fixed,
reproducible parameter offsets around the paper's canonical S1/S2/S3
baselines.  The "a" variant is always identical to the original schedule so
that existing Table-3 results and tests remain unchanged.

Variant naming convention:
    Phase 1 (initial)  → S1a … S1e   (canonical = S1a ≡ S1)
    Phase 3 (fast)     → S2a … S2e   (canonical = S2a ≡ S2)
    Phase 5 (balanced) → S3a … S3e   (canonical = S3a ≡ S3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from cbpa.config.scenario import ScenarioConfig
from cbpa.llm.client import ClaudeClient
from cbpa.llm.prompts import PLANNING_SYSTEM, PLANNING_TOOL_SCHEMA
from cbpa.models.contract import OutcomeContract
from cbpa.models.schedule import Schedule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic parameter offsets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Offsets:
    """Fixed, deterministic offsets applied to a baseline schedule.

    Fields (all additive unless noted):
        r1_delta    – r1_speed_fraction offset
        r2_delta    – r2_speed_fraction offset
        hr_delta    – human_cycle_rate_multiplier offset
        buf_factor  – multiplicative scale applied to buffer_time_s
                      (1.0 = no change; 0.85 = −15% etc.)
    """
    r1_delta: float = 0.0
    r2_delta: float = 0.0
    hr_delta: float = 0.0
    buf_factor: float = 1.0


# Five fixed offsets shared across all three phases.  Variant "a" has zero
# offsets so the canonical S1/S2/S3 values are reproduced exactly.
_VARIANT_OFFSETS: dict[str, _Offsets] = {
    "a": _Offsets(),                                           # conservative / canonical
    "b": _Offsets(r1_delta=+0.10),                            # slightly faster R1 (+10%)
    "c": _Offsets(r2_delta=+0.10),                            # slightly faster R2 (+10%)
    "d": _Offsets(buf_factor=0.85),                           # reduced buffer (−15%)
    "e": _Offsets(r1_delta=+0.05, r2_delta=+0.05,
                  buf_factor=0.90),                            # balanced speedup
}


# Speed fractions are clamped to [0.0, 1.0]; human rate to [0.5, 2.0].
def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class PlanGenerator:
    """Generates candidate schedules for a given contract.

    LLM mode: Claude proposes candidates.
    Deterministic mode: returns 5 parameterised variants of each of the
    paper's S1/S2/S3 templates (named S1a–S1e, S2a–S2e, S3a–S3e).
    The "a" variant is always identical to the original S1/S2/S3 schedule.
    """

    def __init__(
        self,
        config: ScenarioConfig | None = None,
        client: ClaudeClient | None = None,
        use_llm: bool = False,
    ):
        self.config = config or ScenarioConfig()
        self.client = client
        self.use_llm = use_llm and client is not None and client.is_available
        self.llm_only = bool(getattr(self.config, "llm_only", False))
        self.optimiser_anchor = bool(getattr(self.config, "optimiser_anchor", False))

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    llm_only: bool = False
    """Ablation switch: when True, deterministic anchors are withheld and the
    pool holds LLM candidates only (anchors are used solely if the LLM returns
    nothing, which is logged).  Used for the LLM-only condition of H2."""

    def _blend(self, llm: list[Schedule], det: list[Schedule]) -> list[Schedule]:
        """Merge LLM candidates with deterministic anchors.

        Deterministic anchors are renamed with a ``_det`` suffix and
        guarantee at least one feasible candidate survives Pareto
        selection even when LLM candidates are ill-conditioned.
        """
        det_renamed = [
            c.model_copy(update={"name": c.name + "_det", "source": "deterministic"})
            for c in det
        ]
        if self.llm_only:
            if llm:
                return list(llm)
            logger.warning("llm_only: the LLM produced no candidates; deterministic anchors used as fallback")
        return llm + det_renamed

    def generate_initial(
        self, contract: OutcomeContract, learning_bias: dict | None = None,
    ) -> list[Schedule]:
        """Generate candidates for Phase 1 (initial deployment).

        Returns S1a … S1e in deterministic mode; S1a ≡ paper's S1.
        """
        det = self._make_variants("S1", self._base_s1_params(), learning_bias=learning_bias)
        det = self._add_optimiser_anchor(det, contract, self._base_s1_params()["demand"], "S1")
        if not self.use_llm:
            return det
        llm = self._generate_llm(contract, phase="initial", learning_bias=learning_bias,
                                 demand_target=self._base_s1_params()["demand"])
        return self._blend(llm, det)

    def generate_fast(
        self, contract: OutcomeContract, demand_increase_pct: float = 20.0,
        learning_bias: dict | None = None,
    ) -> list[Schedule]:
        """Generate aggressive schedules for Phase 3 (demand surge).

        Returns S2a … S2e in deterministic mode; S2a ≡ paper's S2.
        """
        det = self._make_variants("S2", self._base_s2_params(demand_increase_pct), learning_bias=learning_bias)
        if not self.use_llm:
            return det
        llm = self._generate_llm(contract, phase="fast", learning_bias=learning_bias,
                                 demand_target=self._base_s2_params(demand_increase_pct)["demand"])
        return self._blend(llm, det)

    def generate_balanced(
        self, contract: OutcomeContract, demand_increase_pct: float = 20.0,
        learning_bias: dict | None = None,
    ) -> list[Schedule]:
        """Generate best-effort schedules for Phase 5 (post-negotiation).

        Returns S3a … S3e in deterministic mode; S3a ≡ paper's S3.
        """
        det = self._make_variants("S3", self._base_s3_params(demand_increase_pct), learning_bias=learning_bias)
        det = self._add_optimiser_anchor(det, contract, self._base_s3_params(demand_increase_pct)["demand"], "S3")
        if not self.use_llm:
            return det
        llm = self._generate_llm(contract, phase="balanced", learning_bias=learning_bias,
                                 demand_target=self._base_s3_params(demand_increase_pct)["demand"])
        return self._blend(llm, det)

    # ------------------------------------------------------------------
    # Baseline parameter extraction (unchanged from original _make_sX)
    # ------------------------------------------------------------------

    def _base_s1_params(self) -> dict:
        p = self.config.schedules
        return dict(
            r1=p.s1_r1_speed,
            r2=p.s1_r2_speed,
            hr=p.s1_human_rate,
            buf=p.s1_buffer_s,
            demand=self.config.cell.demand_base_uph,
        )

    def _base_s2_params(self, demand_increase_pct: float = 20.0) -> dict:
        p = self.config.schedules
        return dict(
            r1=p.s2_r1_speed,
            r2=p.s2_r2_speed,
            hr=p.s2_human_rate,
            buf=p.s2_buffer_s,
            demand=self.config.cell.demand_base_uph * (1 + demand_increase_pct / 100),
        )

    def _base_s3_params(self, demand_increase_pct: float = 20.0) -> dict:
        p = self.config.schedules
        return dict(
            r1=p.s3_r1_speed,
            r2=p.s3_r2_speed,
            hr=p.s3_human_rate,
            buf=p.s3_buffer_s,
            demand=self.config.cell.demand_base_uph * (1 + demand_increase_pct / 100),
        )

    # ------------------------------------------------------------------
    # Variant construction
    # ------------------------------------------------------------------

    def _add_optimiser_anchor(self, det: list[Schedule], contract: OutcomeContract, demand: float, prefix: str) -> list[Schedule]:
        """Append the certified constrained-optimiser schedule to the anchor pool (optional)."""
        if not self.optimiser_anchor:
            return det
        from cbpa.baselines.optimizer_planner import OptimiserPlanner
        from cbpa.layer2_planning.vr_scorer import VRScorer
        from cbpa.layer3_verification.constraint_checker import ConstraintChecker
        from cbpa.physics.cell_evaluator import CellEvaluator
        ev = CellEvaluator(cell=self.config.cell, mode="analytical")
        r = OptimiserPlanner(ev, VRScorer(mode="analytical"), ConstraintChecker(), certify=True).plan(contract, demand, name=f"{prefix}opt")
        if r.schedule is None:
            logger.warning("optimiser anchor: no certified schedule found for %s", prefix)
            return det
        return list(det) + [r.schedule.model_copy(update={"name": f"{prefix}opt", "source": "deterministic"})]

    def _make_variants(
        self, base_name: str, base: dict, learning_bias: dict | None = None,
    ) -> list[Schedule]:
        """Apply each of the five fixed offsets to *base* and return schedules.

        The variant letter is appended to *base_name*, e.g. "S1" → "S1a" … "S1e".
        Variant "a" reproduces the canonical schedule exactly (zero offsets).

        When *learning_bias* is provided (from :pyclass:`LearningStore`):
        - Variants whose base name appears in ``avoid_policies`` are skipped
          (variant "a" is never skipped to preserve backward compatibility).
        - ``parameter_hints`` narrow offset values toward historically
          successful operating points.
        """
        avoid = set(learning_bias.get("avoid_policies", [])) if learning_bias else set()
        hints = learning_bias.get("parameter_hints", {}) if learning_bias else {}

        schedules: list[Schedule] = []
        for letter, off in _VARIANT_OFFSETS.items():
            name = f"{base_name}{letter}"

            # Skip variants whose base name matches an avoided policy,
            # but always keep the "a" (canonical) variant.
            if letter != "a" and base_name in avoid:
                logger.debug("Learning bias: skipping variant %s (base %s avoided)", name, base_name)
                continue

            r1 = _clamp(base["r1"] + off.r1_delta, 0.0, 1.0)
            r2 = _clamp(base["r2"] + off.r2_delta, 0.0, 1.0)
            hr = _clamp(base["hr"] + off.hr_delta, 0.5, 2.0)
            buf = max(0.0, base["buf"] * off.buf_factor)

            # Nudge parameters toward hint bounds when available.
            # For throughput hints we leave scheduling parameters alone
            # (throughput is an outcome, not a direct input).  Buffer and
            # speed hints are applied conservatively — pull toward the
            # midpoint of the historically successful range.
            if hints and letter != "a":
                if "throughput_uph" in hints:
                    hint_mid = (hints["throughput_uph"]["min"] + hints["throughput_uph"]["max"]) / 2
                    if base["demand"] > 0:
                        ratio = hint_mid / base["demand"]
                        # Gently scale speeds toward the successful ratio
                        r1 = _clamp(r1 * (0.8 + 0.2 * ratio), 0.0, 1.0)
                        r2 = _clamp(r2 * (0.8 + 0.2 * ratio), 0.0, 1.0)

            schedules.append(
                Schedule(
                    name=name,
                    r1_speed_fraction=round(r1, 4),
                    r2_speed_fraction=round(r2, 4),
                    human_cycle_rate_multiplier=round(hr, 4),
                    buffer_time_s=round(buf, 4),
                    demand_target_uph=base["demand"],
                )
            )

        if learning_bias:
            logger.info(
                "Learning bias applied to %s: avoid=%s, hints=%s, generated %d variants",
                base_name, list(avoid), list(hints.keys()), len(schedules),
            )

        return schedules

    # ------------------------------------------------------------------
    # Backward-compatible single-schedule helpers (kept for external use)
    # ------------------------------------------------------------------

    def _make_s1(self) -> Schedule:
        """Return the canonical S1 schedule (name='S1')."""
        b = self._base_s1_params()
        return Schedule(
            name="S1",
            r1_speed_fraction=b["r1"],
            r2_speed_fraction=b["r2"],
            human_cycle_rate_multiplier=b["hr"],
            buffer_time_s=b["buf"],
            demand_target_uph=b["demand"],
        )

    def _make_s2(self, demand_increase_pct: float = 20.0) -> Schedule:
        """Return the canonical S2 schedule (name='S2')."""
        b = self._base_s2_params(demand_increase_pct)
        return Schedule(
            name="S2",
            r1_speed_fraction=b["r1"],
            r2_speed_fraction=b["r2"],
            human_cycle_rate_multiplier=b["hr"],
            buffer_time_s=b["buf"],
            demand_target_uph=b["demand"],
        )

    def _make_s3(self, demand_increase_pct: float = 20.0) -> Schedule:
        """Return the canonical S3 schedule (name='S3')."""
        b = self._base_s3_params(demand_increase_pct)
        return Schedule(
            name="S3",
            r1_speed_fraction=b["r1"],
            r2_speed_fraction=b["r2"],
            human_cycle_rate_multiplier=b["hr"],
            buffer_time_s=b["buf"],
            demand_target_uph=b["demand"],
        )

    # ------------------------------------------------------------------
    # LLM generation path (unchanged)
    # ------------------------------------------------------------------

    def _generate_llm(
        self, contract: OutcomeContract, phase: str,
        demand_target: float | None = None,
        learning_bias: dict | None = None,
    ) -> list[Schedule]:
        """Use Claude to generate candidate schedules."""
        assert self.client is not None
        if demand_target is None:
            demand_target = self.config.cell.demand_base_uph

        constraint_str = ", ".join(
            f"{c.name} {c.operator} {c.limit}" for c in contract.hard_constraints
        )

        # Build learning-bias context for the LLM prompt when available.
        bias_context = ""
        if learning_bias:
            parts: list[str] = []
            if learning_bias.get("avoid_policies"):
                parts.append(f"AVOID these previously-rejected policies: {learning_bias['avoid_policies']}")
            if learning_bias.get("prefer_families"):
                parts.append(f"PREFER policy families similar to: {learning_bias['prefer_families']}")
            if learning_bias.get("parameter_hints"):
                parts.append(f"Historical parameter hints: {learning_bias['parameter_hints']}")
            if parts:
                bias_context = "\n\nLearning from past shifts:\n" + "\n".join(f"- {p}" for p in parts)

        user_msg = (
            f"Phase: {phase}\n"
            f"Contract: {contract.name}\n"
            f"KPIs: {[k.name for k in contract.kpi_targets]}\n"
            f"Hard constraints: {constraint_str}\n"
            f"Context: {contract.context}\n\n"
            f"Generate candidate schedules. For '{phase}' phase, "
            f"{'propose moderate, comfortable parameters' if phase == 'initial' else ''}"
            f"{'propose aggressive parameters to meet +20% demand' if phase == 'fast' else ''}"
            f"{'propose balanced parameters that maximise output within safe limits' if phase == 'balanced' else ''}"
            f"{bias_context}"
        )

        try:
            result = self.client.query_structured(
                system=PLANNING_SYSTEM,
                user_message=user_msg,
                tool_name="propose_schedules",
                tool_schema=PLANNING_TOOL_SCHEMA,
                tool_description="Propose candidate production schedules",
            )
            schedules = []
            for c in result.get("candidates", []):
                schedules.append(
                    Schedule(
                        name=c.get("name", "S_llm"),
                        source="llm",
                        r1_speed_fraction=c["r1_speed_fraction"],
                        r2_speed_fraction=c["r2_speed_fraction"],
                        human_cycle_rate_multiplier=c["human_cycle_rate_multiplier"],
                        buffer_time_s=c["buffer_time_s"],
                        demand_target_uph=demand_target,
                    )
                )
            return schedules if schedules else [self._make_s1()]
        except Exception as e:
            print(f"  [LLM FALLBACK] Planning: {e} -> using deterministic")
            logger.warning(f"LLM planning failed: {e}, falling back to deterministic")
            if phase == "fast":
                return [self._make_s2()]
            elif phase == "balanced":
                return [self._make_s3()]
            return [self._make_s1()]
