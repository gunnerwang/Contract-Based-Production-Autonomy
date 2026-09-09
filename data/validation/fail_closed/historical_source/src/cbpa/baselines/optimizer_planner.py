"""Layer-2 baseline: optimisation-based planning over the cell physics.

A constrained search over the same decision variables the anchors and the
LLM propose (robot speed fractions, human cycle-rate multiplier, buffer
time; per cell in the factory), maximising the V/R index (single cell) or
total throughput (factory, as in the factory Pareto objective) subject to
deterministic feasibility on the nominal twin and, optionally, Monte-Carlo
certification (P(feasible) >= threshold).  Coarse grid / random search
followed by local refinement; deterministic given the seed.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

from cbpa.layer2_planning.vr_scorer import VRScorer
from cbpa.layer3_verification.constraint_checker import ConstraintChecker
from cbpa.models.contract import OutcomeContract
from cbpa.models.schedule import CellSchedule, FactorySchedule, Schedule
from cbpa.physics.cell_evaluator import CellEvaluator
from cbpa.physics.factory_evaluator import FactoryEvaluator

BOUNDS = {"r1": (0.30, 1.00), "r2": (0.30, 1.00), "hr": (0.80, 1.30), "buf": (0.5, 6.0)}


@dataclass
class OptimResult:
    schedule: Schedule | FactorySchedule | None
    objective: float
    p_feasible: float | None
    evaluations: int
    seconds: float
    feasible_found: bool


class OptimiserPlanner:
    """Grid + local random search for the best feasible single-cell schedule."""

    def __init__(self, evaluator: CellEvaluator, scorer: VRScorer, checker: ConstraintChecker,
                 certify: bool = False, threshold: float = 0.95, mc_samples: int = 200, seed: int = 42):
        self.ev, self.scorer, self.checker = evaluator, scorer, checker
        self.certify, self.threshold, self.mc, self.seed = certify, threshold, mc_samples, seed

    def _score(self, s: Schedule, contract: OutcomeContract) -> tuple[bool, float]:
        m = self.ev.evaluate(s)
        if not self.checker.check(contract, m).is_feasible:
            return False, -1.0
        return True, self.scorer.compute(m, schedule_name=s.name).vr_score

    def _certified(self, s: Schedule, contract: OutcomeContract) -> float:
        mc = self.ev.evaluate_monte_carlo(s, n_samples=self.mc, seed=self.seed)
        return self.checker.check_stochastic(contract, mc, seed=self.seed).p_feasible

    def plan(self, contract: OutcomeContract, demand: float, name: str = "OPT",
             grid: int = 7, refine: int = 300) -> OptimResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        axes = {k: np.linspace(lo, hi, grid) for k, (lo, hi) in BOUNDS.items()}
        best, best_s, n = -1.0, None, 0
        cand = []
        for r1, r2, hr, buf in itertools.product(axes["r1"], axes["r2"], axes["hr"], axes["buf"]):
            s = Schedule(name=name, source="optimiser", r1_speed_fraction=float(r1), r2_speed_fraction=float(r2),
                         human_cycle_rate_multiplier=float(hr), buffer_time_s=float(buf), demand_target_uph=demand)
            ok, v = self._score(s, contract); n += 1
            if ok:
                cand.append((v, s))
        cand.sort(key=lambda t: -t[0])
        # local refinement around the top grid points
        for v0, s0 in cand[:5]:
            for _ in range(refine // 5):
                s = s0.model_copy(update={
                    "r1_speed_fraction": float(np.clip(s0.r1_speed_fraction + rng.normal(0, 0.03), *BOUNDS["r1"])),
                    "r2_speed_fraction": float(np.clip(s0.r2_speed_fraction + rng.normal(0, 0.03), *BOUNDS["r2"])),
                    "human_cycle_rate_multiplier": float(np.clip(s0.human_cycle_rate_multiplier + rng.normal(0, 0.03), *BOUNDS["hr"])),
                    "buffer_time_s": float(np.clip(s0.buffer_time_s + rng.normal(0, 0.3), *BOUNDS["buf"]))})
                ok, v = self._score(s, contract); n += 1
                if ok:
                    cand.append((v, s))
        cand.sort(key=lambda t: -t[0])
        p_best = None
        for v, s in cand:
            if self.certify:
                p = self._certified(s, contract)
                if p < self.threshold:
                    continue
                p_best = p
            best, best_s = v, s
            break
        if best_s is not None and p_best is None:
            p_best = self._certified(best_s, contract)
        return OptimResult(best_s, best, p_best, n, round(time.perf_counter() - t0, 1), best_s is not None)


class FactoryOptimiserPlanner:
    """Random search + refinement over both cells' parameters for a fixed routing/assignment."""

    def __init__(self, evaluator: FactoryEvaluator, checker: ConstraintChecker,
                 certify: bool = False, threshold: float = 0.95, mc_samples: int = 200, seed: int = 42):
        self.ev, self.checker = evaluator, checker
        self.certify, self.threshold, self.mc, self.seed = certify, threshold, mc_samples, seed

    def _build(self, name, pa, pb, routing, assignments, va, vb, da, db) -> FactorySchedule:
        cells = {}
        for cid, p, variants, demand in (("A", pa, va, da), ("B", pb, vb, db)):
            cells[cid] = CellSchedule(name=f"{name}_{cid}", cell_id=cid, r1_speed_fraction=float(p[0]),
                                      r2_speed_fraction=float(p[1]), human_cycle_rate_multiplier=float(p[2]),
                                      buffer_time_s=float(p[3]), demand_target_uph=demand,
                                      assigned_operator=next((o for o, c in assignments.items() if c == cid), "H1"),
                                      assigned_variants=variants)
        return FactorySchedule(name=name, source="optimiser", cell_schedules=cells, variant_routing=routing,
                               operator_assignments=assignments, agv_priority="balanced")

    def _score(self, fs: FactorySchedule, contract: OutcomeContract) -> tuple[bool, float]:
        m = self.ev.evaluate(fs)
        rep = self.checker.check_factory(contract, m)
        return rep.is_feasible, (m.total_throughput_uph if rep.is_feasible else -1.0)

    def _certified(self, fs: FactorySchedule, contract: OutcomeContract) -> float:
        mc = self.ev.evaluate_monte_carlo(fs, n_samples=self.mc, seed=self.seed)
        return self.checker.check_factory_stochastic(contract, mc, seed=self.seed).p_feasible

    def plan(self, contract: OutcomeContract, routing, assignments, va, vb, da, db, name="FOPT",
             samples: int = 3000, refine: int = 400) -> OptimResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        lo = np.array([BOUNDS[k][0] for k in ("r1", "r2", "hr", "buf")])
        hi = np.array([BOUNDS[k][1] for k in ("r1", "r2", "hr", "buf")])
        cand, n = [], 0
        for _ in range(samples):
            pa = lo + rng.random(4) * (hi - lo); pb = lo + rng.random(4) * (hi - lo)
            fs = self._build(name, pa, pb, routing, assignments, va, vb, da, db)
            ok, v = self._score(fs, contract); n += 1
            if ok:
                cand.append((v, pa, pb))
        cand.sort(key=lambda t: -t[0])
        for v0, pa0, pb0 in cand[:5]:
            for _ in range(refine // 5):
                pa = np.clip(pa0 + rng.normal(0, [0.03, 0.03, 0.03, 0.3]), lo, hi)
                pb = np.clip(pb0 + rng.normal(0, [0.03, 0.03, 0.03, 0.3]), lo, hi)
                fs = self._build(name, pa, pb, routing, assignments, va, vb, da, db)
                ok, v = self._score(fs, contract); n += 1
                if ok:
                    cand.append((v, pa, pb))
        cand.sort(key=lambda t: -t[0])
        best_fs, best, p_best = None, -1.0, None
        for v, pa, pb in cand:
            fs = self._build(name, pa, pb, routing, assignments, va, vb, da, db)
            if self.certify:
                p = self._certified(fs, contract)
                if p < self.threshold:
                    continue
                p_best = p
            best_fs, best = fs, v
            break
        if best_fs is not None and p_best is None:
            p_best = self._certified(best_fs, contract)
        return OptimResult(best_fs, best, p_best, n, round(time.perf_counter() - t0, 1), best_fs is not None)
