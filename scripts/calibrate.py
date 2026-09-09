#!/usr/bin/env python3
"""Calibration check: verify physics models match Table 3 targets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cbpa.config.defaults import TABLE3_TARGETS
from cbpa.config.scenario import ScenarioConfig
from cbpa.models.schedule import Schedule
from cbpa.physics.cell_evaluator import CellEvaluator


def main() -> None:
    config = ScenarioConfig()
    evaluator = CellEvaluator(config.cell, mode="deterministic")

    schedules = {
        "S1": Schedule(
            name="S1",
            r1_speed_fraction=config.schedules.s1_r1_speed,
            r2_speed_fraction=config.schedules.s1_r2_speed,
            human_cycle_rate_multiplier=config.schedules.s1_human_rate,
            buffer_time_s=config.schedules.s1_buffer_s,
            demand_target_uph=config.cell.demand_base_uph,
        ),
        "S2": Schedule(
            name="S2",
            r1_speed_fraction=config.schedules.s2_r1_speed,
            r2_speed_fraction=config.schedules.s2_r2_speed,
            human_cycle_rate_multiplier=config.schedules.s2_human_rate,
            buffer_time_s=config.schedules.s2_buffer_s,
            demand_target_uph=config.cell.demand_base_uph * 1.2,
        ),
        "S3": Schedule(
            name="S3",
            r1_speed_fraction=config.schedules.s3_r1_speed,
            r2_speed_fraction=config.schedules.s3_r2_speed,
            human_cycle_rate_multiplier=config.schedules.s3_human_rate,
            buffer_time_s=config.schedules.s3_buffer_s,
            demand_target_uph=config.cell.demand_base_uph * 1.2,
        ),
    }

    print("Calibration Check: Physics Models vs Table 3")
    print("=" * 65)

    all_pass = True
    tol = config.table3_tolerance_pct / 100

    for name, schedule in schedules.items():
        metrics = evaluator.evaluate(schedule)
        targets = TABLE3_TARGETS[name]

        print(f"\n{name}:")
        for field, target in targets.items():
            actual = getattr(metrics, field)
            if target != 0:
                err = abs(actual - target) / abs(target)
            else:
                err = abs(actual - target)
            ok = err <= tol
            status = "OK" if ok else "FAIL"
            print(f"  {field:<20} actual={actual:<8} target={target:<8} err={err:.3%} [{status}]")
            if not ok:
                all_pass = False

    print("\n" + "=" * 65)
    if all_pass:
        print("CALIBRATION PASSED: All metrics within tolerance")
    else:
        print("CALIBRATION FAILED: Some metrics exceed tolerance")
        sys.exit(1)


if __name__ == "__main__":
    main()
