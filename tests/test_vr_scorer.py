"""Tests for V/R scorer."""

import pytest
from cbpa.layer2_planning.vr_scorer import VRScorer


class TestVRScorer:
    def test_deterministic_s1(self, s1_metrics):
        scorer = VRScorer(mode="deterministic")
        result = scorer.compute(s1_metrics, schedule_name="S1")
        assert result.vr_score == 0.519

    def test_deterministic_s2(self, s2_metrics):
        scorer = VRScorer(mode="deterministic")
        result = scorer.compute(s2_metrics, schedule_name="S2")
        assert result.vr_score == 0.561

    def test_deterministic_s3(self, s3_metrics):
        scorer = VRScorer(mode="deterministic")
        result = scorer.compute(s3_metrics, schedule_name="S3")
        assert result.vr_score == 0.521

    def test_s2_highest_vr(self, s1_metrics, s2_metrics, s3_metrics):
        """S2 should have the highest V/R score (but violates constraints)."""
        scorer = VRScorer(mode="deterministic")
        vr1 = scorer.compute(s1_metrics, "S1").vr_score
        vr2 = scorer.compute(s2_metrics, "S2").vr_score
        vr3 = scorer.compute(s3_metrics, "S3").vr_score
        assert vr2 > vr3 > vr1

    def test_analytical_mode_positive(self, s1_metrics):
        scorer = VRScorer(mode="analytical")
        result = scorer.compute(s1_metrics, schedule_name="S1")
        assert result.vr_score > 0
        assert result.value_numerator > 0
        assert result.resource_denominator > 0

    def test_component_breakdown_present(self, s1_metrics):
        scorer = VRScorer(mode="deterministic")
        result = scorer.compute(s1_metrics, schedule_name="S1")
        assert "throughput_hat" in result.component_breakdown
        assert "energy_hat" in result.component_breakdown
