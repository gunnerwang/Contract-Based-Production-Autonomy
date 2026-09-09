"""Tests for constraint checker."""

import pytest
from cbpa.layer3_verification.constraint_checker import ConstraintChecker
from cbpa.layer3_verification.feasibility_cert import CertificateIssuer
from cbpa.layer3_verification.guard_synthesizer import GuardSynthesizer


class TestConstraintChecker:
    def test_s1_feasible(self, c1, s1_metrics):
        checker = ConstraintChecker()
        report = checker.check(c1, s1_metrics)
        assert report.is_feasible
        assert len(report.violations) == 0

    def test_s2_infeasible(self, c1, s2_metrics):
        checker = ConstraintChecker()
        report = checker.check(c1, s2_metrics)
        assert not report.is_feasible
        assert len(report.violations) == 2

    def test_s2_violates_fatigue(self, c1, s2_metrics):
        checker = ConstraintChecker()
        report = checker.check(c1, s2_metrics)
        fatigue_violations = [v for v in report.violations if "FatigueIndex" in v]
        assert len(fatigue_violations) == 1

    def test_s2_violates_noise(self, c1, s2_metrics):
        checker = ConstraintChecker()
        report = checker.check(c1, s2_metrics)
        noise_violations = [v for v in report.violations if "Noise" in v]
        assert len(noise_violations) == 1

    def test_s3_feasible(self, c1, s3_metrics):
        checker = ConstraintChecker()
        report = checker.check(c1, s3_metrics)
        assert report.is_feasible

    def test_constraint_margins(self, c1, s1_metrics):
        checker = ConstraintChecker()
        report = checker.check(c1, s1_metrics)
        assert report.constraint_margins["FatigueIndex"] > 0
        assert report.constraint_margins["Noise"] > 0


class TestGuardSynthesizer:
    def test_synthesizes_guards(self, c1):
        synth = GuardSynthesizer()
        guards = synth.synthesize(c1)
        assert len(guards) >= 2  # encoded constraints and optional noise warning
        names = [g.name for g in guards]
        assert "FatigueIndex" in names
        assert "Noise" in names

    def test_guard_check_pass(self, c1):
        synth = GuardSynthesizer()
        guards = synth.synthesize(c1)
        fatigue_guard = next(g for g in guards if g.name == "FatigueIndex")
        assert fatigue_guard.check(0.3)
        assert not fatigue_guard.check(0.5)


class TestCertificateIssuer:
    def test_issues_certificate_for_feasible(self, c1, s1_metrics):
        checker = ConstraintChecker()
        report = checker.check(c1, s1_metrics)
        issuer = CertificateIssuer()
        cert = issuer.issue("S1", report, c1)
        assert cert.is_certified
        assert len(cert.guards) > 0

    def test_rejects_certificate_for_infeasible(self, c1, s2_metrics):
        checker = ConstraintChecker()
        report = checker.check(c1, s2_metrics)
        issuer = CertificateIssuer()
        cert = issuer.issue("S2", report, c1)
        assert not cert.is_certified
        assert len(cert.guards) == 0
