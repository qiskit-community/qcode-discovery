"""Tests for evaluation/gates.py (Phase A3).

Covers the plan's "Backward compatibility" bullet that every threshold's
Python default matches existing (pre-weight-5) campaign behavior when no
environment variable is set, and that each getter reads os.environ fresh
on every call (not cached at import time) so a campaign or test can
override it without a process restart.
"""

from __future__ import annotations

import evaluation.gates as gates

_INT_GETTERS = {
    "QCODE_MIN_K_THRESHOLD": gates.min_k_threshold,
    "QCODE_MIN_K_THRESHOLD_NONCSS": gates.min_k_threshold_noncss,
    "QCODE_MIN_RELEVANT_D": gates.min_relevant_d,
    "QCODE_MIN_RELEVANT_D_NONCSS": gates.min_relevant_d_noncss,
    "QCODE_MAX_BUILD_CANDIDATES_CSS": gates.max_build_candidates_css,
    "QCODE_MAX_BUILD_CANDIDATES_NONCSS": gates.max_build_candidates_noncss,
}

_FLOAT_GETTERS = {
    "QCODE_FOM_THRESHOLD_REFINE": gates.fom_threshold_refine,
    "QCODE_FOM_THRESHOLD_EXACT": gates.fom_threshold_exact,
    "QCODE_SAVE_FOM_THRESHOLD_CSS": gates.save_fom_threshold_css,
    "QCODE_SAVE_FOM_THRESHOLD_NONCSS": gates.save_fom_threshold_noncss,
    "QCODE_SAVE_TRUST_RATIO": gates.save_trust_ratio,
    "QCODE_SAVE_TRUST_RATIO_NONCSS": gates.save_trust_ratio_noncss,
}


class TestDefaultsMatchExistingBehavior:
    def test_int_defaults_match_documented_table(self, monkeypatch):
        for name, getter in _INT_GETTERS.items():
            monkeypatch.delenv(name, raising=False)
            assert getter() == gates._INT_DEFAULTS[name]

    def test_float_defaults_match_documented_table(self, monkeypatch):
        for name, getter in _FLOAT_GETTERS.items():
            monkeypatch.delenv(name, raising=False)
            assert getter() == gates._FLOAT_DEFAULTS[name]

    def test_fom_threshold_exact_or_disabled_defaults_to_infinity(self, monkeypatch):
        monkeypatch.delenv("QCODE_FOM_THRESHOLD_EXACT", raising=False)
        assert gates.fom_threshold_exact_or_disabled() == float("inf")

    def test_weight5_connectivity_gate_defaults_to_reject(self, monkeypatch):
        monkeypatch.delenv("QCODE_WEIGHT5_CONNECTIVITY_POLICY", raising=False)
        assert gates.weight5_connectivity_policy() == "reject"

    def test_fom_threshold_exact_or_disabled_opt_in_via_env(self, monkeypatch):
        monkeypatch.setenv("QCODE_FOM_THRESHOLD_EXACT", "9.5")
        assert gates.fom_threshold_exact_or_disabled() == 9.5


class TestEnvOverridesReadFreshEachCall:
    def test_int_getter_reflects_env_change_without_reimport(self, monkeypatch):
        monkeypatch.delenv("QCODE_MIN_K_THRESHOLD", raising=False)
        default = gates.min_k_threshold()
        monkeypatch.setenv("QCODE_MIN_K_THRESHOLD", str(default + 7))
        assert gates.min_k_threshold() == default + 7
        monkeypatch.delenv("QCODE_MIN_K_THRESHOLD", raising=False)
        assert gates.min_k_threshold() == default

    def test_float_getter_reflects_env_change_without_reimport(self, monkeypatch):
        monkeypatch.delenv("QCODE_SAVE_FOM_THRESHOLD_NONCSS", raising=False)
        default = gates.save_fom_threshold_noncss()
        monkeypatch.setenv("QCODE_SAVE_FOM_THRESHOLD_NONCSS", str(default + 1.5))
        assert gates.save_fom_threshold_noncss() == default + 1.5

    def test_every_getter_is_independently_overridable(self, monkeypatch):
        # Overriding one variable must not affect any other getter's default.
        monkeypatch.setenv("QCODE_MIN_K_THRESHOLD", "999")
        for name, getter in _INT_GETTERS.items():
            if name == "QCODE_MIN_K_THRESHOLD":
                continue
            monkeypatch.delenv(name, raising=False)
            assert getter() == gates._INT_DEFAULTS[name]

    def test_weight5_connectivity_policy_allow_is_explicit_and_live(
        self, monkeypatch
    ):
        monkeypatch.setenv("QCODE_WEIGHT5_CONNECTIVITY_POLICY", "ALLOW")
        assert gates.weight5_connectivity_policy() == "allow"
        monkeypatch.setenv("QCODE_WEIGHT5_CONNECTIVITY_POLICY", "reject")
        assert gates.weight5_connectivity_policy() == "reject"

    def test_invalid_weight5_connectivity_policy_fails_closed(
        self, monkeypatch
    ):
        monkeypatch.setenv("QCODE_WEIGHT5_CONNECTIVITY_POLICY", "off")
        try:
            gates.weight5_connectivity_policy()
        except ValueError as exc:
            assert "QCODE_WEIGHT5_CONNECTIVITY_POLICY" in str(exc)
        else:
            raise AssertionError("invalid connectivity policy was accepted")
