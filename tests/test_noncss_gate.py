"""Tests for evaluation/noncss_gate.py (Phase A4).

Exercises passes_noncss_gate's short-circuit behavior (is_equivalently_css
is only called when verify_not_lc_css does not already report is_lc_css),
the shape of its returned dict, and its optional cache -- using
monkeypatched stand-ins for the two underlying (expensive) checks so this
tests the gate's own control flow, not clifford_equivalence's algorithms
(those have their own test coverage).
"""

from __future__ import annotations

import evaluation.noncss_gate as noncss_gate


def _patch_checks(monkeypatch, *, is_lc_css: bool, is_css: bool | None):
    calls = {"lc": 0, "css": 0}

    def fake_verify_not_lc_css(ell, m, A_terms, B_terms, C_terms, D_terms):
        calls["lc"] += 1
        return {"is_lc_css": is_lc_css, "ell": ell, "m": m}

    def fake_is_equivalently_css(code):
        calls["css"] += 1
        assert is_css is not None, "is_equivalently_css should not be called"
        return {"is_css": is_css, "code": code}

    monkeypatch.setattr(noncss_gate, "verify_not_lc_css", fake_verify_not_lc_css)
    monkeypatch.setattr(noncss_gate, "is_equivalently_css", fake_is_equivalently_css)
    return calls


A_TERMS = [(0, 0), (1, 0)]
B_TERMS = [(0, 1), (1, 1), (2, 1)]
C_TERMS = [(0, 0)]
D_TERMS = [(1, 1)]


class TestPassesNoncssGate:
    def test_lc_css_short_circuits_before_css_check(self, monkeypatch):
        calls = _patch_checks(monkeypatch, is_lc_css=True, is_css=None)
        result = noncss_gate.passes_noncss_gate(
            6, 6, A_TERMS, B_TERMS, C_TERMS, D_TERMS, "fake_code",
        )
        assert calls == {"lc": 1, "css": 0}
        assert result["passes"] is False
        assert result["is_lc_css"] is True
        assert result["is_css"] is None
        assert result["css_details"] is None
        assert result["lc_css_details"] is not None

    def test_not_lc_css_and_not_css_passes(self, monkeypatch):
        calls = _patch_checks(monkeypatch, is_lc_css=False, is_css=False)
        result = noncss_gate.passes_noncss_gate(
            6, 6, A_TERMS, B_TERMS, C_TERMS, D_TERMS, "fake_code",
        )
        assert calls == {"lc": 1, "css": 1}
        assert result["passes"] is True
        assert result["is_lc_css"] is False
        assert result["is_css"] is False
        assert result["css_details"] is not None

    def test_not_lc_css_but_equivalently_css_fails(self, monkeypatch):
        calls = _patch_checks(monkeypatch, is_lc_css=False, is_css=True)
        result = noncss_gate.passes_noncss_gate(
            6, 6, A_TERMS, B_TERMS, C_TERMS, D_TERMS, "fake_code",
        )
        assert calls == {"lc": 1, "css": 1}
        assert result["passes"] is False
        assert result["is_lc_css"] is False
        assert result["is_css"] is True

    def test_constructed_code_object_passed_through_to_css_check(self, monkeypatch):
        received = {}

        def fake_verify_not_lc_css(*args):
            return {"is_lc_css": False}

        def fake_is_equivalently_css(code):
            received["code"] = code
            return {"is_css": False}

        monkeypatch.setattr(noncss_gate, "verify_not_lc_css", fake_verify_not_lc_css)
        monkeypatch.setattr(noncss_gate, "is_equivalently_css", fake_is_equivalently_css)

        sentinel_code = object()
        noncss_gate.passes_noncss_gate(
            6, 6, A_TERMS, B_TERMS, C_TERMS, D_TERMS, sentinel_code,
        )
        assert received["code"] is sentinel_code


class TestPassesNoncssGateCache:
    def test_repeated_call_with_same_key_hits_cache(self, monkeypatch):
        calls = _patch_checks(monkeypatch, is_lc_css=False, is_css=False)
        cache: dict = {}
        r1 = noncss_gate.passes_noncss_gate(
            6, 6, A_TERMS, B_TERMS, C_TERMS, D_TERMS, "code1", cache=cache,
        )
        r2 = noncss_gate.passes_noncss_gate(
            6, 6, A_TERMS, B_TERMS, C_TERMS, D_TERMS, "code2", cache=cache,
        )
        # Second call reused the cached result -- underlying checks ran once.
        assert calls == {"lc": 1, "css": 1}
        assert r1 is r2

    def test_different_canonical_key_bypasses_cache(self, monkeypatch):
        calls = _patch_checks(monkeypatch, is_lc_css=False, is_css=False)
        cache: dict = {}
        noncss_gate.passes_noncss_gate(
            6, 6, A_TERMS, B_TERMS, C_TERMS, D_TERMS, "code1", cache=cache,
        )
        noncss_gate.passes_noncss_gate(
            6, 6, A_TERMS, B_TERMS, [], [], "code2", cache=cache,
        )
        assert calls == {"lc": 2, "css": 2}

    def test_term_order_does_not_affect_cache_key(self, monkeypatch):
        calls = _patch_checks(monkeypatch, is_lc_css=False, is_css=False)
        cache: dict = {}
        noncss_gate.passes_noncss_gate(
            6, 6, A_TERMS, B_TERMS, C_TERMS, D_TERMS, "code1", cache=cache,
        )
        reordered_B = list(reversed(B_TERMS))
        noncss_gate.passes_noncss_gate(
            6, 6, A_TERMS, reordered_B, C_TERMS, D_TERMS, "code2", cache=cache,
        )
        assert calls == {"lc": 1, "css": 1}

    def test_no_cache_supplied_recomputes_every_call(self, monkeypatch):
        calls = _patch_checks(monkeypatch, is_lc_css=False, is_css=False)
        noncss_gate.passes_noncss_gate(
            6, 6, A_TERMS, B_TERMS, C_TERMS, D_TERMS, "code1",
        )
        noncss_gate.passes_noncss_gate(
            6, 6, A_TERMS, B_TERMS, C_TERMS, D_TERMS, "code2",
        )
        assert calls == {"lc": 2, "css": 2}
