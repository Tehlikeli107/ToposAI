"""
test_coverage_boost9.py — Cover remaining accessible gaps in formal_category.

Covered lines:
  formal_category.py : 1031, 1036, 1667, 1673, 2700
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


# ===========================================================================
# Helpers
# ===========================================================================

def _make_single_obj_cat():
    from topos_ai.formal_category import FiniteCategory
    return FiniteCategory(
        objects=("A",),
        morphisms={"id_A": ("A", "A")},
        identities={"A": "id_A"},
        composition={("id_A", "id_A"): "id_A"},
    )


def _make_walking_arrow_cat():
    from topos_ai.formal_category import FiniteCategory
    return FiniteCategory(
        objects=("0", "1"),
        morphisms={"id0": ("0", "0"), "id1": ("1", "1"), "f": ("0", "1")},
        identities={"0": "id0", "1": "id1"},
        composition={
            ("id0", "id0"): "id0", ("id1", "id1"): "id1",
            ("f",   "id0"): "f",  ("id1", "f"):   "f",
        },
    )


def _left_kan_setup():
    """Minimal functor and presheaves for left Kan adjunction tests.

    validate_left_kan_adjunction(functor, source_presheaf, target_presheaf):
      source_presheaf lives on functor.source (cat_src)
      target_presheaf lives on functor.target = self.category (cat_dst)
    """
    from topos_ai.formal_category import (
        FiniteFunctor, Presheaf, PresheafTopos,
    )
    cat_src = _make_single_obj_cat()
    cat_dst = _make_single_obj_cat()
    functor = FiniteFunctor(
        source=cat_src, target=cat_dst,
        object_map={"A": "A"},
        morphism_map={"id_A": "id_A"},
    )
    topos = PresheafTopos(cat_dst)
    src_presheaf = Presheaf(cat_src, {"A": {"s"}}, {"id_A": {"s": "s"}})
    tgt_presheaf = Presheaf(cat_dst, {"A": {"t"}}, {"id_A": {"t": "t"}})
    return topos, functor, src_presheaf, tgt_presheaf


def _right_kan_setup():
    """Minimal functor and presheaves for right Kan adjunction tests.

    validate_right_kan_adjunction(functor, source_presheaf, target_presheaf):
      source_presheaf lives on functor.target = self.category (cat_dst)
      target_presheaf lives on functor.source (cat_src)
    """
    from topos_ai.formal_category import (
        FiniteFunctor, Presheaf, PresheafTopos,
    )
    cat_src = _make_single_obj_cat()
    cat_dst = _make_single_obj_cat()
    functor = FiniteFunctor(
        source=cat_src, target=cat_dst,
        object_map={"A": "A"},
        morphism_map={"id_A": "id_A"},
    )
    topos = PresheafTopos(cat_dst)
    # source_presheaf must live on functor.target (cat_dst): reindex_presheaf check
    src_presheaf = Presheaf(cat_dst, {"A": {"s"}}, {"id_A": {"s": "s"}})
    # target_presheaf must live on functor.source (cat_src): right_kan check
    tgt_presheaf = Presheaf(cat_src, {"A": {"t"}}, {"id_A": {"t": "t"}})
    return topos, functor, src_presheaf, tgt_presheaf


# ===========================================================================
# formal_category.py — lines 1031, 1036: validate_right_kan_adjunction
# ===========================================================================

class TestValidateRightKanAdjunction:
    def test_untranspose_wrong_returns_false_1031(self):
        """Line 1031: mock right_kan_untranspose to return wrong components."""
        topos, functor, src, tgt = _right_kan_setup()
        wrong = MagicMock()
        wrong.components = {"A": {"WRONG": "VALUE"}}
        with patch.object(topos, "right_kan_untranspose", return_value=wrong):
            result = topos.validate_right_kan_adjunction(functor, src, tgt)
        assert result is False  # line 1031

    def test_transpose_wrong_on_second_call_returns_false_1036(self):
        """Line 1036: first loop (1031) passes naturally; mock right_kan_transpose
        on 2nd call so round-trip fails → line 1036."""
        topos, functor, src, tgt = _right_kan_setup()
        call_count = [0]
        orig_transpose = topos.right_kan_transpose
        wrong = MagicMock()
        wrong.components = {"A": {"WRONG": "VALUE"}}

        def patched_transpose(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                return orig_transpose(*args, **kwargs)
            return wrong

        with patch.object(topos, "right_kan_transpose", side_effect=patched_transpose):
            result = topos.validate_right_kan_adjunction(functor, src, tgt)
        assert result is False  # line 1031 or 1036


# ===========================================================================
# formal_category.py — lines 1667, 1673: validate_exponential_adjunction
# ===========================================================================

class TestValidateExponentialAdjunction:
    def _exp_setup(self):
        from topos_ai.formal_category import Presheaf, PresheafTopos
        cat = _make_single_obj_cat()
        topos = PresheafTopos(cat)
        domain  = Presheaf(cat, {"A": {"d"}}, {"id_A": {"d": "d"}})
        exponent = Presheaf(cat, {"A": {"e"}}, {"id_A": {"e": "e"}})
        base     = Presheaf(cat, {"A": {"b"}}, {"id_A": {"b": "b"}})
        return topos, domain, exponent, base

    def test_untranspose_wrong_returns_false_1667(self):
        """Line 1667: mock untranspose to return wrong recovered on first call."""
        topos, domain, exponent, base = self._exp_setup()
        call_count = [0]
        orig_untranspose = topos.untranspose
        wrong = MagicMock()
        wrong.components = {"A": {"WRONG": "VALUE"}}

        def patched_untranspose(*args, **kwargs):
            call_count[0] += 1
            product, real_recovered = orig_untranspose(*args, **kwargs)
            if call_count[0] == 1:
                return product, wrong  # wrong → 1667 fires
            return product, real_recovered

        with patch.object(topos, "untranspose", side_effect=patched_untranspose):
            result = topos.validate_exponential_adjunction(domain, exponent, base)
        assert result is False  # line 1667

    def test_transpose_wrong_returns_false_1673(self):
        """Line 1673: let first loop (1667) pass, then mock transpose on 2nd-loop call."""
        topos, domain, exponent, base = self._exp_setup()
        call_count = [0]
        orig_transpose = topos.transpose
        wrong = MagicMock()
        wrong.components = {"A": {"WRONG": "VALUE"}}

        def patched_transpose(*args, **kwargs):
            call_count[0] += 1
            power, real_recovered = orig_transpose(*args, **kwargs)
            # First call is inside the FIRST loop (for line 1667 path) → let it pass.
            # Second call is inside the SECOND loop (for line 1673 path) → return wrong.
            if call_count[0] <= 1:
                return power, real_recovered
            return power, wrong

        with patch.object(topos, "transpose", side_effect=patched_transpose):
            result = topos.validate_exponential_adjunction(domain, exponent, base)
        assert result is False  # line 1667 or 1673


# ===========================================================================
# formal_category.py — line 2700: GrothendieckTopology transitivity violation
# ===========================================================================

class TestGrothendieckTopologyTransitivity:
    def test_transitivity_violation_raises_2700(self):
        """Line 2700: patch PresheafTopos.pullback_sieve at class level.

        When the class-level attribute is replaced by a Mock, calling
        instance.pullback_sieve(morphism, sieve) passes (morphism, sieve) directly
        to the side_effect (no self).  We save a bound method of a temporary
        PresheafTopos instance before patching to call the real implementation for
        non-empty sieves.  For the empty sieve we return a covering sieve, making
        pullbacks_cover=True for a non-covering sieve → line 2700 fires.
        """
        from topos_ai.formal_category import (
            PresheafTopos, GrothendieckTopology,
        )
        cat = _make_walking_arrow_cat()

        # Save a bound method to the original implementation before patching.
        temp_topos = PresheafTopos(cat)
        original_bound = temp_topos.pullback_sieve   # captured before patch is active

        def patched_pullback(morphism, sieve):
            # class-level patch: 'self' is NOT passed; only (morphism, sieve) are.
            sieve_fs = frozenset(sieve)
            if sieve_fs == frozenset():
                # Return a covering sieve for every morphism → pullbacks_cover=True
                # while the empty sieve is not in covering_sieves → line 2700.
                src = cat.source(morphism)
                return frozenset({"id0"}) if src == "0" else frozenset({"id1", "f"})
            # Delegate to the real implementation via the saved bound method.
            return original_bound(morphism, sieve)

        with patch.object(PresheafTopos, "pullback_sieve", side_effect=patched_pullback):
            with pytest.raises(ValueError, match="transitivity"):
                GrothendieckTopology(
                    cat,
                    {"0": [frozenset({"id0"})],
                     "1": [frozenset({"id1", "f"})]},
                )
