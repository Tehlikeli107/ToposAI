"""
test_coverage_boost8.py — Cover remaining accessible gaps in formal_category.

Covered lines:
  formal_category.py : 882, 887, 1447, 1449, 1466, 1800, 1808,
                       2310, 2596, 2601, 2606
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


def _make_trivial_setup():
    """Single-object category with trivial GT and simple presheaves."""
    from topos_ai.formal_category import (
        PresheafTopos, Presheaf, GrothendieckTopology,
        NaturalTransformation,
    )
    cat = _make_single_obj_cat()
    topos = PresheafTopos(cat)
    gt = GrothendieckTopology(cat, {"A": [frozenset({"id_A"})]})
    P = Presheaf(cat, {"A": {"p"}}, {"id_A": {"p": "p"}})
    Q = Presheaf(cat, {"A": {"q"}}, {"id_A": {"q": "q"}})
    alpha = NaturalTransformation(source=P, target=Q, components={"A": {"p": "q"}})
    return topos, cat, gt, P, Q, alpha


# ===========================================================================
# formal_category.py — lines 1447, 1449: validate_quantifier_adjunctions
# ===========================================================================

class TestValidateQuantifierAdjunctions:
    def test_exists_wrong_returns_false_1447(self):
        """Line 1447: mock exists_along to return top_Q so leq comparison fails."""
        topos, cat, gt, P, Q, alpha = _make_trivial_setup()
        top_Q = topos.subobject_top(Q)
        # With exists_source = top_Q and target_subobject = ⊥_Q:
        #   leq(top_Q, ⊥_Q) = False  but  leq(⊥_P, ⊥_P) = True  → False ≠ True → 1447
        with patch.object(topos, "exists_along", return_value=top_Q):
            result = topos.validate_quantifier_adjunctions(alpha)
        assert result is False  # line 1447

    def test_forall_wrong_returns_false_1449(self):
        """Line 1449: mock forall_along to return bottom_Q so second leq pair fails."""
        topos, cat, gt, P, Q, alpha = _make_trivial_setup()
        bottom_Q = topos.subobject_bottom(Q)
        # With forall_source = ⊥_Q and target_subobject = ⊤_Q:
        #   1447: leq(real_exists, ⊤_Q) == leq(⊤_P, pullback_of_⊤_Q) → passes
        #   1449: leq(pullback_of_⊤_Q=⊤_P, ⊤_P) = True but leq(⊤_Q, ⊥_Q) = False → 1449
        with patch.object(topos, "forall_along", return_value=bottom_Q):
            result = topos.validate_quantifier_adjunctions(alpha)
        assert result is False  # line 1449


# ===========================================================================
# formal_category.py — line 1466: validate_frobenius_reciprocity
# ===========================================================================

class TestValidateFrobeniusReciprocity:
    def test_exists_wrong_returns_false_1466(self):
        """Line 1466: mock exists_along to return top_Q so left.subsets ≠ right.subsets."""
        topos, cat, gt, P, Q, alpha = _make_trivial_setup()
        top_Q = topos.subobject_top(Q)
        bottom_Q = topos.subobject_bottom(Q)
        # left = exists_along(alpha, meet(⊥_P, ⊥_P)) = top_Q (mock)
        # right = subobject_meet(top_Q, ⊥_Q) = ⊥_Q
        # top_Q.subsets ≠ bottom_Q.subsets → 1466
        with patch.object(topos, "exists_along", return_value=top_Q):
            result = topos.validate_frobenius_reciprocity(alpha)
        assert result is False  # line 1466


# ===========================================================================
# formal_category.py — lines 1800, 1808: validate_j_subobject_heyting_laws
# ===========================================================================

class TestValidateJSubobjectHeytingLaws:
    def _setup(self):
        from topos_ai.formal_category import (
            PresheafTopos, Presheaf, GrothendieckTopology,
        )
        cat = _make_single_obj_cat()
        topos = PresheafTopos(cat)
        gt = GrothendieckTopology(cat, {"A": [frozenset({"id_A"})]})
        P = Presheaf(cat, {"A": {"p"}}, {"id_A": {"p": "p"}})
        return topos, P, gt

    def test_implication_not_j_closed_returns_false_1800(self):
        """Line 1800: patch j_closed_subobjects to return non-empty list,
        then patch is_j_closed_subobject to return False → line 1800."""
        topos, P, gt = self._setup()
        closed = topos.j_closed_subobjects(P, gt)   # pre-compute real list

        with patch.object(topos, "j_closed_subobjects", return_value=closed):
            with patch.object(topos, "is_j_closed_subobject", return_value=False):
                result = topos.validate_j_subobject_heyting_laws(P, gt)
        assert result is False  # line 1800

    def test_adjoint_inequality_returns_false_1808(self):
        """Line 1808: patch subobject_j_meet to return top_P so meet_below ≠ adjoint_below."""
        topos, P, gt = self._setup()
        closed = topos.j_closed_subobjects(P, gt)   # pre-compute real list
        top_P = topos.subobject_top(P)

        # meet_below = leq(top_P, consequent=⊥_P) = False
        # adjoint_below = leq(probe=⊥_P, implication(⊥,⊥)=⊤_P) = True
        # False ≠ True → line 1808
        with patch.object(topos, "j_closed_subobjects", return_value=closed):
            with patch.object(topos, "subobject_j_meet", return_value=top_P):
                result = topos.validate_j_subobject_heyting_laws(P, gt)
        assert result is False  # line 1808


# ===========================================================================
# formal_category.py — line 2310: j_operator_on_sieve invalid closed sieve
# ===========================================================================

class TestJOperatorInvalidSieve:
    def test_closed_not_sieve_raises_2310(self):
        """Line 2310: patch _is_sieve so 2nd call (on closed) returns False → ValueError."""
        from topos_ai.formal_category import PresheafTopos, GrothendieckTopology
        cat = _make_single_obj_cat()
        topos = PresheafTopos(cat)
        gt = GrothendieckTopology(cat, {"A": [frozenset({"id_A"})]})

        call_count = [0]
        orig_is_sieve = topos._is_sieve
        def patched_is_sieve(obj, sieve):
            call_count[0] += 1
            if call_count[0] == 1:
                return True   # first call: input sieve is valid
            return False      # second call: closed set is not a sieve → 2310

        with patch.object(topos, "_is_sieve", side_effect=patched_is_sieve):
            with pytest.raises(ValueError, match="did not produce a sieve"):
                topos.j_operator_on_sieve(gt, "A", frozenset({"id_A"}))


# ===========================================================================
# formal_category.py — lines 2596, 2601, 2606:
#   validate_subobject_classifier_universal_property
# ===========================================================================

class TestValidateSubobjectClassifierUniversalProperty:
    def _setup(self):
        from topos_ai.formal_category import PresheafTopos, Presheaf
        cat = _make_single_obj_cat()
        topos = PresheafTopos(cat)
        P = Presheaf(cat, {"A": {"p"}}, {"id_A": {"p": "p"}})
        return topos, P

    def test_subobject_names_mismatch_returns_false_2596(self):
        """Line 2596: patch _transformation_key so first call gives a fake key
        that is not in all_maps → subobject_names ≠ all_maps → returns False."""
        topos, P = self._setup()
        call_count = [0]
        orig_key = topos._transformation_key
        def patched_key(t):
            call_count[0] += 1
            if call_count[0] == 1:
                return "FAKE_KEY_ABSENT_FROM_ALL_MAPS"
            return orig_key(t)

        with patch.object(topos, "_transformation_key", side_effect=patched_key):
            result = topos.validate_subobject_classifier_universal_property(P)
        assert result is False  # line 2596

    def test_pullback_wrong_subsets_returns_false_2601(self):
        """Line 2601: mock pullback_truth to return ⊥_P → recovered.subsets ≠ ⊤_P.subsets."""
        topos, P = self._setup()
        bottom_P = topos.subobject_bottom(P)
        # 2596 passes (real characteristic_map = real all_maps, theorem holds).
        # Second iteration (subobject=⊤_P): recovered=⊥_P, subsets mismatch → 2601.
        with patch.object(topos, "pullback_truth", return_value=bottom_P):
            result = topos.validate_subobject_classifier_universal_property(P)
        assert result is False  # line 2601

    def test_characteristic_map_round_trip_fails_returns_false_2606(self):
        """Line 2606: let 2596+2601 pass, then mock pullback_truth in 3rd call (2606 loop)
        to return ⊥_P → characteristic_map(⊥_P).components ≠ classifier.components."""
        topos, P = self._setup()
        bottom_P = topos.subobject_bottom(P)
        call_count = [0]
        orig_pullback = topos.pullback_truth
        def patched_pullback(t):
            call_count[0] += 1
            if call_count[0] <= 2:   # first 2 calls: 2601 loop → let them be correct
                return orig_pullback(t)
            return bottom_P          # 3rd+ calls: 2606 loop → wrong recovered → 2606

        with patch.object(topos, "pullback_truth", side_effect=patched_pullback):
            result = topos.validate_subobject_classifier_universal_property(P)
        assert result is False  # line 2606


# ===========================================================================
# formal_category.py — lines 882, 887: validate_left_kan_adjunction
# ===========================================================================

class TestValidateLeftKanAdjunction:
    def _setup(self):
        from topos_ai.formal_category import (
            FiniteCategory, FiniteFunctor, Presheaf, PresheafTopos,
        )
        cat_src = _make_single_obj_cat()
        cat_dst = _make_single_obj_cat()   # separate Python object
        functor = FiniteFunctor(
            source=cat_src, target=cat_dst,
            object_map={"A": "A"},
            morphism_map={"id_A": "id_A"},
        )
        topos = PresheafTopos(cat_dst)
        src_presheaf = Presheaf(cat_src, {"A": {"s"}}, {"id_A": {"s": "s"}})
        tgt_presheaf = Presheaf(cat_dst, {"A": {"t"}}, {"id_A": {"t": "t"}})
        return topos, functor, src_presheaf, tgt_presheaf

    def test_untranspose_wrong_returns_false_882(self):
        """Line 882: mock left_kan_untranspose to return object with wrong .components."""
        topos, functor, src_presheaf, tgt_presheaf = self._setup()
        wrong = MagicMock()
        wrong.components = {"A": {"WRONG": "VALUE"}}
        with patch.object(topos, "left_kan_untranspose", return_value=wrong):
            result = topos.validate_left_kan_adjunction(functor, src_presheaf, tgt_presheaf)
        assert result is False  # line 882

    def test_transpose_wrong_on_second_call_returns_false_887(self):
        """Line 887: first loop uses correct transpose (line 882 passes naturally),
        then 2nd loop: mock left_kan_transpose to return wrong → line 887."""
        topos, functor, src_presheaf, tgt_presheaf = self._setup()
        call_count = [0]
        orig_transpose = topos.left_kan_transpose
        wrong = MagicMock()
        wrong.components = {"A": {"WRONG": "VALUE"}}

        def patched_transpose(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                # First call (inside 1st loop) → correct, so 882 passes
                return orig_transpose(*args, **kwargs)
            # Second call (inside 2nd loop) → wrong, triggers 887
            return wrong

        with patch.object(topos, "left_kan_transpose", side_effect=patched_transpose):
            result = topos.validate_left_kan_adjunction(functor, src_presheaf, tgt_presheaf)
        assert result is False  # line 882 or 887
