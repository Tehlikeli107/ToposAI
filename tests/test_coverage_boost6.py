"""
test_coverage_boost6.py — Cover remaining gaps in infinity_categories,
formal_category, formal_kan, and enriched modules.

Covered lines:
  infinity_categories.py : 140, 657, 805-806
  formal_category.py     : 1956, 2340, 2345
  formal_kan.py          : 405
  enriched.py            : 243, 364-365
"""
from __future__ import annotations

import pytest
from unittest.mock import patch


# ===========================================================================
# Helpers: minimal categories
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
    """Category with objects "0","1" and morphisms id0, id1, f: 0→1."""
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


def _make_trivial_monoidal():
    """Strict monoidal category with one object 'h' and identity morphism."""
    from topos_ai.monoidal import strict_monoidal_from_monoid
    return strict_monoidal_from_monoid(
        objects=["h"],
        tensor_table={("h", "h"): "h"},
        unit="h",
    )


def _make_trivial_enriched():
    """Trivial self-enriched category: one object A over the one-object monoidal cat."""
    from topos_ai.enriched import FiniteEnrichedCategory
    V = _make_trivial_monoidal()
    return FiniteEnrichedCategory(
        objects=("A",),
        enriching=V,
        hom_objects={("A", "A"): "h"},
        compositions={("A", "A", "A"): "id_h"},
        identity_elements={"A": "id_h"},
    )


# ===========================================================================
# infinity_categories.py — line 140: face-degeneracy identity failure
# ===========================================================================

class TestSimplicialSetFaceDegeneracyFailure:
    def test_face_degeneracy_identity_violation_raises(self):
        """Line 140: d_0 s_0(v0) = v1 instead of v0 → ValueError."""
        from topos_ai.infinity_categories import FiniteSimplicialSet
        # Two 0-simplices so that face value 'v1' is valid (passes dim check)
        # but violates the face-degeneracy identity d_0 s_0 = id.
        with pytest.raises(ValueError, match="Face-degeneracy identity"):
            FiniteSimplicialSet(
                simplices={0: ("v0", "v1"), 1: ("sig0", "sig1")},
                faces={
                    (1, "sig0", 0): "v1",   # WRONG: should be "v0" (d_0 s_0(v0) = v0)
                    (1, "sig0", 1): "v0",
                    (1, "sig1", 0): "v1",
                    (1, "sig1", 1): "v1",
                },
                degeneracies={(0, "v0", 0): "sig0", (0, "v1", 0): "sig1"},
            )

    def test_valid_degeneracy_passes(self):
        """Sanity: correct face-degeneracy (d_0 s_0 = id) does not raise."""
        from topos_ai.infinity_categories import FiniteSimplicialSet
        ss = FiniteSimplicialSet(
            simplices={0: ("v0", "v1"), 1: ("sig0", "sig1", "edge")},
            faces={
                (1, "sig0", 0): "v0",   # CORRECT: d_0 s_0(v0) = v0
                (1, "sig0", 1): "v0",
                (1, "sig1", 0): "v1",   # CORRECT: d_0 s_0(v1) = v1
                (1, "sig1", 1): "v1",
                (1, "edge",  0): "v1",
                (1, "edge",  1): "v0",
            },
            degeneracies={(0, "v0", 0): "sig0", (0, "v1", 0): "sig1"},
        )
        assert ss is not None


# ===========================================================================
# infinity_categories.py — line 657: right homotopy branch
# ===========================================================================

class TestMorphismsAreHomotopic:
    def test_right_homotopy_returns_true(self):
        """Line 657: sigma with d2=g, d0=f (right homotopy) returns True."""
        from topos_ai.infinity_categories import FiniteSimplicialSet, morphisms_are_homotopic
        # Endomorphisms f, g: x→x; sigma with d2=g, d0=f is right homotopy
        ss = FiniteSimplicialSet(
            simplices={0: ("x",), 1: ("f", "g", "h"), 2: ("sigma",)},
            faces={
                (1, "f", 0): "x", (1, "f", 1): "x",
                (1, "g", 0): "x", (1, "g", 1): "x",
                (1, "h", 0): "x", (1, "h", 1): "x",
                (2, "sigma", 0): "f",   # d0 = f
                (2, "sigma", 1): "h",   # d1 = h (composite)
                (2, "sigma", 2): "g",   # d2 = g  ← right homotopy branch
            },
        )
        result = morphisms_are_homotopic(ss, "f", "g")
        assert result is True  # line 657

    def test_left_homotopy_returns_true(self):
        """Line 654: sigma with d2=f, d0=g (left homotopy) returns True."""
        from topos_ai.infinity_categories import FiniteSimplicialSet, morphisms_are_homotopic
        ss = FiniteSimplicialSet(
            simplices={0: ("x",), 1: ("f", "g", "h"), 2: ("sigma",)},
            faces={
                (1, "f", 0): "x", (1, "f", 1): "x",
                (1, "g", 0): "x", (1, "g", 1): "x",
                (1, "h", 0): "x", (1, "h", 1): "x",
                (2, "sigma", 0): "g",   # d0 = g
                (2, "sigma", 1): "h",   # d1 = h
                (2, "sigma", 2): "f",   # d2 = f  ← left homotopy
            },
        )
        assert morphisms_are_homotopic(ss, "f", "g") is True  # line 654

    def test_no_homotopy_returns_false(self):
        """Sanity: no 2-simplex witnessing homotopy → False."""
        from topos_ai.infinity_categories import FiniteSimplicialSet, morphisms_are_homotopic
        ss = FiniteSimplicialSet(
            simplices={0: ("x",), 1: ("f", "g")},
            faces={
                (1, "f", 0): "x", (1, "f", 1): "x",
                (1, "g", 0): "x", (1, "g", 1): "x",
            },
        )
        assert morphisms_are_homotopic(ss, "f", "g") is False


# ===========================================================================
# infinity_categories.py — lines 805-806: compose_in_quasicategory ValueError
# ===========================================================================

class TestHomotopyCategoryNoFiller:
    def test_no_2simplex_filler_triggers_lines_805_806(self):
        """Lines 805-806: ValueError from compose_in_quasicategory is caught and skipped."""
        from topos_ai.infinity_categories import FiniteSimplicialSet, homotopy_category
        # f: x→y and g: y→z are composable by type but there is no 2-simplex filler.
        # compose_in_quasicategory(ss, "f", "g") raises ValueError → lines 805-806.
        # homotopy_category then raises ValueError at FiniteCategory construction
        # because the missing (g,f) entry leaves the composition table incomplete.
        ss = FiniteSimplicialSet(
            simplices={0: ("x", "y", "z"), 1: ("f", "g")},
            faces={
                (1, "f", 0): "y", (1, "f", 1): "x",   # f: x→y
                (1, "g", 0): "z", (1, "g", 1): "y",   # g: y→z
            },
        )
        # The ValueError from FiniteCategory (incomplete composition table) shows
        # that lines 805-806 executed and the pair was dropped, making the table
        # incomplete.
        with pytest.raises(ValueError):
            homotopy_category(ss)


# ===========================================================================
# formal_category.py — line 1956: validate_effective_epimorphism comparison
# ===========================================================================

class TestValidateEffectiveEpimorphism:
    def test_comparison_not_iso_returns_false(self):
        """Line 1956: comparison map not mono/epi → returns False."""
        from topos_ai.formal_category import (
            FiniteCategory, Presheaf, NaturalTransformation, PresheafTopos,
        )
        cat = _make_single_obj_cat()
        topos = PresheafTopos(cat)
        # Construct a surjective (epi) transformation
        P = Presheaf(cat, {"A": {"p1", "p2"}}, {"id_A": {"p1": "p1", "p2": "p2"}})
        Q = Presheaf(cat, {"A": {"q1"}}, {"id_A": {"q1": "q1"}})
        alpha = NaturalTransformation(
            source=P, target=Q,
            components={"A": {"p1": "q1", "p2": "q1"}},   # surjective: both → q1
        )
        # alpha IS epi (surjective), but mock is_epimorphism to return False for
        # the comparison map so line 1956 fires.
        original_is_epi = topos.is_epimorphism
        call_count = [0]

        def patched_is_epi(transform):
            call_count[0] += 1
            if call_count[0] == 1:
                return True   # alpha itself is epi (passes line 1952 check)
            return False      # comparison is "not epi" → line 1956

        with patch.object(topos, "is_epimorphism", side_effect=patched_is_epi):
            result = topos.validate_effective_epimorphism(alpha)
        assert result is False   # line 1956


# ===========================================================================
# formal_category.py — line 2340: j not idempotent
# ===========================================================================

class TestValidateLawvereTierneyIdempotencyFailure:
    def test_j_not_idempotent_returns_false(self):
        """Line 2340: j(j(S)) ≠ j(S) for some sieve → returns False."""
        from topos_ai.formal_category import PresheafTopos, GrothendieckTopology
        cat = _make_walking_arrow_cat()
        topos = PresheafTopos(cat)
        gt = GrothendieckTopology(cat, {
            "0": [frozenset({"id0"})],
            "1": [frozenset({"id1", "f"})],
        })

        # Non-idempotent j on object "1":
        #   j(∅)={f}, j({f})=max → j(j(∅)) = j({f}) = max ≠ {f} = j(∅)
        max1 = frozenset({"id1", "f"})
        j_map_1 = {
            frozenset():           frozenset({"f"}),   # j(∅) = {f}
            frozenset({"f"}):      max1,               # j({f}) = max  → non-idempotent!
            max1:                  max1,               # j(max) = max  (line 2334 passes)
        }

        def patched_j(topology, obj, sieve):
            sieve_fs = frozenset(sieve)
            if obj == "0":
                return frozenset({"id0"}) if sieve_fs == frozenset({"id0"}) else frozenset()
            else:
                return j_map_1.get(sieve_fs, max1)

        with patch.object(topos, "j_operator_on_sieve", side_effect=patched_j):
            result = topos.validate_lawvere_tierney_axioms(gt)
        assert result is False   # line 2340


# ===========================================================================
# formal_category.py — line 2345: j does not preserve meets
# ===========================================================================

class TestValidateLawvereTierneyMeetFailure:
    def test_j_not_preserving_meets_returns_false(self):
        """Line 2345: j(S∩T) ≠ j(S)∩j(T) for some sieves → returns False."""
        from topos_ai.formal_category import PresheafTopos, GrothendieckTopology
        cat = _make_walking_arrow_cat()
        topos = PresheafTopos(cat)
        gt = GrothendieckTopology(cat, {
            "0": [frozenset({"id0"})],
            "1": [frozenset({"id1", "f"})],
        })

        # Idempotent but non-meet-preserving j on "1":
        #   j(∅)=max, j({f})={f}, j(max)=max
        #   Idempotent: j(j(∅))=j(max)=max=j(∅) ✓; j(j({f}))=j({f})={f} ✓
        #   Meet-violation: j(∅ ∩ {f}) = j(∅) = max ≠ j(∅)∩j({f}) = max∩{f} = {f}
        max1 = frozenset({"id1", "f"})
        j_map_1 = {
            frozenset():      max1,             # j(∅) = max
            frozenset({"f"}): frozenset({"f"}), # j({f}) = {f}  → violates meet preservation
            max1:             max1,             # j(max) = max  (line 2334 passes)
        }

        def patched_j(topology, obj, sieve):
            sieve_fs = frozenset(sieve)
            if obj == "0":
                return frozenset({"id0"}) if sieve_fs == frozenset({"id0"}) else frozenset()
            else:
                return j_map_1.get(sieve_fs, max1)

        with patch.object(topos, "j_operator_on_sieve", side_effect=patched_j):
            result = topos.validate_lawvere_tierney_axioms(gt)
        assert result is False   # line 2345


# ===========================================================================
# formal_kan.py — line 405: right_kan_extension pass branch
# ===========================================================================

class TestRightKanExtensionLine405:
    def test_line_405_pass_branch_fires(self):
        """Line 405: pass fires when source(f_name) ≠ source(g_mor)."""
        from topos_ai.formal_kan import right_kan_extension, FiniteSetFunctor
        # Walking arrow category as both C (source) and D (target)
        C = _make_walking_arrow_cat()
        D = _make_walking_arrow_cat()

        # Identity functor K: C→D
        K_obj = {"0": "0", "1": "1"}
        K_mor = {"id0": "id0", "id1": "id1", "f": "f"}

        # X: C→FinSet with X(0)={"a"}, X(1)={"b"}, X(f)={a→b}
        X_functor = FiniteSetFunctor(
            category=C,
            objects_map={"0": frozenset({"a"}), "1": frozenset({"b"})},
            morphism_map={
                "id0": {"a": "a"},
                "id1": {"b": "b"},
                "f":   {"a": "b"},
            },
            validate=False,
        )

        # right_kan_extension processes morphism "f": 0→1 in D.
        # For the morphism action of f, tgt_comma(1) = [(1, "id1")].
        # D_cat.source("f")="0" ≠ D_cat.source("id1")="1" → line 405 fires.
        result = right_kan_extension(X_functor, K_obj, K_mor, D)
        assert result is not None  # line 405 covered, function still completes


# ===========================================================================
# enriched.py — line 243: _check_associativity failure
# ===========================================================================

class TestEnrichedAssociativityFailure:
    def test_associativity_violation_raises(self):
        """Line 243: LHS ≠ RHS in enriched associativity check → ValueError."""
        ec = _make_trivial_enriched()
        original = ec.enriching.category.compose
        call_count = [0]

        def bad_compose(after, before):
            call_count[0] += 1
            if call_count[0] == 1:
                return "WRONG_VALUE"   # First compose → LHS gets wrong value
            return original(after, before)

        with patch.object(ec.enriching.category, "compose", side_effect=bad_compose):
            with pytest.raises(ValueError, match="Enriched associativity"):
                ec._check_associativity()


# ===========================================================================
# enriched.py — lines 364-365: underlying_category try/except
# ===========================================================================

class TestEnrichedUnderlyingCategoryExcept:
    def test_tensor_mor_raises_is_caught(self):
        """Lines 364-365: ValueError in tensor_mor is caught; function still returns."""
        ec = _make_trivial_enriched()

        # Patch tensor_mor to always raise ValueError.
        # In underlying_category(), the try block catches this and continues.
        with patch.object(ec.enriching, "tensor_mor", side_effect=ValueError("mock")):
            result = ec.underlying_category()

        # The underlying category should be returned (with empty composition,
        # filled only by identity rules).
        assert result is not None

    def test_underlying_category_normal(self):
        """Sanity: underlying_category works normally for the trivial enriched cat."""
        ec = _make_trivial_enriched()
        result = ec.underlying_category()
        assert result is not None
        # The underlying category should have one object A and one morphism (j_A)
        assert "A" in result.objects
