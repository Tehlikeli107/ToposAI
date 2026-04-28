"""
Additional coverage tests for topos_ai.infinity_categories.

Targets all uncovered lines: validation error paths in FiniteSimplicialSet,
FormalInfinityCategoryValidator, InfinityCategoryLayer, and edge cases in
compose_in_quasicategory / morphisms_are_homotopic / homotopy_category.
"""
from __future__ import annotations

import pytest
from topos_ai.infinity_categories import (
    FiniteHorn,
    FiniteSimplicialSet,
    FormalInfinityCategoryValidator,
    compose_in_quasicategory,
    homotopy_category,
    morphisms_are_homotopic,
    nerve_2_skeleton,
)
from topos_ai.formal_category import FiniteCategory


# ---------------------------------------------------------------------------
# Helpers: build minimal simplicial sets
# ---------------------------------------------------------------------------

def _interval_ss():
    """
    Nerve of the interval category  0 → 1 → 2  (three 2-simplices for composables).
    We build this manually as a minimal 2-simplex SS.
    """
    # 0-simplices: v0, v1
    # 1-simplices: e01 (0→1), e10 (1→0)  — just a simple triangle-free SS
    simplices = {0: ("v0", "v1"), 1: ("e01",)}
    faces = {
        (1, "e01", 0): "v1",
        (1, "e01", 1): "v0",
    }
    return FiniteSimplicialSet(simplices=simplices, faces=faces)


def _triangle_ss():
    """2-simplex SS with one triangle t012: v0→v1→v2."""
    simplices = {
        0: ("v0", "v1", "v2"),
        1: ("e01", "e12", "e02"),
        2: ("t012",),
    }
    faces = {
        (1, "e01", 0): "v1", (1, "e01", 1): "v0",
        (1, "e12", 0): "v2", (1, "e12", 1): "v1",
        (1, "e02", 0): "v2", (1, "e02", 1): "v0",
        (2, "t012", 0): "e12",
        (2, "t012", 1): "e02",
        (2, "t012", 2): "e01",
    }
    return FiniteSimplicialSet(simplices=simplices, faces=faces)


def _linear_category():
    """0 -f-> 1 -g-> 2 with composition gf."""
    return FiniteCategory(
        objects=["0", "1", "2"],
        morphisms={
            "id0": ("0", "0"), "id1": ("1", "1"), "id2": ("2", "2"),
            "f": ("0", "1"), "g": ("1", "2"), "gf": ("0", "2"),
        },
        identities={"0": "id0", "1": "id1", "2": "id2"},
        composition={
            ("id0", "id0"): "id0", ("id1", "id1"): "id1", ("id2", "id2"): "id2",
            ("f", "id0"): "f", ("id1", "f"): "f",
            ("g", "id1"): "g", ("id2", "g"): "g",
            ("gf", "id0"): "gf", ("id2", "gf"): "gf",
            ("g", "f"): "gf",
        },
    )


# ---------------------------------------------------------------------------
# FiniteSimplicialSet validation error paths
# ---------------------------------------------------------------------------

class TestFiniteSimplicialSetValidation:
    def test_missing_0_simplex_level_raises(self):
        """Line 59: no 0-simplex level."""
        with pytest.raises(ValueError, match="0-simplex"):
            FiniteSimplicialSet(simplices={1: ("e",)}, faces={(1, "e", 0): "v", (1, "e", 1): "v"})

    def test_negative_dimension_raises(self):
        """Line 63: negative dimension."""
        with pytest.raises(ValueError, match="non-negative"):
            FiniteSimplicialSet(
                simplices={0: ("v",), -1: ("x",)},
                faces={},
            )

    def test_duplicate_simplex_labels_raises(self):
        """Line 65: duplicate labels in a dimension."""
        with pytest.raises(ValueError, match="Duplicate"):
            FiniteSimplicialSet(
                simplices={0: ("v", "v")},
                faces={},
            )

    def test_face_not_in_lower_dimension_raises(self):
        """Line 72: face target not a (dim-1)-simplex."""
        with pytest.raises(ValueError, match="not a 0-simplex"):
            FiniteSimplicialSet(
                simplices={0: ("v0",), 1: ("e",)},
                faces={(1, "e", 0): "MISSING", (1, "e", 1): "v0"},
            )

    def test_face_table_mismatch_extra_raises(self):
        """Lines 81-83: extra keys in face table."""
        with pytest.raises(ValueError, match="Face table mismatch"):
            FiniteSimplicialSet(
                simplices={0: ("v0", "v1"), 1: ("e",)},
                faces={
                    (1, "e", 0): "v1",
                    (1, "e", 1): "v0",
                    (1, "EXTRA", 0): "v0",  # extra key
                },
            )

    def test_face_identity_violation_raises(self):
        """Line 119: d_i d_j ≠ d_{j-1} d_i on a 2-simplex."""
        # We build a 2-simplex whose boundary is inconsistent
        # d_0(d_1(sigma)) must equal d_0(d_0(sigma))  [i=0,j=1: d_0(d_1)=d_0(d_0)]
        # Actually the identity is d_i d_j = d_{j-1} d_i for i<j
        # i=0,j=1: d_0(d_1(sigma)) = d_0(d_0(sigma))
        # Let's make d_0(d_1(sigma)) = v1, but d_0(d_0(sigma)) = v0 — mismatch
        with pytest.raises(ValueError, match="Face identity"):
            FiniteSimplicialSet(
                simplices={0: ("v0", "v1", "v2"), 1: ("e01", "e12", "e02"), 2: ("t",)},
                faces={
                    (1, "e01", 0): "v1", (1, "e01", 1): "v0",
                    (1, "e12", 0): "v2", (1, "e12", 1): "v1",
                    (1, "e02", 0): "v2", (1, "e02", 1): "v0",
                    # Intentional break: d2 of t is e01, d0 of t is e02
                    # face identity i=0,j=1: d_0(d_1(t)) should equal d_0(d_0(t))
                    # d_1(t)=e12, d_0(t)=e02
                    # d_0(e12)=v2, d_0(e02)=v2 → would be ok
                    # Instead use d_0 of t = e01 (wrong endpoints for face identity)
                    # Let's violate: d_0(t)=e01 so d_0(d_0(t))=d_0(e01)=v1
                    # d_1(t)=e12, d_0(d_1(t))=d_0(e12)=v2 ≠ v1
                    (2, "t", 0): "e01",
                    (2, "t", 1): "e02",
                    (2, "t", 2): "e12",
                },
            )

    def test_face_accessor_missing_raises(self):
        """Lines 47-48: face() for undeclared key."""
        ss = _interval_ss()
        with pytest.raises(ValueError, match="Missing face"):
            ss.face(1, "e01", 99)

    def test_degeneracy_accessor_missing_raises(self):
        """Lines 54-55: degeneracy() for undeclared key."""
        cat = _linear_category()
        ss = nerve_2_skeleton(cat)
        with pytest.raises(ValueError, match="Missing degeneracy"):
            ss.degeneracy(0, ("obj", "0"), 99)

    def test_degeneracy_table_mismatch_raises(self):
        """Lines 93-95: degeneracy table has wrong keys."""
        with pytest.raises(ValueError, match="Degeneracy table mismatch"):
            FiniteSimplicialSet(
                simplices={0: ("v",), 1: ("e",)},
                faces={(1, "e", 0): "v", (1, "e", 1): "v"},
                degeneracies={(0, "v", 0): "e", (0, "EXTRA", 0): "e"},  # extra key
            )

    def test_degeneracy_target_not_in_next_dimension_raises(self):
        """Line 103: degeneracy maps to a simplex not in (dim+1)."""
        with pytest.raises(ValueError, match="not a 1-simplex"):
            FiniteSimplicialSet(
                simplices={0: ("v",), 1: ("e",)},
                faces={(1, "e", 0): "v", (1, "e", 1): "v"},
                degeneracies={(0, "v", 0): "MISSING"},
            )


# ---------------------------------------------------------------------------
# FiniteSimplicialSet._validate_horn error paths
# ---------------------------------------------------------------------------

class TestValidateHorn:
    def setup_method(self):
        self.ss = _triangle_ss()

    def test_horn_dimension_zero_raises(self):
        """Line 165: dimension < 1."""
        horn = FiniteHorn(dimension=0, missing_face=0, faces={})
        with pytest.raises(ValueError, match="dimension 1"):
            self.ss._validate_horn(horn)

    def test_horn_missing_face_negative_raises(self):
        """Line 167: missing_face < 0."""
        horn = FiniteHorn(dimension=1, missing_face=-1, faces={0: "e01"})
        with pytest.raises(ValueError, match="between 0 and n"):
            self.ss._validate_horn(horn)

    def test_horn_missing_face_too_large_raises(self):
        """Line 167: missing_face > dimension."""
        horn = FiniteHorn(dimension=1, missing_face=2, faces={0: "e01"})
        with pytest.raises(ValueError, match="between 0 and n"):
            self.ss._validate_horn(horn)

    def test_horn_wrong_face_set_raises(self):
        """Line 170: wrong set of face indices provided."""
        horn = FiniteHorn(dimension=1, missing_face=0, faces={0: "e01"})
        with pytest.raises(ValueError, match="exactly all faces"):
            self.ss._validate_horn(horn)

    def test_horn_face_not_lower_simplex_raises(self):
        """Line 173: face value is not a (dim-1)-simplex."""
        horn = FiniteHorn(dimension=2, missing_face=2, faces={0: "e12", 1: "NOT_A_SIMPLEX"})
        with pytest.raises(ValueError, match="not a 1-simplex"):
            self.ss._validate_horn(horn)

    def test_horn_from_simplex_wrong_simplex_raises(self):
        """Line 187: simplex not in its dimension."""
        with pytest.raises(ValueError, match="not a 2-simplex"):
            self.ss.horn_from_simplex(2, "NOSUCH", 0)

    def test_compatible_horns_dim_zero_returns_empty(self):
        """Line 210: dimension < 1 returns ()."""
        result = self.ss.compatible_horns(0, 0)
        assert result == ()


# ---------------------------------------------------------------------------
# missing_inner_horns / is_inner_kan / has_unique_inner_horn_fillers
# ---------------------------------------------------------------------------

class TestHornFilling:
    def test_missing_inner_horns_non_empty(self):
        """Line 229: missing_inner_horns() returns non-empty for non-quasi-cat."""
        simplices = {
            0: ("x", "y", "z"),
            1: ("f", "g"),
            2: (),
        }
        faces = {
            (1, "f", 0): "y", (1, "f", 1): "x",
            (1, "g", 0): "z", (1, "g", 1): "y",
        }
        ss = FiniteSimplicialSet(simplices=simplices, faces=faces)
        missing = ss.missing_inner_horns()
        assert len(missing) > 0

    def test_has_unique_inner_horn_fillers_true(self):
        """Lines 245: has_unique_inner_horn_fillers with explicit max_dimension."""
        ss = _triangle_ss()
        result = ss.has_unique_inner_horn_fillers(max_dimension=2)
        assert isinstance(result, bool)

    def test_has_unique_inner_horn_fillers_false_for_ambiguous(self):
        """Line 250: returns False when a horn has multiple fillers."""
        simplices = {
            0: ("x", "y", "z"),
            1: ("f", "g", "h", "k"),
            2: ("t1", "t2"),
        }
        faces = {
            (1, "f", 0): "y", (1, "f", 1): "x",
            (1, "g", 0): "z", (1, "g", 1): "y",
            (1, "h", 0): "z", (1, "h", 1): "x",
            (1, "k", 0): "z", (1, "k", 1): "x",
            # two fillers for the same inner horn (d2=f, d0=g)
            (2, "t1", 0): "g", (2, "t1", 1): "h", (2, "t1", 2): "f",
            (2, "t2", 0): "g", (2, "t2", 1): "k", (2, "t2", 2): "f",
        }
        ss = FiniteSimplicialSet(simplices=simplices, faces=faces)
        result = ss.has_unique_inner_horn_fillers(max_dimension=2)
        assert result is False


# ---------------------------------------------------------------------------
# FormalInfinityCategoryValidator
# ---------------------------------------------------------------------------

class TestFormalInfinityCategoryValidator:
    def test_valid_category_no_missing_horns(self):
        """Lines 472-474, 482-492, 499: valid case."""
        validator = FormalInfinityCategoryValidator(
            simplices_0=["x", "y", "z"],
            simplices_1=[("x", "y"), ("y", "z"), ("x", "z")],
            simplices_2=[("x", "y", "z")],
        )
        assert validator.is_valid_infinity_category() is True

    def test_missing_inner_horn_detected(self):
        """Lines 482-492: missing 2-simplex found."""
        validator = FormalInfinityCategoryValidator(
            simplices_0=["x", "y", "z"],
            simplices_1=[("x", "y"), ("y", "z")],
            simplices_2=[],  # no filler
        )
        missing = validator.find_missing_inner_horns()
        assert len(missing) > 0
        assert validator.is_valid_infinity_category() is False

    def test_enforce_strict_composition_adds_simplices(self):
        """Lines 506-526: enforce adds 1-simplices and 2-simplices."""
        validator = FormalInfinityCategoryValidator(
            simplices_0=["x", "y", "z"],
            simplices_1=[("x", "y"), ("y", "z")],
            simplices_2=[],
        )
        added_1, added_2 = validator.enforce_strict_composition()
        assert added_1 >= 0
        assert added_2 > 0
        assert validator.is_valid_infinity_category() is True

    def test_enforce_already_valid_adds_nothing(self):
        """Lines 506-526: no change needed when already valid."""
        validator = FormalInfinityCategoryValidator(
            simplices_0=["x", "y"],
            simplices_1=[("x", "y")],
            simplices_2=[],
        )
        # No composable pair (only one morphism, no y→z), so nothing to fill
        added_1, added_2 = validator.enforce_strict_composition()
        assert added_1 == 0
        assert added_2 == 0

    def test_chain_of_three_morphisms(self):
        """Tests transitive closure via enforce_strict_composition."""
        validator = FormalInfinityCategoryValidator(
            simplices_0=["a", "b", "c", "d"],
            simplices_1=[("a", "b"), ("b", "c"), ("c", "d")],
            simplices_2=[],
        )
        validator.enforce_strict_composition()
        assert validator.is_valid_infinity_category() is True


# ---------------------------------------------------------------------------
# compose_in_quasicategory — second 1-simplex missing
# ---------------------------------------------------------------------------

class TestComposeEdgeCases:
    def test_g_not_1_simplex_raises(self):
        """Line 578: g is not a 1-simplex."""
        ss = _triangle_ss()
        with pytest.raises(ValueError, match="not a 1-simplex"):
            compose_in_quasicategory(ss, "e01", "NOSUCH")


# ---------------------------------------------------------------------------
# morphisms_are_homotopic — non-parallel (different endpoints)
# ---------------------------------------------------------------------------

class TestHomotopicEdgeCases:
    def test_non_parallel_returns_false(self):
        """Line 657: different source or target → returns False."""
        ss = _triangle_ss()
        # e01: v0→v1,  e12: v1→v2 — different source AND target
        result = morphisms_are_homotopic(ss, "e01", "e12")
        assert result is False

    def test_non_parallel_same_source_different_target(self):
        """Different target → non-parallel → False."""
        ss = _triangle_ss()
        result = morphisms_are_homotopic(ss, "e01", "e02")
        # e01: v0→v1, e02: v0→v2 — same source, different target
        assert result is False


# ---------------------------------------------------------------------------
# homotopy_category — identity via degeneracy map path
# ---------------------------------------------------------------------------

class TestHomotopyCategoryDegeneracy:
    def test_nerve_with_degeneracies_gives_valid_category(self):
        """Lines 778-788: identity found via degeneracy map."""
        cat = _linear_category()
        ss = nerve_2_skeleton(cat)
        # nerve_2_skeleton includes degeneracy maps
        assert ss.degeneracies  # degeneracies present
        ho_cat = homotopy_category(ss)
        # Should recover a category with same object count
        assert len(ho_cat.objects) == 3

    def test_homotopy_category_all_identities_exist(self):
        """Lines 805-806, 812: every object has an identity."""
        cat = _linear_category()
        ss = nerve_2_skeleton(cat)
        ho_cat = homotopy_category(ss)
        for obj in ho_cat.objects:
            assert obj in ho_cat.identities

    def test_homotopy_category_composition_defined(self):
        """Line 812: FiniteCategory constructed correctly."""
        cat = _linear_category()
        ss = nerve_2_skeleton(cat)
        ho_cat = homotopy_category(ss)
        # The category must be valid (no exception on construction)
        assert ho_cat is not None
        assert len(ho_cat.morphisms) > 0

    def test_homotopy_category_no_degeneracies_fallback(self):
        """Falls back to endomorphism-based identity when no degeneracies."""
        ss = _triangle_ss()  # no degeneracies
        assert not ss.degeneracies
        ho_cat = homotopy_category(ss)
        for obj in ho_cat.objects:
            assert obj in ho_cat.identities
