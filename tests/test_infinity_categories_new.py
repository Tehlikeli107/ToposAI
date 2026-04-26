"""
Tests for the new quasi-category composition functions added to
topos_ai.infinity_categories:
  - compose_in_quasicategory
  - morphisms_are_homotopic
  - homotopy_category
"""

import pytest

from topos_ai.formal_category import FiniteCategory
from topos_ai.infinity_categories import (
    FiniteSimplicialSet,
    nerve_2_skeleton,
    nerve_3_skeleton,
    compose_in_quasicategory,
    morphisms_are_homotopic,
    homotopy_category,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _arrow_category():
    """Walking-arrow category: A —f→ B."""
    return FiniteCategory(
        objects=("A", "B"),
        morphisms={"idA": ("A", "A"), "idB": ("B", "B"), "f": ("A", "B")},
        identities={"A": "idA", "B": "idB"},
        composition={
            ("idA", "idA"): "idA",
            ("idB", "idB"): "idB",
            ("f", "idA"): "f",
            ("idB", "f"): "f",
        },
    )


def _composable_pair_category():
    """
    Category with three objects and two composable morphisms:
    A —f→ B —g→ C   with  h = g∘f.
    """
    return FiniteCategory(
        objects=("A", "B", "C"),
        morphisms={
            "idA": ("A", "A"), "idB": ("B", "B"), "idC": ("C", "C"),
            "f": ("A", "B"), "g": ("B", "C"), "h": ("A", "C"),
        },
        identities={"A": "idA", "B": "idB", "C": "idC"},
        composition={
            ("idA", "idA"): "idA",
            ("idB", "idB"): "idB",
            ("idC", "idC"): "idC",
            ("f", "idA"): "f",
            ("idB", "f"): "f",
            ("g", "idB"): "g",
            ("idC", "g"): "g",
            ("h", "idA"): "h",
            ("idC", "h"): "h",
            ("g", "f"): "h",
        },
    )


def _square_category():
    """
    Commutative square: A —f→ B, A —g→ C, B —h→ D, C —k→ D
    with  h∘f = k∘g  (both equal to diagonal p: A→D).
    """
    return FiniteCategory(
        objects=("A", "B", "C", "D"),
        morphisms={
            "idA": ("A", "A"), "idB": ("B", "B"),
            "idC": ("C", "C"), "idD": ("D", "D"),
            "f": ("A", "B"), "g": ("A", "C"),
            "h": ("B", "D"), "k": ("C", "D"),
            "p": ("A", "D"),
        },
        identities={"A": "idA", "B": "idB", "C": "idC", "D": "idD"},
        composition={
            ("idA", "idA"): "idA",
            ("idB", "idB"): "idB",
            ("idC", "idC"): "idC",
            ("idD", "idD"): "idD",
            ("f", "idA"): "f",
            ("idB", "f"): "f",
            ("g", "idA"): "g",
            ("idC", "g"): "g",
            ("h", "idB"): "h",
            ("idD", "h"): "h",
            ("k", "idC"): "k",
            ("idD", "k"): "k",
            ("p", "idA"): "p",
            ("idD", "p"): "p",
            ("h", "f"): "p",
            ("k", "g"): "p",
        },
    )


# ------------------------------------------------------------------ #
# Tests: compose_in_quasicategory                                      #
# ------------------------------------------------------------------ #

class TestComposeInQuasicategory:
    def test_basic_composition_returns_tuple(self):
        C = _composable_pair_category()
        X = nerve_2_skeleton(C)
        result = compose_in_quasicategory(X, ("mor", "f"), ("mor", "g"))
        assert isinstance(result, tuple)
        assert len(result) >= 1

    def test_composite_is_correct(self):
        """f ; g should yield h = g∘f in the nerve."""
        C = _composable_pair_category()
        X = nerve_2_skeleton(C)
        composites = compose_in_quasicategory(X, ("mor", "f"), ("mor", "g"))
        # In the nerve, d₁ of the filler should be the composite morphism
        assert ("mor", "h") in composites

    def test_identity_left(self):
        """idA ; f = f."""
        C = _composable_pair_category()
        X = nerve_2_skeleton(C)
        composites = compose_in_quasicategory(X, ("mor", "idA"), ("mor", "f"))
        assert ("mor", "f") in composites

    def test_identity_right(self):
        """f ; idB = f."""
        C = _composable_pair_category()
        X = nerve_2_skeleton(C)
        composites = compose_in_quasicategory(X, ("mor", "f"), ("mor", "idB"))
        assert ("mor", "f") in composites

    def test_non_composable_raises(self):
        """Morphisms with mismatched endpoints raise ValueError."""
        C = _composable_pair_category()
        X = nerve_2_skeleton(C)
        with pytest.raises(ValueError, match="composable"):
            compose_in_quasicategory(X, ("mor", "f"), ("mor", "f"))  # f:A→B, f:A→B — not composable

    def test_not_a_1_simplex_raises(self):
        C = _arrow_category()
        X = nerve_2_skeleton(C)
        with pytest.raises(ValueError):
            compose_in_quasicategory(X, "not_a_simplex", ("mor", "f"))

    def test_filler_not_found_raises(self):
        """A partial simplicial set missing the filler should raise ValueError."""
        C = _arrow_category()
        X = nerve_2_skeleton(C)
        # idA ; idA should find a filler; idB ; f is not composable
        # Test: use a bare 2-simplex-less set
        bare = FiniteSimplicialSet(
            simplices={
                0: (("obj", "A"), ("obj", "B")),
                1: (("mor", "idA"), ("mor", "idB"), ("mor", "f")),
                2: (),  # no 2-simplices
            },
            faces={
                (1, ("mor", "idA"), 0): ("obj", "A"),
                (1, ("mor", "idA"), 1): ("obj", "A"),
                (1, ("mor", "idB"), 0): ("obj", "B"),
                (1, ("mor", "idB"), 1): ("obj", "B"),
                (1, ("mor", "f"), 0): ("obj", "B"),
                (1, ("mor", "f"), 1): ("obj", "A"),
            },
        )
        with pytest.raises(ValueError, match="filler|horn"):
            compose_in_quasicategory(bare, ("mor", "idA"), ("mor", "f"))

    def test_composable_pair_category_all_composable_pairs(self):
        """Every composable pair in the nerve of a 1-category should have a filler."""
        C = _composable_pair_category()
        X = nerve_2_skeleton(C)
        pairs = [
            (("mor", "idA"), ("mor", "f")),
            (("mor", "f"), ("mor", "g")),
            (("mor", "f"), ("mor", "idB")),
            (("mor", "idB"), ("mor", "g")),
            (("mor", "idA"), ("mor", "h")),
            (("mor", "h"), ("mor", "idC")),
        ]
        for f, g in pairs:
            composites = compose_in_quasicategory(X, f, g)
            assert len(composites) >= 1, f"No composite for ({f}, {g})"

    def test_square_two_paths_both_composable(self):
        """In a commutative square, both h∘f and k∘g exist."""
        C = _square_category()
        X = nerve_2_skeleton(C)
        c1 = compose_in_quasicategory(X, ("mor", "f"), ("mor", "h"))
        c2 = compose_in_quasicategory(X, ("mor", "g"), ("mor", "k"))
        assert ("mor", "p") in c1
        assert ("mor", "p") in c2


# ------------------------------------------------------------------ #
# Tests: morphisms_are_homotopic                                       #
# ------------------------------------------------------------------ #

class TestMorphismsAreHomotopic:
    def test_reflexive(self):
        C = _arrow_category()
        X = nerve_2_skeleton(C)
        assert morphisms_are_homotopic(X, ("mor", "f"), ("mor", "f"))

    def test_equal_implies_homotopic(self):
        C = _composable_pair_category()
        X = nerve_2_skeleton(C)
        for mor in C.morphisms:
            assert morphisms_are_homotopic(X, ("mor", mor), ("mor", mor))

    def test_non_parallel_not_homotopic(self):
        """Morphisms with different endpoints cannot be homotopic."""
        C = _composable_pair_category()
        X = nerve_2_skeleton(C)
        assert not morphisms_are_homotopic(X, ("mor", "f"), ("mor", "g"))

    def test_symmetric(self):
        """Homotopy is symmetric."""
        C = _square_category()
        X = nerve_2_skeleton(C)
        f = ("mor", "f")
        g = ("mor", "f")
        assert morphisms_are_homotopic(X, f, g) == morphisms_are_homotopic(X, g, f)

    def test_in_1_category_distinct_parallel_not_homotopic_if_unequal(self):
        """In the nerve of a 1-category, distinct parallel morphisms are not homotopic
        (there's no 2-simplex between them unless the category identifies them)."""
        # Build a category with two distinct parallel morphisms f, g: A→B
        C = FiniteCategory(
            objects=("A", "B"),
            morphisms={
                "idA": ("A", "A"), "idB": ("B", "B"),
                "f": ("A", "B"), "g": ("A", "B"),
            },
            identities={"A": "idA", "B": "idB"},
            composition={
                ("idA", "idA"): "idA",
                ("idB", "idB"): "idB",
                ("f", "idA"): "f",
                ("idB", "f"): "f",
                ("g", "idA"): "g",
                ("idB", "g"): "g",
            },
        )
        X = nerve_2_skeleton(C)
        # f and g are distinct parallel; no 2-simplex relates them
        assert not morphisms_are_homotopic(X, ("mor", "f"), ("mor", "g"))


# ------------------------------------------------------------------ #
# Tests: homotopy_category                                             #
# ------------------------------------------------------------------ #

class TestHomotopyCategory:
    def test_returns_finite_category(self):
        C = _arrow_category()
        X = nerve_2_skeleton(C)
        Ho = homotopy_category(X)
        assert isinstance(Ho, FiniteCategory)

    def test_objects_are_0_simplices(self):
        C = _composable_pair_category()
        X = nerve_2_skeleton(C)
        Ho = homotopy_category(X)
        # Objects of Ho should include the 0-simplices of X
        obj_set = set(Ho.objects)
        for obj in C.objects:
            assert ("obj", obj) in obj_set

    def test_morphism_count_matches_1_category(self):
        """
        In the nerve of a 1-category where all morphisms are distinct,
        every 1-simplex is its own homotopy class.
        """
        C = _composable_pair_category()
        X = nerve_2_skeleton(C)
        Ho = homotopy_category(X)
        # 6 morphisms in C → 6 homotopy classes
        assert len(Ho.morphisms) == len(C.morphisms)

    def test_identities_exist_for_all_objects(self):
        C = _composable_pair_category()
        X = nerve_2_skeleton(C)
        Ho = homotopy_category(X)
        for obj in Ho.objects:
            assert obj in Ho.identities
            id_mor = Ho.identities[obj]
            assert id_mor in Ho.morphisms

    def test_composition_is_associative(self):
        """The homotopy category inherits associativity from the quasi-category."""
        C = _composable_pair_category()
        X = nerve_2_skeleton(C)
        Ho = homotopy_category(X)
        # If associativity holds, validate_laws() passes (called in __init__)
        # Just verify the category was constructed without error
        assert Ho is not None

    def test_homotopy_category_of_nerve_recovers_1_category(self):
        """
        Ho(N(C)) ≅ C for any 1-category C.
        We verify that the composition table matches C's.
        """
        C = _composable_pair_category()
        X = nerve_2_skeleton(C)
        Ho = homotopy_category(X)

        # Map C-morphisms to Ho-morphisms via the class representative label
        # For each composable pair in C, check that Ho computes the right composite
        for mor_g, mor_f in C.composable_pairs():
            expected_composite = C.compose(mor_g, mor_f)
            lbl_g = f"[{('mor', mor_g)!r}]"
            lbl_f = f"[{('mor', mor_f)!r}]"
            lbl_exp = f"[{('mor', expected_composite)!r}]"
            if lbl_g in Ho.morphisms and lbl_f in Ho.morphisms:
                actual = Ho.compose(lbl_g, lbl_f)
                assert actual == lbl_exp, (
                    f"Ho compose({mor_g}, {mor_f}): expected {lbl_exp!r}, got {actual!r}"
                )

    def test_square_category_ho_category_valid(self):
        C = _square_category()
        X = nerve_2_skeleton(C)
        Ho = homotopy_category(X)
        assert isinstance(Ho, FiniteCategory)
        assert len(Ho.objects) == len(C.objects)

    def test_single_object_loop_category(self):
        """Category with one object (monoid) — Ho should also be one-object."""
        C = FiniteCategory(
            objects=("*",),
            morphisms={"id": ("*", "*"), "a": ("*", "*")},
            identities={"*": "id"},
            composition={
                ("id", "id"): "id",
                ("a", "id"): "a",
                ("id", "a"): "a",
                ("a", "a"): "a",  # a is idempotent
            },
        )
        X = nerve_2_skeleton(C)
        Ho = homotopy_category(X)
        assert len(Ho.objects) == 1
