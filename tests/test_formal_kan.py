"""
Tests for topos_ai.formal_kan — Formal Left and Right Kan Extensions.

Test families:
1. FiniteSetFunctor — construction and validation
2. Left Kan Extension — objects, morphisms, unit
3. Right Kan Extension — matching families
4. Natural transformations — enumeration
5. Universal property — cardinality bijection
"""
import pytest
from topos_ai.formal_category import FiniteCategory
from topos_ai.formal_kan import (
    FiniteSetFunctor,
    left_kan_extension,
    right_kan_extension,
    all_natural_transformations,
    verify_left_kan_universal_property,
    verify_right_kan_universal_property,
    left_kan_unit,
)


# ------------------------------------------------------------------ #
# Shared fixtures                                                       #
# ------------------------------------------------------------------ #

def _terminal_C():
    """Single-object category C = {0}."""
    return FiniteCategory(
        objects=["0"],
        morphisms={"id0": ("0", "0")},
        identities={"0": "id0"},
        composition={("id0", "id0"): "id0"},
    )


def _arrow_D():
    """Category D = {A, B} with a single non-identity arrow f: A → B."""
    return FiniteCategory(
        objects=["A", "B"],
        morphisms={"idA": ("A", "A"), "idB": ("B", "B"), "f": ("A", "B")},
        identities={"A": "idA", "B": "idB"},
        composition={
            ("idA", "idA"): "idA", ("idB", "idB"): "idB",
            ("f", "idA"): "f", ("idB", "f"): "f",
        },
    )


def _discrete_two():
    """Discrete category on {A, B} (no non-identity morphisms)."""
    return FiniteCategory(
        objects=["A", "B"],
        morphisms={"idA": ("A", "A"), "idB": ("B", "B")},
        identities={"A": "idA", "B": "idB"},
        composition={("idA", "idA"): "idA", ("idB", "idB"): "idB"},
    )


def _X_singleton(C):
    """X: C → FinSet mapping the single object 0 to {x1}."""
    return FiniteSetFunctor(
        category=C,
        objects_map={"0": frozenset(["x1"])},
        morphism_map={"id0": {"x1": "x1"}},
    )


def _X_two_elem(C):
    """X: C → FinSet mapping 0 to {x1, x2}."""
    return FiniteSetFunctor(
        category=C,
        objects_map={"0": frozenset(["x1", "x2"])},
        morphism_map={"id0": {"x1": "x1", "x2": "x2"}},
    )


def _K_to_A():
    """Functor K: C → D mapping 0 ↦ A."""
    return {"0": "A"}, {"id0": "idA"}


def _Y_two_two(D):
    """Y: D → FinSet with Y(A)={y1,y2}, Y(B)={y3,y4}, Y(f): y1↦y3, y2↦y4."""
    return FiniteSetFunctor(
        category=D,
        objects_map={"A": frozenset(["y1", "y2"]), "B": frozenset(["y3", "y4"])},
        morphism_map={
            "idA": {"y1": "y1", "y2": "y2"},
            "idB": {"y3": "y3", "y4": "y4"},
            "f":   {"y1": "y3", "y2": "y4"},
        },
    )


# ------------------------------------------------------------------ #
# FiniteSetFunctor                                                       #
# ------------------------------------------------------------------ #

class TestFiniteSetFunctor:
    def test_constructs(self):
        C = _terminal_C()
        X = _X_singleton(C)
        assert isinstance(X, FiniteSetFunctor)

    def test_apply_obj(self):
        C = _terminal_C()
        X = _X_singleton(C)
        assert X.apply_obj("0") == frozenset(["x1"])

    def test_apply_mor(self):
        C = _terminal_C()
        X = _X_singleton(C)
        assert X.apply_mor("id0") == {"x1": "x1"}

    def test_missing_object_raises(self):
        C = _terminal_C()
        with pytest.raises(ValueError, match="missing objects_map"):
            FiniteSetFunctor(
                category=C,
                objects_map={},            # missing "0"
                morphism_map={"id0": {}},
            )

    def test_missing_morphism_raises(self):
        C = _terminal_C()
        with pytest.raises(ValueError, match="missing morphism_map"):
            FiniteSetFunctor(
                category=C,
                objects_map={"0": frozenset(["x"])},
                morphism_map={},           # missing "id0"
            )

    def test_morphism_out_of_range_raises(self):
        C = _terminal_C()
        with pytest.raises(ValueError):
            FiniteSetFunctor(
                category=C,
                objects_map={"0": frozenset(["x"])},
                morphism_map={"id0": {"x": "INVALID"}},
            )

    def test_repr(self):
        C = _terminal_C()
        X = _X_singleton(C)
        assert "FiniteSetFunctor" in repr(X)

    def test_two_element_functor(self):
        C = _terminal_C()
        X = _X_two_elem(C)
        assert len(X.apply_obj("0")) == 2


# ------------------------------------------------------------------ #
# Left Kan Extension — objects                                          #
# ------------------------------------------------------------------ #

class TestLeftKanObjects:
    def test_frontier1_object_A(self):
        """Lan_K X(A) should contain one class (X(0) = {x1}, K(0) = A)."""
        C, D = _terminal_C(), _arrow_D()
        X = _X_singleton(C)
        K_obj, K_mor = _K_to_A()
        lan = left_kan_extension(X, K_obj, K_mor, D)
        assert len(lan.apply_obj("A")) == 1

    def test_frontier1_object_B(self):
        """Lan_K X(B) = colim over {f: A→B}; also 1 class."""
        C, D = _terminal_C(), _arrow_D()
        X = _X_singleton(C)
        K_obj, K_mor = _K_to_A()
        lan = left_kan_extension(X, K_obj, K_mor, D)
        assert len(lan.apply_obj("B")) == 1

    def test_two_elem_lan_sizes(self):
        """With X(0) = {x1, x2}, Lan_K X(A) has 2 elements."""
        C, D = _terminal_C(), _arrow_D()
        X = _X_two_elem(C)
        K_obj, K_mor = _K_to_A()
        lan = left_kan_extension(X, K_obj, K_mor, D)
        assert len(lan.apply_obj("A")) == 2

    def test_lan_is_finite_set_functor(self):
        C, D = _terminal_C(), _arrow_D()
        X = _X_singleton(C)
        K_obj, K_mor = _K_to_A()
        lan = left_kan_extension(X, K_obj, K_mor, D)
        assert isinstance(lan, FiniteSetFunctor)

    def test_lan_on_discrete_target(self):
        """K: C→D_discrete maps 0→A; no morphisms to B, so Lan_K X(B) = ∅."""
        C = _terminal_C()
        D = _discrete_two()
        X = _X_singleton(C)
        K_obj = {"0": "A"}
        K_mor = {"id0": "idA"}
        lan = left_kan_extension(X, K_obj, K_mor, D)
        # Lan at A: comma category over A has (0, idA) — 1 element
        assert len(lan.apply_obj("A")) == 1
        # Lan at B: comma category over B is empty — 0 elements
        assert len(lan.apply_obj("B")) == 0


# ------------------------------------------------------------------ #
# Left Kan Extension — morphism action                                  #
# ------------------------------------------------------------------ #

class TestLeftKanMorphisms:
    def test_morphism_map_keys_match_D(self):
        C, D = _terminal_C(), _arrow_D()
        X = _X_singleton(C)
        K_obj, K_mor = _K_to_A()
        lan = left_kan_extension(X, K_obj, K_mor, D)
        assert set(lan.morphism_map.keys()) == set(D.morphisms.keys())

    def test_identity_morphism_is_identity_function(self):
        """Lan_K X applied to idA should map each element to itself."""
        C, D = _terminal_C(), _arrow_D()
        X = _X_singleton(C)
        K_obj, K_mor = _K_to_A()
        lan = left_kan_extension(X, K_obj, K_mor, D)
        idA_map = lan.apply_mor("idA")
        for x, y in idA_map.items():
            assert x == y

    def test_f_maps_A_elements_to_B_elements(self):
        """f: A→B in D; Lan_K X(f) maps each class in Lan(A) to a class in Lan(B)."""
        C, D = _terminal_C(), _arrow_D()
        X = _X_singleton(C)
        K_obj, K_mor = _K_to_A()
        lan = left_kan_extension(X, K_obj, K_mor, D)
        f_map = lan.apply_mor("f")
        for img in f_map.values():
            assert img in lan.apply_obj("B")


# ------------------------------------------------------------------ #
# Left Kan — unit                                                       #
# ------------------------------------------------------------------ #

class TestLeftKanUnit:
    def test_unit_defined_for_all_C_objects(self):
        C, D = _terminal_C(), _arrow_D()
        X = _X_singleton(C)
        K_obj, K_mor = _K_to_A()
        eta = left_kan_unit(X, K_obj, K_mor, D)
        assert "0" in eta

    def test_unit_maps_X_elements_to_lan_elements(self):
        C, D = _terminal_C(), _arrow_D()
        X = _X_singleton(C)
        K_obj, K_mor = _K_to_A()
        eta = left_kan_unit(X, K_obj, K_mor, D)
        lan = left_kan_extension(X, K_obj, K_mor, D)
        for x, cls in eta["0"].items():
            assert cls in lan.apply_obj("A")   # K(0) = A


# ------------------------------------------------------------------ #
# Right Kan Extension                                                   #
# ------------------------------------------------------------------ #

class TestRightKanExtension:
    def test_ran_is_finite_set_functor(self):
        C, D = _terminal_C(), _arrow_D()
        X = _X_singleton(C)
        K_obj, K_mor = _K_to_A()
        ran = right_kan_extension(X, K_obj, K_mor, D)
        assert isinstance(ran, FiniteSetFunctor)

    def test_ran_object_A_nonempty(self):
        C, D = _terminal_C(), _arrow_D()
        X = _X_singleton(C)
        K_obj, K_mor = _K_to_A()
        ran = right_kan_extension(X, K_obj, K_mor, D)
        # (A↓K) has at least (0, idA), so Ran(A) has elements
        assert len(ran.apply_obj("A")) >= 1

    def test_ran_two_elem_sizes(self):
        """With X(0) = {x1, x2}, Ran at A (which has comma (0,idA)) has size 2."""
        C, D = _terminal_C(), _arrow_D()
        X = _X_two_elem(C)
        K_obj, K_mor = _K_to_A()
        ran = right_kan_extension(X, K_obj, K_mor, D)
        assert len(ran.apply_obj("A")) == 2

    def test_ran_morphism_keys_match_D(self):
        C, D = _terminal_C(), _arrow_D()
        X = _X_singleton(C)
        K_obj, K_mor = _K_to_A()
        ran = right_kan_extension(X, K_obj, K_mor, D)
        assert set(ran.morphism_map.keys()) == set(D.morphisms.keys())


# ------------------------------------------------------------------ #
# Natural transformations                                               #
# ------------------------------------------------------------------ #

class TestNaturalTransformations:
    def _two_singleton_funs(self, D):
        """Two constant functors D → FinSet: const_{x} and const_{y}."""
        F = FiniteSetFunctor(
            category=D,
            objects_map={"A": frozenset(["x"]), "B": frozenset(["x"])},
            morphism_map={"idA": {"x": "x"}, "idB": {"x": "x"}, "f": {"x": "x"}},
        )
        return F, F

    def test_identity_nat_exists(self):
        D = _arrow_D()
        F, G = self._two_singleton_funs(D)
        nats = all_natural_transformations(F, G)
        # Exactly one nat between singleton functors
        assert len(nats) == 1

    def test_nat_from_frontier1(self):
        """Nat(Lan_K X, Y) should equal Nat(X, Y∘K) in cardinality (=2)."""
        C, D = _terminal_C(), _arrow_D()
        X = _X_singleton(C)
        K_obj, K_mor = _K_to_A()
        Y = _Y_two_two(D)
        lan = left_kan_extension(X, K_obj, K_mor, D)
        nats = all_natural_transformations(lan, Y)
        assert len(nats) == 2   # matches |Nat(X, Y∘K)| = |Y(A)| = 2

    def test_no_nat_between_different_size_singletons(self):
        """No nat from a 2-element to an empty functor."""
        D = _arrow_D()
        F = FiniteSetFunctor(
            category=D,
            objects_map={"A": frozenset(["x", "y"]), "B": frozenset(["p", "q"])},
            morphism_map={
                "idA": {"x": "x", "y": "y"},
                "idB": {"p": "p", "q": "q"},
                "f":   {"x": "p", "y": "q"},
            },
        )
        G = FiniteSetFunctor(
            category=D,
            objects_map={"A": frozenset(), "B": frozenset()},
            morphism_map={"idA": {}, "idB": {}, "f": {}},
        )
        nats = all_natural_transformations(F, G)
        assert len(nats) == 0


# ------------------------------------------------------------------ #
# Universal property                                                    #
# ------------------------------------------------------------------ #

class TestUniversalProperty:
    def test_left_kan_universal_property_frontier1(self):
        """Reproduce Frontier 1: |Nat(Lan_K X, Y)| == |Nat(X, Y∘K)|."""
        C, D = _terminal_C(), _arrow_D()
        X = _X_singleton(C)
        K_obj, K_mor = _K_to_A()
        Y = _Y_two_two(D)
        assert verify_left_kan_universal_property(X, K_obj, K_mor, D, Y)

    def test_left_kan_universal_two_elem(self):
        """Same check with a 2-element X(0)."""
        C, D = _terminal_C(), _arrow_D()
        X = _X_two_elem(C)
        K_obj, K_mor = _K_to_A()
        Y = _Y_two_two(D)
        assert verify_left_kan_universal_property(X, K_obj, K_mor, D, Y)

    def test_right_kan_universal_property(self):
        """Check |Nat(Y∘K, X)| == |Nat(Y, Ran_K X)|."""
        C, D = _terminal_C(), _arrow_D()
        X = _X_singleton(C)
        K_obj, K_mor = _K_to_A()
        Y = _Y_two_two(D)
        assert verify_right_kan_universal_property(X, K_obj, K_mor, D, Y)

    def test_universal_property_singleton_Y(self):
        """With singleton Y, both sides of the adjunction should agree."""
        C, D = _terminal_C(), _arrow_D()
        X = _X_singleton(C)
        K_obj, K_mor = _K_to_A()
        Y_single = FiniteSetFunctor(
            category=D,
            objects_map={"A": frozenset(["z"]), "B": frozenset(["z"])},
            morphism_map={"idA": {"z": "z"}, "idB": {"z": "z"}, "f": {"z": "z"}},
        )
        assert verify_left_kan_universal_property(X, K_obj, K_mor, D, Y_single)
