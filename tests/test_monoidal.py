"""Tests for topos_ai.monoidal — FiniteMonoidalCategory and FiniteSymmetricMonoidalCategory."""

import pytest

from topos_ai.formal_category import FiniteCategory
from topos_ai.monoidal import (
    FiniteMonoidalCategory,
    FiniteSymmetricMonoidalCategory,
    strict_monoidal_from_monoid,
)


# ------------------------------------------------------------------ #
# Helpers: small monoidal categories                                  #
# ------------------------------------------------------------------ #

def _trivial_monoidal():
    """Single-object strict symmetric monoidal category."""
    return strict_monoidal_from_monoid(
        objects=["*"],
        tensor_table={("*", "*"): "*"},
        unit="*",
    )


def _bool_and_smc():
    """
    Discrete 2-object (False/True) symmetric monoidal category
    where ⊗ = logical AND and unit = True.
    """
    return strict_monoidal_from_monoid(
        objects=["F", "T"],
        tensor_table={
            ("F", "F"): "F",
            ("F", "T"): "F",
            ("T", "F"): "F",
            ("T", "T"): "T",
        },
        unit="T",
    )


def _bool_or_smc():
    """Discrete 2-object SMC where ⊗ = logical OR and unit = False."""
    return strict_monoidal_from_monoid(
        objects=["F", "T"],
        tensor_table={
            ("F", "F"): "F",
            ("F", "T"): "T",
            ("T", "F"): "T",
            ("T", "T"): "T",
        },
        unit="F",
    )


def _walking_arrow_monoidal():
    """
    Non-strict monoidal structure on the walking-arrow category 0 → 1.

    We take ⊗ = max on objects {0,1} with unit 0, so:
      0⊗0=0, 0⊗1=1, 1⊗0=1, 1⊗1=1.

    Structure maps:
      α_{A,B,C} = identity on (A⊗B)⊗C  (strict associator)
      λ_A = morphism (0⊗A) → A
      ρ_A = morphism (A⊗0) → A

    Because the category has non-trivial morphisms (id_0, id_1, f: 0→1)
    we need to supply all tensor morphisms explicitly.
    """
    C = FiniteCategory(
        objects=("0", "1"),
        morphisms={"id0": ("0", "0"), "id1": ("1", "1"), "f": ("0", "1")},
        identities={"0": "id0", "1": "id1"},
        composition={
            ("id0", "id0"): "id0",
            ("id1", "id1"): "id1",
            ("f", "id0"): "f",
            ("id1", "f"): "f",
        },
    )

    # A⊗B = max(A,B): 0⊗0=0, 0⊗1=1, 1⊗0=1, 1⊗1=1
    def to(A, B):
        return "1" if A == "1" or B == "1" else "0"

    tensor_objects = {(A, B): to(A, B) for A in ("0", "1") for B in ("0", "1")}

    # f⊗g: src(f)⊗src(g) → dst(f)⊗dst(g)
    # All morphisms: id0, id1, f
    # id0⊗id0: 0⊗0=0 → 0⊗0=0  = id0
    # id0⊗id1: 0⊗1=1 → 0⊗1=1  = id1
    # id0⊗f  : 0⊗0=0 → 0⊗1=1  = f
    # id1⊗id0: 1⊗0=1 → 1⊗0=1  = id1
    # id1⊗id1: 1⊗1=1 → 1⊗1=1  = id1
    # id1⊗f  : 1⊗0=1 → 1⊗1=1  = id1
    # f⊗id0  : 0⊗0=0 → 1⊗0=1  = f
    # f⊗id1  : 0⊗1=1 → 1⊗1=1  = id1
    # f⊗f    : 0⊗0=0 → 1⊗1=1  = f
    tensor_morphisms = {
        ("id0", "id0"): "id0",
        ("id0", "id1"): "id1",
        ("id0", "f"): "f",
        ("id1", "id0"): "id1",
        ("id1", "id1"): "id1",
        ("id1", "f"): "id1",
        ("f", "id0"): "f",
        ("f", "id1"): "id1",
        ("f", "f"): "f",
    }

    # Strict associator: all identity morphisms on (A⊗B)⊗C
    def alpha_mor(A, B, Cv):
        AB = to(A, B)
        ABC = to(AB, Cv)
        return "id0" if ABC == "0" else "id1"

    associator = {(A, B, Cv): alpha_mor(A, B, Cv) for A in ("0", "1") for B in ("0", "1") for Cv in ("0", "1")}

    # λ_A: 0⊗A → A
    # 0⊗0=0 → 0: id0
    # 0⊗1=1 → 1: id1
    left_unitor = {"0": "id0", "1": "id1"}
    # ρ_A: A⊗0 → A
    # 0⊗0=0 → 0: id0
    # 1⊗0=1 → 1: id1
    right_unitor = {"0": "id0", "1": "id1"}

    return FiniteMonoidalCategory(
        category=C,
        tensor_objects=tensor_objects,
        tensor_morphisms=tensor_morphisms,
        unit="0",
        associator=associator,
        left_unitor=left_unitor,
        right_unitor=right_unitor,
    )


# ------------------------------------------------------------------ #
# Tests: construction and basic axioms                                #
# ------------------------------------------------------------------ #

def test_trivial_monoidal_constructs():
    mc = _trivial_monoidal()
    assert mc.tensor_obj("*", "*") == "*"
    assert mc.unit == "*"
    assert mc.alpha("*", "*", "*") == "id_*"
    assert mc.lambda_("*") == "id_*"
    assert mc.rho("*") == "id_*"


def test_bool_and_smc_constructs():
    smc = _bool_and_smc()
    assert smc.tensor_obj("T", "T") == "T"
    assert smc.tensor_obj("T", "F") == "F"
    assert smc.unit == "T"
    # Braiding is identity on symmetric strict monoidal
    assert smc.gamma("T", "F") == "id_F"


def test_bool_or_smc_constructs():
    smc = _bool_or_smc()
    assert smc.tensor_obj("T", "F") == "T"
    assert smc.unit == "F"


def test_walking_arrow_monoidal_constructs():
    mc = _walking_arrow_monoidal()
    assert mc.tensor_obj("0", "1") == "1"
    assert mc.tensor_obj("1", "1") == "1"
    assert mc.unit == "0"
    assert mc.tensor_mor("f", "id0") == "f"


def test_pentagon_holds_for_bool_and():
    """Pentagon is validated at construction; this test just re-asserts it passes."""
    smc = _bool_and_smc()
    smc._check_pentagon()  # should not raise


def test_triangle_holds_for_bool_and():
    smc = _bool_and_smc()
    smc._check_triangle()


def test_hexagon_holds_for_trivial():
    smc = _trivial_monoidal()
    smc._check_hexagon()


def test_braiding_involutivity():
    smc = _bool_and_smc()
    C = smc.category
    for A in C.objects:
        for B in C.objects:
            AB = smc.tensor_obj(A, B)
            assert C.compose(smc.gamma(B, A), smc.gamma(A, B)) == C.identities[AB]


def test_braiding_naturality():
    smc = _bool_and_smc()
    smc._check_braiding_naturality()


# ------------------------------------------------------------------ #
# Tests: violation detection                                           #
# ------------------------------------------------------------------ #

def test_wrong_unit_raises():
    with pytest.raises(ValueError, match="Unit"):
        strict_monoidal_from_monoid(
            objects=["a"],
            tensor_table={("a", "a"): "a"},
            unit="b",  # not an object
        )


def test_associator_wrong_type_raises():
    """An associator with incorrect source/target type is rejected."""
    smc = _bool_and_smc()
    bad_assoc = dict(smc.associator)
    # Point α_{F,F,T} to the wrong identity (id_T instead of id_F)
    bad_assoc[("F", "F", "T")] = "id_T"
    # This breaks the type check because (F⊗F)⊗T = F⊗T = F, but id_T : T→T ≠ F→F
    with pytest.raises(ValueError, match="[Aa]ssociator|type"):
        FiniteSymmetricMonoidalCategory(
            category=smc.category,
            tensor_objects=smc.tensor_objects,
            tensor_morphisms=smc.tensor_morphisms,
            unit=smc.unit,
            associator=bad_assoc,
            left_unitor=smc.left_unitor,
            right_unitor=smc.right_unitor,
            braiding=smc.braiding,
        )


def test_bifunctoriality_violation_raises():
    """Tensor morphism that breaks id_A⊗id_B = id_{A⊗B} is rejected."""
    smc = _bool_and_smc()
    bad_tensor_morphisms = dict(smc.tensor_morphisms)
    # id_F⊗id_T should be id_F (since F AND T = F), corrupt to id_T
    bad_tensor_morphisms[("id_F", "id_T")] = "id_T"
    with pytest.raises(ValueError, match="[Bb]ifunctorial|type"):
        FiniteSymmetricMonoidalCategory(
            category=smc.category,
            tensor_objects=smc.tensor_objects,
            tensor_morphisms=bad_tensor_morphisms,
            unit=smc.unit,
            associator=smc.associator,
            left_unitor=smc.left_unitor,
            right_unitor=smc.right_unitor,
            braiding=smc.braiding,
        )


def test_braiding_involutivity_violation_raises():
    smc = _bool_and_smc()
    bad_braiding = dict(smc.braiding)
    # Make γ_{F,T} point to id_T instead of id_F (breaks involutivity)
    bad_braiding[("F", "T")] = "id_T"
    with pytest.raises(ValueError, match="[Bb]raiding|involutiv"):
        FiniteSymmetricMonoidalCategory(
            category=smc.category,
            tensor_objects=smc.tensor_objects,
            tensor_morphisms=smc.tensor_morphisms,
            unit=smc.unit,
            associator=smc.associator,
            left_unitor=smc.left_unitor,
            right_unitor=smc.right_unitor,
            braiding=bad_braiding,
        )


# ------------------------------------------------------------------ #
# Tests: strict_monoidal_from_monoid helper                           #
# ------------------------------------------------------------------ #

def test_z2_monoidal():
    """Z/2Z additive group as a symmetric monoidal discrete category."""
    smc = strict_monoidal_from_monoid(
        objects=["0", "1"],
        tensor_table={
            ("0", "0"): "0",
            ("0", "1"): "1",
            ("1", "0"): "1",
            ("1", "1"): "0",
        },
        unit="0",
    )
    assert smc.tensor_obj("1", "1") == "0"
    assert smc.unit == "0"
    smc._check_pentagon()
    smc._check_triangle()
    smc._check_hexagon()


def test_unit_object_is_neutral():
    """I⊗A = A and A⊗I = A for every A in a strict monoidal category."""
    smc = _bool_and_smc()
    I = smc.unit
    for A in smc.category.objects:
        assert smc.tensor_obj(I, A) == A
        assert smc.tensor_obj(A, I) == A
