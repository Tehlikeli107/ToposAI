"""
Tests for topos_ai.enriched — V-Enriched Categories.

We test three families of enriched categories built over discrete SMCs:

1. **Bool-chaotic (trivial)**: V = (Bool, ∧, T), all hom(A,B) = T.
   This is the "fully connected" preorder.  Works because T∧T=T for all triples.

2. **Z/2Z symmetric enrichment**: V = (Z/2Z, +, 0).
   For 2 objects {A,B}: hom(A,A)=hom(B,B)=0, hom(A,B)=hom(B,A)=1.
   Check: 1+1=0=hom(A,A) ✓; 0+1=1=hom(A,B) ✓; etc.

3. **Trivial self-enrichment**: single-object C(*) = unit in any SMC.

Note: over a *discrete* SMC V (only identity morphisms), the composition
morphism ∘_{A,B,C}: hom(B,C)⊗hom(A,B) → hom(A,C) must BE an identity,
i.e. hom(B,C)⊗hom(A,B) = hom(A,C) strictly.  This constrains valid hom
assignments to those where the hom-matrix satisfies the "module" equation.
"""

import pytest

from topos_ai.formal_category import FiniteCategory
from topos_ai.monoidal import FiniteMonoidalCategory, strict_monoidal_from_monoid
from topos_ai.enriched import FiniteEnrichedCategory, discrete_enriched_category


# ------------------------------------------------------------------ #
# Helpers: enriching categories                                        #
# ------------------------------------------------------------------ #

def _bool_and_V():
    """V = (Bool, ∧, T) — fully-connected case has all hom(A,B)=T."""
    return strict_monoidal_from_monoid(
        objects=["F", "T"],
        tensor_table={
            ("F", "F"): "F", ("F", "T"): "F",
            ("T", "F"): "F", ("T", "T"): "T",
        },
        unit="T",
    )


def _z2_V():
    """V = (Z/2Z, +, 0)."""
    return strict_monoidal_from_monoid(
        objects=["0", "1"],
        tensor_table={
            ("0", "0"): "0", ("0", "1"): "1",
            ("1", "0"): "1", ("1", "1"): "0",
        },
        unit="0",
    )


# ------------------------------------------------------------------ #
# Helpers: specific enriched categories                               #
# ------------------------------------------------------------------ #

def _bool_chaotic_enriched(n_objs=2):
    """
    Bool-enriched 'chaotic' category: all hom-objects = T (True = unit).

    Valid because T∧T=T for all triples, so all compositions are id_T : T→T.
    Identity elements are also id_T : T→T (T = unit).
    """
    V = _bool_and_V()
    C_cat = V.category
    objs = [chr(65 + i) for i in range(n_objs)]  # ["A","B",...]

    hom = {(X, Y): "T" for X in objs for Y in objs}
    comps = {
        (X, Y, Z): C_cat.identities["T"]
        for X in objs for Y in objs for Z in objs
    }
    ids = {X: C_cat.identities["T"] for X in objs}
    return FiniteEnrichedCategory(
        objects=objs, enriching=V, hom_objects=hom,
        compositions=comps, identity_elements=ids,
    )


def _z2_symmetric_two_obj():
    """
    Z/2Z-enriched 2-object category: hom(A,A)=hom(B,B)=0, hom(A,B)=hom(B,A)=1.

    Verified:
      ∘_{A,B,A}: 1+1=0=hom(A,A) → id_0: 0→0  ✓
      ∘_{A,A,B}: 1+0=1=hom(A,B) → id_1: 1→1  ✓
      ∘_{A,B,B}: 0+1=1=hom(A,B) → id_1: 1→1  ✓
      ∘_{A,A,A}: 0+0=0=hom(A,A) → id_0: 0→0  ✓
      (symmetric in B)
    """
    V = _z2_V()
    C_cat = V.category
    objs = ["A", "B"]

    hom = {
        ("A", "A"): "0", ("A", "B"): "1",
        ("B", "A"): "1", ("B", "B"): "0",
    }
    comps = {}
    for X in objs:
        for Y in objs:
            for Z in objs:
                src = V.tensor_obj(hom[(Y, Z)], hom[(X, Y)])
                dst = hom[(X, Z)]
                assert src == dst, f"Z/2Z hom matrix broken for ({X},{Y},{Z}): {src}≠{dst}"
                comps[(X, Y, Z)] = C_cat.identities[src]

    # j_A: unit=0 → hom(A,A)=0 = id_0
    ids = {X: C_cat.identities["0"] for X in objs}
    return FiniteEnrichedCategory(
        objects=objs, enriching=V, hom_objects=hom,
        compositions=comps, identity_elements=ids,
    )


def _trivial_single_obj():
    """Single-object enriched category over Z/2Z: hom(*,*)=unit=0."""
    V = _z2_V()
    C_cat = V.category
    return FiniteEnrichedCategory(
        objects=["*"],
        enriching=V,
        hom_objects={("*", "*"): "0"},
        compositions={("*", "*", "*"): C_cat.identities["0"]},
        identity_elements={"*": C_cat.identities["0"]},
    )


# ------------------------------------------------------------------ #
# Tests: construction                                                  #
# ------------------------------------------------------------------ #

class TestFiniteEnrichedCategoryConstruct:
    def test_trivial_constructs(self):
        ec = _trivial_single_obj()
        assert isinstance(ec, FiniteEnrichedCategory)

    def test_bool_chaotic_2obj_constructs(self):
        ec = _bool_chaotic_enriched(2)
        assert isinstance(ec, FiniteEnrichedCategory)

    def test_bool_chaotic_3obj_constructs(self):
        ec = _bool_chaotic_enriched(3)
        assert isinstance(ec, FiniteEnrichedCategory)

    def test_z2_symmetric_constructs(self):
        ec = _z2_symmetric_two_obj()
        assert isinstance(ec, FiniteEnrichedCategory)

    def test_objects_preserved(self):
        ec = _bool_chaotic_enriched(3)
        assert set(ec.objects) == {"A", "B", "C"}

    def test_hom_accessor_bool_chaotic(self):
        ec = _bool_chaotic_enriched(2)
        assert ec.hom("A", "B") == "T"
        assert ec.hom("A", "A") == "T"

    def test_hom_accessor_z2(self):
        ec = _z2_symmetric_two_obj()
        assert ec.hom("A", "A") == "0"
        assert ec.hom("A", "B") == "1"

    def test_hom_missing_raises(self):
        ec = _bool_chaotic_enriched(2)
        with pytest.raises(ValueError, match="Hom-object"):
            ec.hom("X", "Y")

    def test_compose_accessor(self):
        ec = _z2_symmetric_two_obj()
        result = ec.compose("A", "B", "A")
        assert result is not None

    def test_compose_missing_raises(self):
        ec = _z2_symmetric_two_obj()
        with pytest.raises(ValueError, match="Composition"):
            ec.compose("X", "Y", "Z")

    def test_identity_element_accessor(self):
        ec = _z2_symmetric_two_obj()
        assert ec.identity_element("A") is not None

    def test_identity_element_missing_raises(self):
        ec = _z2_symmetric_two_obj()
        with pytest.raises(ValueError, match="Identity element"):
            ec.identity_element("Z")

    def test_enriching_accessible(self):
        V = _z2_V()
        ec = _trivial_single_obj()
        assert ec.enriching is not None


# ------------------------------------------------------------------ #
# Tests: type validation errors                                        #
# ------------------------------------------------------------------ #

class TestTypeValidation:
    def test_bad_hom_object_raises(self):
        V = _bool_and_V()
        C_cat = V.category
        with pytest.raises(ValueError, match="V-object"):
            FiniteEnrichedCategory(
                objects=["A"],
                enriching=V,
                hom_objects={("A", "A"): "INVALID"},
                compositions={("A", "A", "A"): C_cat.identities["T"]},
                identity_elements={"A": C_cat.identities["T"]},
            )

    def test_missing_hom_object_raises(self):
        V = _bool_and_V()
        C_cat = V.category
        with pytest.raises(ValueError, match="Missing hom-object"):
            FiniteEnrichedCategory(
                objects=["A", "B"],
                enriching=V,
                hom_objects={("A", "A"): "T", ("B", "B"): "T"},
                compositions={
                    (X, Y, Z): C_cat.identities["T"]
                    for X in ["A","B"] for Y in ["A","B"] for Z in ["A","B"]
                },
                identity_elements={"A": C_cat.identities["T"], "B": C_cat.identities["T"]},
            )

    def test_missing_composition_raises(self):
        V = _z2_V()
        C_cat = V.category
        with pytest.raises(ValueError, match="Missing composition"):
            FiniteEnrichedCategory(
                objects=["A"],
                enriching=V,
                hom_objects={("A", "A"): "0"},
                compositions={},  # empty — should raise on missing ∘_{A,A,A}
                identity_elements={"A": C_cat.identities["0"]},
            )

    def test_bad_composition_type_raises(self):
        """Composition morphism with wrong source/target type."""
        V = _z2_V()
        C_cat = V.category
        with pytest.raises(ValueError, match="[Cc]omposition.*type|type.*[Cc]omposition"):
            FiniteEnrichedCategory(
                objects=["A"],
                enriching=V,
                hom_objects={("A", "A"): "0"},
                compositions={("A", "A", "A"): C_cat.identities["1"]},  # 1→1 instead of 0→0
                identity_elements={"A": C_cat.identities["0"]},
            )

    def test_missing_identity_raises(self):
        V = _z2_V()
        C_cat = V.category
        with pytest.raises(ValueError, match="Missing identity"):
            FiniteEnrichedCategory(
                objects=["A"],
                enriching=V,
                hom_objects={("A", "A"): "0"},
                compositions={("A", "A", "A"): C_cat.identities["0"]},
                identity_elements={},  # missing j_A
            )

    def test_bad_identity_type_raises(self):
        """Identity element pointing to morphism with wrong type."""
        V = _z2_V()
        C_cat = V.category
        with pytest.raises(ValueError, match="[Ii]dentity.*type|type.*[Ii]dentity"):
            FiniteEnrichedCategory(
                objects=["A"],
                enriching=V,
                hom_objects={("A", "A"): "0"},
                compositions={("A", "A", "A"): C_cat.identities["0"]},
                identity_elements={"A": C_cat.identities["1"]},  # 1→1, should be 0→0
            )


# ------------------------------------------------------------------ #
# Tests: axiom verification                                            #
# ------------------------------------------------------------------ #

class TestAxioms:
    def test_trivial_satisfies_all_axioms(self):
        ec = _trivial_single_obj()
        assert ec is not None

    def test_bool_chaotic_satisfies_all_axioms(self):
        ec = _bool_chaotic_enriched(2)
        assert ec is not None

    def test_z2_symmetric_satisfies_all_axioms(self):
        ec = _z2_symmetric_two_obj()
        assert ec is not None

    def test_associativity_explicitly(self):
        ec = _z2_symmetric_two_obj()
        ec._check_associativity()  # should not raise

    def test_left_unitality_explicitly(self):
        ec = _z2_symmetric_two_obj()
        ec._check_left_unitality()

    def test_right_unitality_explicitly(self):
        ec = _z2_symmetric_two_obj()
        ec._check_right_unitality()

    def test_bool_chaotic_three_objects_associativity(self):
        ec = _bool_chaotic_enriched(3)
        ec._check_associativity()

    def test_z2_four_objects_all_hom_zero(self):
        """Four-object category with all hom = 0 (trivial Z/2Z enrichment)."""
        V = _z2_V()
        C_cat = V.category
        objs = ["A", "B", "C", "D"]
        hom = {(X, Y): "0" for X in objs for Y in objs}
        comps = {
            (X, Y, Z): C_cat.identities["0"]
            for X in objs for Y in objs for Z in objs
        }
        ids = {X: C_cat.identities["0"] for X in objs}
        ec = FiniteEnrichedCategory(
            objects=objs, enriching=V, hom_objects=hom,
            compositions=comps, identity_elements=ids,
        )
        assert ec is not None


# ------------------------------------------------------------------ #
# Tests: discrete_enriched_category helper                            #
# ------------------------------------------------------------------ #

class TestDiscreteEnrichedCategory:
    def test_trivial_one_object_z2(self):
        V = _z2_V()
        ec = discrete_enriched_category(["*"], V, {("*", "*"): "0"})
        assert isinstance(ec, FiniteEnrichedCategory)

    def test_bool_chaotic_two_obj_via_helper(self):
        V = _bool_and_V()
        hom = {("A", "A"): "T", ("A", "B"): "T", ("B", "A"): "T", ("B", "B"): "T"}
        ec = discrete_enriched_category(["A", "B"], V, hom)
        assert ec.hom("A", "B") == "T"

    def test_z2_symmetric_via_helper(self):
        V = _z2_V()
        hom = {
            ("A", "A"): "0", ("A", "B"): "1",
            ("B", "A"): "1", ("B", "B"): "0",
        }
        ec = discrete_enriched_category(["A", "B"], V, hom)
        assert ec.hom("A", "B") == "1"

    def test_three_objects_z2_all_zero(self):
        V = _z2_V()
        objs = ["A", "B", "C"]
        hom = {(X, Y): "0" for X in objs for Y in objs}
        ec = discrete_enriched_category(objs, V, hom)
        assert ec is not None

    def test_incompatible_hom_matrix_raises(self):
        """A hom-matrix where the strict equation hom(B,C)⊗hom(A,B) = hom(A,C) fails."""
        V = _bool_and_V()
        # hom(B,A) = F but hom(A,A) = T: F∧T = F ≠ T → no morphism F→T exists
        hom = {
            ("A", "A"): "T", ("A", "B"): "T",
            ("B", "A"): "F", ("B", "B"): "T",
        }
        with pytest.raises(ValueError):
            discrete_enriched_category(["A", "B"], V, hom)


# ------------------------------------------------------------------ #
# Tests: underlying_category                                           #
# ------------------------------------------------------------------ #

class TestUnderlyingCategory:
    def test_returns_finite_category(self):
        ec = _trivial_single_obj()
        uc = ec.underlying_category()
        assert isinstance(uc, FiniteCategory)

    def test_objects_match(self):
        ec = _trivial_single_obj()
        uc = ec.underlying_category()
        assert set(uc.objects) == set(ec.objects)

    def test_bool_chaotic_underlying_has_morphisms(self):
        ec = _bool_chaotic_enriched(2)
        uc = ec.underlying_category()
        assert len(uc.morphisms) > 0

    def test_z2_underlying_category(self):
        ec = _z2_symmetric_two_obj()
        uc = ec.underlying_category()
        assert isinstance(uc, FiniteCategory)
