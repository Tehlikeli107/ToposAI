"""
Tests for topos_ai.hott — HoTT Path Groupoids, Dependent Families,
and Formal Homotopy Equivalence.
"""
import pytest
from topos_ai.hott import (
    FinitePathGroupoid,
    PathFamily,
    FormalHomotopyEquivalence,
)
from topos_ai.formal_category import FiniteCategory


# ------------------------------------------------------------------ #
# Shared fixtures                                                        #
# ------------------------------------------------------------------ #

def _interval_groupoid():
    """
    Interval groupoid  I = {0 ≃ 1}:
      objects: 0, 1
      paths:   r0: 0->0,  r1: 1->1,  p: 0->1,  q: 1->0
      p and q are mutual inverses (homotopy equivalence between 0 and 1).
    """
    return FinitePathGroupoid(
        objects=["0", "1"],
        paths={
            "r0": ("0", "0"),
            "r1": ("1", "1"),
            "p": ("0", "1"),
            "q": ("1", "0"),
        },
        identities={"0": "r0", "1": "r1"},
        inverses={"r0": "r0", "r1": "r1", "p": "q", "q": "p"},
        composition={
            ("r0", "r0"): "r0",
            ("r1", "r1"): "r1",
            ("p",  "r0"): "p",
            ("r1", "p"):  "p",
            ("q",  "r1"): "q",
            ("r0", "q"):  "q",
            ("q",  "p"):  "r0",
            ("p",  "q"):  "r1",
        },
    )


def _z2_groupoid():
    """
    One-object groupoid (group)  Z/2Z  = {e, a}  with a² = e.
    Convention: composition(after, before).
    """
    return FinitePathGroupoid(
        objects=["*"],
        paths={"e": ("*", "*"), "a": ("*", "*")},
        identities={"*": "e"},
        inverses={"e": "e", "a": "a"},
        composition={
            ("e", "e"): "e",
            ("a", "e"): "a",
            ("e", "a"): "a",
            ("a", "a"): "e",
        },
    )


def _trivial_groupoid():
    """Single-object, single-path groupoid (trivial group)."""
    return FinitePathGroupoid(
        objects=["pt"],
        paths={"id": ("pt", "pt")},
        identities={"pt": "id"},
        inverses={"id": "id"},
        composition={("id", "id"): "id"},
    )


def _interval_family(g=None):
    """PathFamily over the interval groupoid with fibers {x,y} and {u,v}."""
    if g is None:
        g = _interval_groupoid()
    return PathFamily(
        base=g,
        fibers={"0": frozenset(["x", "y"]), "1": frozenset(["u", "v"])},
        transports={
            "r0": {"x": "x", "y": "y"},
            "r1": {"u": "u", "v": "v"},
            "p":  {"x": "u", "y": "v"},
            "q":  {"u": "x", "v": "y"},
        },
    )


# ------------------------------------------------------------------ #
# FinitePathGroupoid — construction and basic accessors                #
# ------------------------------------------------------------------ #

class TestFinitePathGroupoid:
    def test_constructs_interval_groupoid(self):
        g = _interval_groupoid()
        assert isinstance(g, FinitePathGroupoid)

    def test_constructs_z2_groupoid(self):
        g = _z2_groupoid()
        assert isinstance(g, FinitePathGroupoid)

    def test_objects(self):
        g = _interval_groupoid()
        assert set(g.objects) == {"0", "1"}

    def test_source_target(self):
        g = _interval_groupoid()
        assert g.source("p") == "0"
        assert g.target("p") == "1"
        assert g.source("q") == "1"
        assert g.target("q") == "0"

    def test_refl(self):
        g = _interval_groupoid()
        assert g.refl("0") == "r0"
        assert g.refl("1") == "r1"

    def test_identity_type(self):
        g = _interval_groupoid()
        assert g.identity_type("0", "1") == frozenset(["p"])
        assert g.identity_type("0", "0") == frozenset(["r0"])

    def test_inverse(self):
        g = _interval_groupoid()
        assert g.inverse("p") == "q"
        assert g.inverse("q") == "p"
        assert g.inverse("r0") == "r0"

    def test_compose_path_with_identity(self):
        g = _interval_groupoid()
        assert g.compose("p", "r0") == "p"
        assert g.compose("r1", "p") == "p"

    def test_compose_inverse_gives_identity(self):
        g = _interval_groupoid()
        assert g.compose("q", "p") == "r0"
        assert g.compose("p", "q") == "r1"

    def test_compose_non_composable_raises(self):
        g = _interval_groupoid()
        with pytest.raises(ValueError):
            g.compose("p", "r1")   # target(r1)=1, source(p)=0 — mismatch

    def test_compose_missing_key_raises(self):
        """Composable pair not in composition table raises ValueError."""
        g = _interval_groupoid()
        # Manually break the composition table for a test.
        broken = FinitePathGroupoid.__new__(FinitePathGroupoid)
        broken.objects = g.objects
        broken.object_set = g.object_set
        broken.paths = g.paths
        broken.identities = g.identities
        broken.inverses = g.inverses
        # Leave out (q, p) -> r0 to trigger the KeyError path
        comp = dict(g.composition)
        del comp[("q", "p")]
        broken.composition = comp
        with pytest.raises(ValueError):
            broken.compose("q", "p")

    def test_z2_group_law(self):
        g = _z2_groupoid()
        assert g.compose("a", "a") == "e"
        assert g.compose("a", "e") == "a"


# ------------------------------------------------------------------ #
# FinitePathGroupoid — validation errors                               #
# ------------------------------------------------------------------ #

class TestPathGroupoidValidation:
    def test_missing_identity_raises(self):
        with pytest.raises(ValueError, match="reflexivity"):
            FinitePathGroupoid(
                objects=["A", "B"],
                paths={"rA": ("A", "A"), "f": ("A", "B"), "f_inv": ("B", "A")},
                identities={"A": "rA"},       # missing B
                inverses={"rA": "rA", "f": "f_inv", "f_inv": "f"},
                composition={
                    ("rA", "rA"): "rA",
                    ("f", "rA"): "f",
                    ("f_inv", "f"): "rA",
                },
            )

    def test_missing_inverse_raises(self):
        with pytest.raises(ValueError, match="inverse"):
            FinitePathGroupoid(
                objects=["A"],
                paths={"e": ("A", "A"), "x": ("A", "A")},
                identities={"A": "e"},
                inverses={"e": "e"},        # missing x
                composition={
                    ("e", "e"): "e",
                    ("x", "e"): "x",
                    ("e", "x"): "x",
                    ("x", "x"): "e",
                },
            )

    def test_identity_morphism_wrong_type_raises(self):
        """Identity path must have type obj -> obj."""
        with pytest.raises(ValueError):
            FinitePathGroupoid(
                objects=["A", "B"],
                paths={"rA": ("A", "B"), "rB": ("B", "B")},  # rA has wrong type
                identities={"A": "rA", "B": "rB"},
                inverses={"rA": "rA", "rB": "rB"},
                composition={
                    ("rA", "rA"): "rA",
                    ("rB", "rB"): "rB",
                    ("rA", "rB"): "rA",
                    ("rB", "rA"): "rA",
                },
            )

    def test_inverse_wrong_endpoints_raises(self):
        """Inverse of p: A->B must have type B->A."""
        with pytest.raises(ValueError):
            FinitePathGroupoid(
                objects=["A", "B"],
                paths={"rA": ("A", "A"), "rB": ("B", "B"), "p": ("A", "B"), "bad_inv": ("A", "B")},
                identities={"A": "rA", "B": "rB"},
                inverses={"rA": "rA", "rB": "rB", "p": "bad_inv", "bad_inv": "p"},
                composition={
                    ("rA", "rA"): "rA",
                    ("rB", "rB"): "rB",
                    ("p", "rA"): "p",
                    ("rB", "p"): "p",
                    ("bad_inv", "rB"): "bad_inv",
                    ("rA", "bad_inv"): "bad_inv",
                    ("bad_inv", "p"): "rA",
                    ("p", "bad_inv"): "rB",
                },
            )

    def test_trivial_groupoid_valid(self):
        g = _trivial_groupoid()
        assert g.refl("pt") == "id"

    def test_path_endpoint_outside_object_set_raises(self):
        """Path whose source/target is not in objects raises on line 66."""
        with pytest.raises(ValueError, match="endpoint"):
            FinitePathGroupoid(
                objects=["A"],
                paths={"e": ("A", "A"), "bad": ("A", "X")},  # X not in objects
                identities={"A": "e"},
                inverses={"e": "e", "bad": "bad"},
                composition={
                    ("e", "e"): "e",
                    ("bad", "e"): "bad",
                },
            )

    def test_inverse_not_declared_raises(self):
        """Inverse value pointing to a non-existent path name raises."""
        with pytest.raises(ValueError):
            FinitePathGroupoid(
                objects=["A"],
                paths={"e": ("A", "A"), "a": ("A", "A")},
                identities={"A": "e"},
                inverses={"e": "e", "a": "GHOST"},   # GHOST not in paths
                composition={
                    ("e", "e"): "e", ("a", "e"): "a",
                    ("e", "a"): "a", ("a", "a"): "e",
                },
            )

    def test_inverse_wrong_endpoints_raises(self):
        """Inverse path has wrong source/target."""
        with pytest.raises(ValueError):
            FinitePathGroupoid(
                objects=["A", "B"],
                paths={
                    "rA": ("A", "A"), "rB": ("B", "B"),
                    "f": ("A", "B"), "f_inv": ("A", "B"),   # wrong direction
                },
                identities={"A": "rA", "B": "rB"},
                inverses={"rA": "rA", "rB": "rB", "f": "f_inv", "f_inv": "f"},
                composition={
                    ("rA", "rA"): "rA", ("rB", "rB"): "rB",
                    ("f", "rA"): "f", ("rB", "f"): "f",
                    ("f_inv", "rA"): "f_inv", ("rB", "f_inv"): "f_inv",
                },
            )

    def test_inverse_involution_fails_raises(self):
        """inv(inv(a)) != a raises."""
        with pytest.raises(ValueError):
            FinitePathGroupoid(
                objects=["A"],
                paths={"e": ("A", "A"), "a": ("A", "A"), "b": ("A", "A")},
                identities={"A": "e"},
                inverses={"e": "e", "a": "b", "b": "b"},   # inv(b)=b but should be a
                composition={
                    ("e", "e"): "e", ("a", "e"): "a", ("b", "e"): "b",
                    ("e", "a"): "a", ("a", "a"): "b", ("b", "a"): "e",
                    ("e", "b"): "b", ("a", "b"): "e", ("b", "b"): "a",
                },
            )

    def test_composite_result_not_declared_raises(self):
        """Composition table maps to an undeclared path name."""
        with pytest.raises(ValueError):
            FinitePathGroupoid(
                objects=["A"],
                paths={"e": ("A", "A"), "a": ("A", "A"), "b": ("A", "A")},
                identities={"A": "e"},
                inverses={"e": "e", "a": "b", "b": "a"},
                composition={
                    ("e", "e"): "e", ("a", "e"): "a", ("b", "e"): "b",
                    ("e", "a"): "a", ("a", "a"): "GHOST",  # ← not declared
                    ("b", "a"): "e", ("e", "b"): "b", ("a", "b"): "e", ("b", "b"): "a",
                },
            )

    def test_composite_result_wrong_type_raises(self):
        """Composition maps to a path with incorrect source/target."""
        with pytest.raises(ValueError):
            FinitePathGroupoid(
                objects=["A", "B"],
                paths={
                    "rA": ("A", "A"), "rB": ("B", "B"),
                    "f": ("A", "B"), "g": ("B", "A"),
                },
                identities={"A": "rA", "B": "rB"},
                inverses={"rA": "rA", "rB": "rB", "f": "g", "g": "f"},
                composition={
                    ("rA", "rA"): "rA", ("rB", "rB"): "rB",
                    ("f", "rA"): "f", ("rB", "f"): "f",
                    ("g", "rB"): "g", ("rA", "g"): "g",
                    ("g", "f"): "rB",   # wrong: type (A,A) expected, rB has type (B,B)
                    ("f", "g"): "rB",
                },
            )

    def test_right_identity_law_failure_raises(self):
        """compose(path, id_src) != path triggers right identity error."""
        with pytest.raises(ValueError):
            FinitePathGroupoid(
                objects=["A"],
                paths={"e": ("A", "A"), "a": ("A", "A"), "b": ("A", "A")},
                identities={"A": "e"},
                inverses={"e": "e", "a": "b", "b": "a"},
                composition={
                    ("e", "e"): "e",
                    ("a", "e"): "b",    # ← wrong: should be "a"
                    ("b", "e"): "b", ("e", "a"): "a", ("a", "a"): "b",
                    ("b", "a"): "e", ("e", "b"): "b", ("a", "b"): "e", ("b", "b"): "a",
                },
            )

    def test_left_identity_law_failure_raises(self):
        """compose(id_dst, path) != path triggers left identity error."""
        with pytest.raises(ValueError):
            FinitePathGroupoid(
                objects=["A"],
                paths={"e": ("A", "A"), "a": ("A", "A"), "b": ("A", "A")},
                identities={"A": "e"},
                inverses={"e": "e", "a": "b", "b": "a"},
                composition={
                    ("e", "e"): "e", ("a", "e"): "a", ("b", "e"): "b",
                    ("e", "a"): "b",   # ← wrong: should be "a"
                    ("a", "a"): "b", ("b", "a"): "e",
                    ("e", "b"): "b", ("a", "b"): "e", ("b", "b"): "a",
                },
            )

    def test_left_inverse_law_failure_raises(self):
        """compose(inv(path), path) != id_src raises."""
        with pytest.raises(ValueError):
            FinitePathGroupoid(
                objects=["A"],
                paths={"e": ("A", "A"), "a": ("A", "A"), "b": ("A", "A")},
                identities={"A": "e"},
                inverses={"e": "e", "a": "b", "b": "a"},
                composition={
                    ("e", "e"): "e", ("a", "e"): "a", ("b", "e"): "b",
                    ("e", "a"): "a", ("a", "a"): "b",
                    ("b", "a"): "a",   # ← wrong: should be "e"
                    ("e", "b"): "b", ("a", "b"): "e", ("b", "b"): "a",
                },
            )

    def test_right_inverse_law_failure_raises(self):
        """compose(path, inv(path)) != id_dst raises."""
        with pytest.raises(ValueError):
            FinitePathGroupoid(
                objects=["A"],
                paths={"e": ("A", "A"), "a": ("A", "A"), "b": ("A", "A")},
                identities={"A": "e"},
                inverses={"e": "e", "a": "b", "b": "a"},
                composition={
                    ("e", "e"): "e", ("a", "e"): "a", ("b", "e"): "b",
                    ("e", "a"): "a", ("a", "a"): "b", ("b", "a"): "e",
                    ("e", "b"): "b",
                    ("a", "b"): "a",   # ← wrong: should be "e"
                    ("b", "b"): "a",
                },
            )

    def test_associativity_failure_raises(self):
        """Non-associative composition raises at line 107."""
        # Z/3Z structure but with (a,a)=b and (b,b)=b (non-associative)
        # (b ∘ a) ∘ a ≠ b ∘ (a ∘ a)
        with pytest.raises(ValueError, match="associativity"):
            FinitePathGroupoid(
                objects=["A"],
                paths={"e": ("A", "A"), "a": ("A", "A"), "b": ("A", "A")},
                identities={"A": "e"},
                inverses={"e": "e", "a": "b", "b": "a"},
                composition={
                    ("e", "e"): "e", ("a", "e"): "a", ("b", "e"): "b",
                    ("e", "a"): "a", ("a", "a"): "b", ("b", "a"): "e",
                    ("e", "b"): "b", ("a", "b"): "e",
                    ("b", "b"): "b",   # should be "a" for Z/3Z
                },
            )


# ------------------------------------------------------------------ #
# PathFamily — construction and transport                              #
# ------------------------------------------------------------------ #

class TestPathFamily:
    def test_constructs(self):
        fam = _interval_family()
        assert isinstance(fam, PathFamily)

    def test_fibers(self):
        fam = _interval_family()
        assert fam.fibers["0"] == frozenset(["x", "y"])
        assert fam.fibers["1"] == frozenset(["u", "v"])

    def test_transport_along_p(self):
        fam = _interval_family()
        assert fam.transport("p", "x") == "u"
        assert fam.transport("p", "y") == "v"

    def test_transport_along_q(self):
        fam = _interval_family()
        assert fam.transport("q", "u") == "x"
        assert fam.transport("q", "v") == "y"

    def test_transport_along_identity_is_id(self):
        fam = _interval_family()
        assert fam.transport("r0", "x") == "x"
        assert fam.transport("r1", "u") == "u"

    def test_transport_missing_value_raises(self):
        fam = _interval_family()
        with pytest.raises((KeyError, ValueError)):
            fam.transport("p", "z")   # z not in fiber of 0

    def test_transport_equivalence_roundtrip(self):
        fam = _interval_family()
        forward, backward = fam.transport_equivalence("p")
        assert forward["x"] == "u"
        assert backward["u"] == "x"

    def test_validate_transport_equivalences(self):
        fam = _interval_family()
        assert fam.validate_transport_equivalences() is True

    def test_functorial_transport_composition(self):
        """Transport along composite = composite of transports."""
        g = _interval_groupoid()
        fam = _interval_family(g)
        # transport along q∘p should equal transport(q) after transport(p)
        # q∘p = r0 (identity on 0)
        for val in ["x", "y"]:
            via_composite = fam.transport("r0", val)
            via_steps = fam.transport("q", fam.transport("p", val))
            assert via_composite == via_steps

    def test_wrong_fiber_keys_raises(self):
        """Fiber dict must have exactly the base objects."""
        g = _interval_groupoid()
        with pytest.raises(ValueError, match="fiber"):
            PathFamily(
                base=g,
                fibers={"0": frozenset(["x"])},     # missing "1"
                transports={
                    "r0": {"x": "x"},
                    "r1": {},
                    "p": {"x": "u"},
                    "q": {"u": "x"},
                },
            )

    def test_wrong_transport_domain_raises(self):
        """Transport map must be defined on the whole source fiber."""
        g = _interval_groupoid()
        with pytest.raises(ValueError):
            PathFamily(
                base=g,
                fibers={"0": frozenset(["x", "y"]), "1": frozenset(["u", "v"])},
                transports={
                    "r0": {"x": "x"},          # missing "y" — wrong domain
                    "r1": {"u": "u", "v": "v"},
                    "p":  {"x": "u", "y": "v"},
                    "q":  {"u": "x", "v": "y"},
                },
            )

    def test_constant_family(self):
        """A constant family over the trivial groupoid."""
        g = _trivial_groupoid()
        fam = PathFamily(
            base=g,
            fibers={"pt": frozenset(["a", "b"])},
            transports={"id": {"a": "a", "b": "b"}},
        )
        assert fam.transport("id", "a") == "a"
        assert fam.validate_transport_equivalences()

    def test_missing_transport_for_path_raises(self):
        """Transport dict must have an entry for every path."""
        g = _interval_groupoid()
        with pytest.raises(ValueError, match="transport map"):
            PathFamily(
                base=g,
                fibers={"0": frozenset(["x", "y"]), "1": frozenset(["u", "v"])},
                transports={
                    "r0": {"x": "x", "y": "y"},
                    "r1": {"u": "u", "v": "v"},
                    "p":  {"x": "u", "y": "v"},
                    # "q" is missing
                },
            )

    def test_transport_codomain_outside_fiber_raises(self):
        """Transport landing outside the target fiber raises."""
        g = _interval_groupoid()
        with pytest.raises(ValueError):
            PathFamily(
                base=g,
                fibers={"0": frozenset(["x", "y"]), "1": frozenset(["u", "v"])},
                transports={
                    "r0": {"x": "x", "y": "y"},
                    "r1": {"u": "u", "v": "v"},
                    "p":  {"x": "OUTSIDE", "y": "v"},   # "OUTSIDE" not in fibers["1"]
                    "q":  {"u": "x", "v": "y"},
                },
            )

    def test_transport_identity_not_identity_raises(self):
        """Transport along a reflexivity path must be the identity function."""
        g = _interval_groupoid()
        with pytest.raises(ValueError, match="refl"):
            PathFamily(
                base=g,
                fibers={"0": frozenset(["x", "y"]), "1": frozenset(["u", "v"])},
                transports={
                    "r0": {"x": "y", "y": "x"},   # swap — not identity on fiber["0"]
                    "r1": {"u": "u", "v": "v"},
                    "p":  {"x": "u", "y": "v"},
                    "q":  {"u": "x", "v": "y"},
                },
            )

    def test_transport_functoriality_failure_raises(self):
        """Transport(composite) != Transport(after) ∘ Transport(before) raises."""
        g = _interval_groupoid()
        # transport(q ∘ p = r0, x) = x  but  transport(q, transport(p, x)) = transport(q, u) = y ≠ x
        with pytest.raises(ValueError, match="functoriality"):
            PathFamily(
                base=g,
                fibers={"0": frozenset(["x", "y"]), "1": frozenset(["u", "v"])},
                transports={
                    "r0": {"x": "x", "y": "y"},
                    "r1": {"u": "u", "v": "v"},
                    "p":  {"x": "u", "y": "u"},   # non-injective: both go to u
                    "q":  {"u": "y", "v": "x"},   # q∘p should be identity but won't be
                },
            )


# ------------------------------------------------------------------ #
# FormalHomotopyEquivalence — isomorphism search                       #
# ------------------------------------------------------------------ #

def _arrow_cat(prefix=""):
    """Arrow category  A --> B (with identities)."""
    p = prefix
    return FiniteCategory(
        objects=[f"{p}A", f"{p}B"],
        morphisms={
            f"{p}idA": (f"{p}A", f"{p}A"),
            f"{p}idB": (f"{p}B", f"{p}B"),
            f"{p}f":   (f"{p}A", f"{p}B"),
        },
        identities={f"{p}A": f"{p}idA", f"{p}B": f"{p}idB"},
        composition={
            (f"{p}idA", f"{p}idA"): f"{p}idA",
            (f"{p}idB", f"{p}idB"): f"{p}idB",
            (f"{p}f",   f"{p}idA"): f"{p}f",
            (f"{p}idB", f"{p}f"):   f"{p}f",
        },
    )


class TestFormalHomotopyEquivalence:
    def test_identical_categories_are_isomorphic(self):
        C = _arrow_cat()
        equiv = FormalHomotopyEquivalence(C, C)
        assert equiv.is_univalent_equivalent()

    def test_isomorphic_relabeled_categories(self):
        """Two arrow categories with different labels are isomorphic."""
        C1 = _arrow_cat("x")
        C2 = _arrow_cat("y")
        equiv = FormalHomotopyEquivalence(C1, C2)
        result = equiv.find_strict_isomorphism()
        assert result is not None
        obj_map, mor_map = result
        # Functoriality check
        for m, (src, dst) in C1.morphisms.items():
            mapped_m = mor_map[m]
            assert C2.morphisms[mapped_m] == (obj_map[src], obj_map[dst])

    def test_non_isomorphic_different_object_counts(self):
        """Categories with different object counts cannot be isomorphic."""
        C1 = _arrow_cat()
        # Discrete 2-object category
        C3 = FiniteCategory(
            objects=["p", "q", "r"],
            morphisms={"idp": ("p","p"), "idq": ("q","q"), "idr": ("r","r")},
            identities={"p": "idp", "q": "idq", "r": "idr"},
            composition={
                ("idp","idp"): "idp", ("idq","idq"): "idq", ("idr","idr"): "idr",
            },
        )
        equiv = FormalHomotopyEquivalence(C1, C3)
        assert equiv.find_strict_isomorphism() is None
        assert not equiv.is_univalent_equivalent()

    def test_non_isomorphic_different_morphism_counts(self):
        """Arrow category vs discrete 2-object category are not isomorphic."""
        arrow = _arrow_cat()
        discrete = FiniteCategory(
            objects=["X", "Y"],
            morphisms={"idX": ("X","X"), "idY": ("Y","Y")},
            identities={"X": "idX", "Y": "idY"},
            composition={("idX","idX"): "idX", ("idY","idY"): "idY"},
        )
        equiv = FormalHomotopyEquivalence(arrow, discrete)
        assert equiv.find_strict_isomorphism() is None

    def test_single_object_categories_isomorphic(self):
        """Two single-object single-morphism categories are isomorphic."""
        C1 = FiniteCategory(
            objects=["a"],
            morphisms={"e": ("a","a")},
            identities={"a": "e"},
            composition={("e","e"): "e"},
        )
        C2 = FiniteCategory(
            objects=["z"],
            morphisms={"id": ("z","z")},
            identities={"z": "id"},
            composition={("id","id"): "id"},
        )
        equiv = FormalHomotopyEquivalence(C1, C2)
        result = equiv.find_strict_isomorphism()
        assert result is not None
        obj_map, mor_map = result
        assert obj_map["a"] == "z"
        assert mor_map["e"] == "id"

    def test_groupoid_self_isomorphism(self):
        """Every groupoid is isomorphic to itself."""
        g = _interval_groupoid()
        equiv = FormalHomotopyEquivalence(g, g)
        assert equiv.is_univalent_equivalent()

    def test_z2_trivial_not_isomorphic(self):
        """Z/2Z groupoid has 2 paths; trivial groupoid has 1 — not isomorphic."""
        z2 = _z2_groupoid()
        triv = _trivial_groupoid()
        equiv = FormalHomotopyEquivalence(z2, triv)
        assert equiv.find_strict_isomorphism() is None

    def test_find_returns_valid_object_and_morphism_maps(self):
        C1 = _arrow_cat("u_")
        C2 = _arrow_cat("v_")
        equiv = FormalHomotopyEquivalence(C1, C2)
        obj_map, mor_map = equiv.find_strict_isomorphism()
        # Every object in C1 maps to an object in C2
        assert set(obj_map.keys()) == set(C1.objects)
        assert set(obj_map.values()).issubset(set(C2.objects))
        # Every morphism in C1 maps to a morphism in C2
        assert set(mor_map.keys()) == set(C1.morphisms.keys())
        assert set(mor_map.values()).issubset(set(C2.morphisms.keys()))

    def test_same_size_non_isomorphic_composition_law_differs(self):
        """
        Two 1-object 2-morphism categories: one with a²=a (idempotent),
        one with b²=e (involution).  Same size, not isomorphic.
        The algorithm must reject all permutations via composition mismatch,
        exercising the preserves_comp = False / break branches.
        """
        # Category A: e=identity, a²=a (idempotent)
        A = FiniteCategory(
            objects=["*"],
            morphisms={"e": ("*", "*"), "a": ("*", "*")},
            identities={"*": "e"},
            composition={
                ("e", "e"): "e", ("a", "e"): "a",
                ("e", "a"): "a", ("a", "a"): "a",
            },
        )
        # Category B: e=identity, b²=e (involution)
        B = FiniteCategory(
            objects=["*"],
            morphisms={"e": ("*", "*"), "b": ("*", "*")},
            identities={"*": "e"},
            composition={
                ("e", "e"): "e", ("b", "e"): "b",
                ("e", "b"): "b", ("b", "b"): "e",
            },
        )
        equiv = FormalHomotopyEquivalence(A, B)
        assert equiv.find_strict_isomorphism() is None
        assert not equiv.is_univalent_equivalent()
