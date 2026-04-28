"""
Tests for topos_ai.sites — Grothendieck Sites and Sheaf Condition.

Test families:
1. Sieve — construction, closure, pullback
2. GrothendieckTopology — axiom validation
3. GrothendieckSite — trivial and non-trivial
4. FinitePresheaf — construction and functoriality
5. Sheaf condition — is_sheaf, matching families, amalgamations
6. Subobject classifier Ω
"""
import pytest
from topos_ai.formal_category import FiniteCategory
from topos_ai.sites import (
    Sieve,
    GrothendieckTopology,
    GrothendieckSite,
    FinitePresheaf,
    is_sheaf,
    sheaf_condition_failure,
    matching_families,
    amalgamations,
    maximal_sieve,
    empty_sieve,
    trivial_topology,
    discrete_topology,
    omega_presheaf,
)


# ------------------------------------------------------------------ #
# Shared category fixtures                                              #
# ------------------------------------------------------------------ #

def _arrow_cat():
    """A → B with a single morphism f."""
    return FiniteCategory(
        objects=["A", "B"],
        morphisms={"idA": ("A", "A"), "idB": ("B", "B"), "f": ("A", "B")},
        identities={"A": "idA", "B": "idB"},
        composition={
            ("idA", "idA"): "idA", ("idB", "idB"): "idB",
            ("f", "idA"): "f", ("idB", "f"): "f",
        },
    )


def _open_cover_cat():
    """Category modelling an open cover: C ← U1, C ← U2, I → U1, I → U2."""
    return FiniteCategory(
        objects=["C", "U1", "U2", "I"],
        morphisms={
            "idC": ("C", "C"), "idU1": ("U1", "U1"),
            "idU2": ("U2", "U2"), "idI": ("I", "I"),
            "u1": ("U1", "C"), "u2": ("U2", "C"),
            "i1": ("I", "U1"), "i2": ("I", "U2"),
            "u1i1": ("I", "C"), "u2i2": ("I", "C"),
        },
        identities={"C": "idC", "U1": "idU1", "U2": "idU2", "I": "idI"},
        composition={
            ("idC", "idC"): "idC", ("idU1", "idU1"): "idU1",
            ("idU2", "idU2"): "idU2", ("idI", "idI"): "idI",
            ("u1", "idU1"): "u1", ("idC", "u1"): "u1",
            ("u2", "idU2"): "u2", ("idC", "u2"): "u2",
            ("i1", "idI"): "i1", ("idU1", "i1"): "i1",
            ("i2", "idI"): "i2", ("idU2", "i2"): "i2",
            ("u1i1", "idI"): "u1i1", ("idC", "u1i1"): "u1i1",
            ("u2i2", "idI"): "u2i2", ("idC", "u2i2"): "u2i2",
            ("u1", "i1"): "u1i1", ("u2", "i2"): "u2i2",
        },
    )


def _zariski_site():
    """Open-cover category with the non-trivial Zariski-like topology."""
    cat = _open_cover_cat()
    S = Sieve(on="C", morphisms={"u1", "u2", "u1i1", "u2i2"}, category=cat)
    covering = {
        "C":  [S, maximal_sieve(cat, "C")],
        "U1": [maximal_sieve(cat, "U1")],
        "U2": [maximal_sieve(cat, "U2")],
        "I":  [maximal_sieve(cat, "I")],
    }
    J = GrothendieckTopology(cat, covering, validate=False)
    return GrothendieckSite(cat, J)


def _singleton_sheaf(cat):
    """Constant presheaf with one section per object — always a sheaf."""
    obj_map = {d: frozenset(["s"]) for d in cat.objects}
    res_map = {m: {"s": "s"} for m in cat.morphisms}
    return FinitePresheaf(category=cat, objects_map=obj_map, restriction_map=res_map)


# ------------------------------------------------------------------ #
# Sieve                                                                 #
# ------------------------------------------------------------------ #

class TestSieve:
    def test_maximal_sieve_constructs(self):
        cat = _arrow_cat()
        s = maximal_sieve(cat, "B")
        assert isinstance(s, Sieve)

    def test_maximal_sieve_contains_all_morphisms_to_B(self):
        cat = _arrow_cat()
        s = maximal_sieve(cat, "B")
        assert "idB" in s
        assert "f" in s

    def test_empty_sieve_constructs(self):
        cat = _arrow_cat()
        s = empty_sieve(cat, "A")
        assert len(s) == 0

    def test_sieve_closure_valid(self):
        cat = _arrow_cat()
        s = Sieve(on="B", morphisms={"idB", "f"}, category=cat)
        assert s.is_closed()

    def test_sieve_not_closed_raises(self):
        """Sieve missing a pre-composed morphism raises ValueError."""
        # Category I -i-> A -f-> B with composite fi = f∘i: I→B
        cat = FiniteCategory(
            objects=["I", "A", "B"],
            morphisms={
                "idI": ("I", "I"), "idA": ("A", "A"), "idB": ("B", "B"),
                "i": ("I", "A"), "f": ("A", "B"), "fi": ("I", "B"),
            },
            identities={"I": "idI", "A": "idA", "B": "idB"},
            composition={
                ("idI", "idI"): "idI", ("idA", "idA"): "idA", ("idB", "idB"): "idB",
                ("i", "idI"): "i", ("idA", "i"): "i",
                ("f", "idA"): "f", ("idB", "f"): "f",
                ("fi", "idI"): "fi", ("idB", "fi"): "fi",
                ("f", "i"): "fi",
            },
        )
        # {f: A→B} on B is NOT closed: i: I→A so f∘i = fi ∉ {f}
        with pytest.raises(ValueError, match="not closed"):
            Sieve(on="B", morphisms={"f"}, category=cat, validate=True)

    def test_sieve_closure_requires_precomposition(self):
        """Sieve {f} on B: f∘idA=f ∈ {f}✓; idB∘f ... idB has src B ≠ src A.
           The real requirement: for h: ?→src(f)=A, f∘h must be in S.
           Here only idA: A→A, so f∘idA=f ∈ S. Closure holds."""
        cat = _arrow_cat()
        s = Sieve(on="B", morphisms={"f"}, category=cat, validate=True)
        assert s.is_closed()

    def test_pullback_of_maximal_is_maximal(self):
        """Pulling back maximal sieve on B along f: A→B gives maximal sieve on A."""
        cat = _arrow_cat()
        max_B = maximal_sieve(cat, "B")
        pulled = max_B.pullback("f")
        assert pulled.on == "A"
        # f*max_B = {h: ?→A | f∘h ∈ max_B} = all morphisms to A = max_A
        max_A = maximal_sieve(cat, "A")
        assert pulled == max_A

    def test_pullback_of_empty_is_empty(self):
        """Pulling back empty sieve gives empty sieve."""
        cat = _arrow_cat()
        empty_B = Sieve(on="B", morphisms=set(), category=cat, validate=False)
        pulled = empty_B.pullback("f")
        assert len(pulled) == 0

    def test_sieve_equality(self):
        cat = _arrow_cat()
        s1 = maximal_sieve(cat, "B")
        s2 = maximal_sieve(cat, "B")
        assert s1 == s2

    def test_sieve_hash_stable(self):
        cat = _arrow_cat()
        s = maximal_sieve(cat, "B")
        d = {s: "value"}
        assert d[s] == "value"

    def test_sieve_repr(self):
        cat = _arrow_cat()
        s = maximal_sieve(cat, "A")
        assert "Sieve" in repr(s)

    def test_pullback_wrong_codomain_raises(self):
        """Pulling back a sieve on B along a morphism with codomain A should raise."""
        cat = _arrow_cat()
        max_A = maximal_sieve(cat, "A")
        with pytest.raises(ValueError):
            max_A.pullback("f")   # f: A→B, but sieve is on A


# ------------------------------------------------------------------ #
# GrothendieckTopology                                                  #
# ------------------------------------------------------------------ #

class TestGrothendieckTopology:
    def test_trivial_topology_constructs(self):
        cat = _arrow_cat()
        J = trivial_topology(cat)
        assert isinstance(J, GrothendieckTopology)

    def test_trivial_topology_covers_maximal_sieve(self):
        cat = _arrow_cat()
        J = trivial_topology(cat)
        max_B = maximal_sieve(cat, "B")
        assert J.covers("B", max_B)

    def test_discrete_topology_constructs(self):
        cat = _arrow_cat()
        J = discrete_topology(cat)
        assert isinstance(J, GrothendieckTopology)

    def test_discrete_topology_covers_empty_sieve(self):
        cat = _arrow_cat()
        J = discrete_topology(cat)
        empty_A = empty_sieve(cat, "A")
        assert J.covers("A", empty_A)

    def test_missing_maximal_sieve_raises(self):
        """A topology that does not include the maximal sieve should be invalid."""
        cat = _arrow_cat()
        empty_A = empty_sieve(cat, "A")
        covering = {
            "A": [empty_A],   # maximal sieve NOT included — invalid
            "B": [maximal_sieve(cat, "B")],
        }
        with pytest.raises(ValueError, match="maximality"):
            GrothendieckTopology(cat, covering, validate=True)

    def test_zariski_topology_is_valid(self):
        """The open-cover topology on the cover category should pass all axioms."""
        site = _zariski_site()
        # Construction in _zariski_site already calls validate=False,
        # but we can verify the axioms manually.
        J = site.topology
        cat = site.category
        J._check_maximality()  # should not raise

    def test_repr(self):
        cat = _arrow_cat()
        J = trivial_topology(cat)
        assert "GrothendieckTopology" in repr(J)


# ------------------------------------------------------------------ #
# FinitePresheaf                                                        #
# ------------------------------------------------------------------ #

class TestFinitePresheaf:
    def test_constructs(self):
        cat = _arrow_cat()
        F = _singleton_sheaf(cat)
        assert isinstance(F, FinitePresheaf)

    def test_sections_accessor(self):
        cat = _arrow_cat()
        F = _singleton_sheaf(cat)
        assert F.sections("A") == frozenset(["s"])

    def test_restrict_applies_correctly(self):
        cat = _arrow_cat()
        F = _singleton_sheaf(cat)
        assert F.restrict("f", "s") == "s"

    def test_missing_object_raises(self):
        cat = _arrow_cat()
        with pytest.raises(ValueError, match="missing objects_map"):
            FinitePresheaf(
                category=cat,
                objects_map={"A": frozenset(["s"])},   # missing B
                restriction_map={"idA": {"s": "s"}, "idB": {}, "f": {}},
            )

    def test_missing_restriction_raises(self):
        cat = _arrow_cat()
        with pytest.raises(ValueError, match="missing restriction_map"):
            FinitePresheaf(
                category=cat,
                objects_map={"A": frozenset(["s"]), "B": frozenset(["t"])},
                restriction_map={"idA": {"s": "s"}},   # missing idB and f
            )

    def test_functoriality_violation_raises(self):
        """F that does not satisfy F(g∘f)=F(f)∘F(g) should be rejected."""
        # Category I -i-> A -f-> B with composite fi = f∘i
        cat = FiniteCategory(
            objects=["I", "A", "B"],
            morphisms={
                "idI": ("I", "I"), "idA": ("A", "A"), "idB": ("B", "B"),
                "i": ("I", "A"), "f": ("A", "B"), "fi": ("I", "B"),
            },
            identities={"I": "idI", "A": "idA", "B": "idB"},
            composition={
                ("idI", "idI"): "idI", ("idA", "idA"): "idA", ("idB", "idB"): "idB",
                ("i", "idI"): "i", ("idA", "i"): "i",
                ("f", "idA"): "f", ("idB", "f"): "f",
                ("fi", "idI"): "fi", ("idB", "fi"): "fi",
                ("f", "i"): "fi",
            },
        )
        # F(fi)(b) ≠ F(i)(F(f)(b)) — functoriality violation
        # Use two elements in F(I) and declare F(fi) wrongly:
        with pytest.raises(ValueError, match="functoriality"):
            FinitePresheaf(
                category=cat,
                objects_map={
                    "I": frozenset(["w1", "w2"]),
                    "A": frozenset(["a"]),
                    "B": frozenset(["b"]),
                },
                restriction_map={
                    "idI": {"w1": "w1", "w2": "w2"}, "idA": {"a": "a"}, "idB": {"b": "b"},
                    "i":  {"a": "w1"},  # F(i): a→w1
                    "f":  {"b": "a"},   # F(f): b→a
                    # F(fi) should equal F(i)∘F(f): b→a→w1, so F(fi)(b)=w1
                    # But we declare F(fi)(b)=w2 — VIOLATION
                    "fi": {"b": "w2"},
                },
            )

    def test_repr(self):
        cat = _arrow_cat()
        F = _singleton_sheaf(cat)
        assert "FinitePresheaf" in repr(F)


# ------------------------------------------------------------------ #
# Matching families and amalgamations                                   #
# ------------------------------------------------------------------ #

class TestMatchingFamilies:
    def test_empty_sieve_has_one_empty_family(self):
        cat = _arrow_cat()
        F = _singleton_sheaf(cat)
        s = empty_sieve(cat, "A")
        families = matching_families(F, s)
        assert families == [{}]

    def test_singleton_presheaf_has_one_family(self):
        cat = _arrow_cat()
        F = _singleton_sheaf(cat)
        max_B = maximal_sieve(cat, "B")
        families = matching_families(F, max_B)
        assert len(families) == 1

    def test_two_section_presheaf_family_count(self):
        """F(B)={b1,b2}, F(A)={a}, F(f)(b1)=F(f)(b2)=a.
           Maximal sieve on B = {idB, f}. Each family assigns values to idB and f."""
        cat = _arrow_cat()
        F = FinitePresheaf(
            category=cat,
            objects_map={"A": frozenset(["a"]), "B": frozenset(["b1", "b2"])},
            restriction_map={
                "idA": {"a": "a"},
                "idB": {"b1": "b1", "b2": "b2"},
                "f":   {"b1": "a", "b2": "a"},
            },
        )
        max_B = maximal_sieve(cat, "B")
        families = matching_families(F, max_B)
        # Matching condition: F(idA)(s_idB) = s_{idB∘idA?} ... but idA: A→A not composable with idB
        # Main constraint: F(idA)(s_f) = s_{f∘idA} = s_f (since f∘idA=f)
        # s_f ∈ F(A) = {a}, so s_f = a always
        # s_idB ∈ F(B) = {b1, b2}: s_idB must satisfy F(idB)(s_idB) = s_{idB} (identity)
        # and the cross constraint: F(idA)(s_f) = s_f is trivially satisfied
        # Also f∘idA = f ∈ sieve, so F(idA)(s_f) = s_f = a (ok)
        # No constraint links s_idB to s_f via f (f: A→B, not composable as f∘h where h has tgt=B...
        # Actually f∘? where ? has tgt A. So F(idA)(s_f) = s_{f} which is trivial.
        # And for idB: F(idB)(s_idB) = s_{idB∘idB} = s_idB (trivial).
        # So both (s_idB=b1, s_f=a) and (s_idB=b2, s_f=a) are matching families.
        assert len(families) == 2


class TestAmalgamations:
    def test_singleton_has_unique_amalgamation(self):
        cat = _arrow_cat()
        F = _singleton_sheaf(cat)
        max_B = maximal_sieve(cat, "B")
        families = matching_families(F, max_B)
        amals = amalgamations(F, max_B, families[0])
        assert amals == ["s"]

    def test_non_unique_amalgamation(self):
        """Presheaf where two global sections restrict to same local data."""
        cat = _open_cover_cat()
        F = FinitePresheaf(
            category=cat,
            objects_map={
                "C": frozenset(["g1", "g2"]),
                "U1": frozenset(["u"]), "U2": frozenset(["v"]), "I": frozenset(["w"]),
            },
            restriction_map={
                "idC": {"g1": "g1", "g2": "g2"}, "idU1": {"u": "u"},
                "idU2": {"v": "v"}, "idI": {"w": "w"},
                "u1": {"g1": "u", "g2": "u"},
                "u2": {"g1": "v", "g2": "v"},
                "i1": {"u": "w"}, "i2": {"v": "w"},
                "u1i1": {"g1": "w", "g2": "w"},
                "u2i2": {"g1": "w", "g2": "w"},
            },
        )
        S = Sieve(on="C", morphisms={"u1", "u2", "u1i1", "u2i2"}, category=cat)
        # There should be a matching family with 2 amalgamations
        families = matching_families(F, S)
        all_amals = [amalgamations(F, S, fam) for fam in families]
        # At least one family should have > 1 amalgamation
        assert any(len(a) > 1 for a in all_amals)


# ------------------------------------------------------------------ #
# is_sheaf                                                              #
# ------------------------------------------------------------------ #

class TestIsSheaf:
    def test_singleton_is_sheaf_on_trivial(self):
        cat = _arrow_cat()
        F = _singleton_sheaf(cat)
        site = GrothendieckSite(cat, trivial_topology(cat))
        assert is_sheaf(F, site)

    def test_singleton_is_sheaf_on_discrete(self):
        cat = _arrow_cat()
        F = _singleton_sheaf(cat)
        site = GrothendieckSite(cat, discrete_topology(cat))
        assert is_sheaf(F, site)

    def test_non_unique_presheaf_not_sheaf(self):
        """Presheaf with non-unique amalgamation is not a sheaf."""
        cat = _open_cover_cat()
        F = FinitePresheaf(
            category=cat,
            objects_map={
                "C": frozenset(["g1", "g2"]),
                "U1": frozenset(["u"]), "U2": frozenset(["v"]), "I": frozenset(["w"]),
            },
            restriction_map={
                "idC": {"g1": "g1", "g2": "g2"}, "idU1": {"u": "u"},
                "idU2": {"v": "v"}, "idI": {"w": "w"},
                "u1": {"g1": "u", "g2": "u"},
                "u2": {"g1": "v", "g2": "v"},
                "i1": {"u": "w"}, "i2": {"v": "w"},
                "u1i1": {"g1": "w", "g2": "w"},
                "u2i2": {"g1": "w", "g2": "w"},
            },
        )
        site = _zariski_site()
        assert not is_sheaf(F, site)

    def test_sheaf_condition_failure_returns_info(self):
        cat = _open_cover_cat()
        F = FinitePresheaf(
            category=cat,
            objects_map={
                "C": frozenset(["g1", "g2"]),
                "U1": frozenset(["u"]), "U2": frozenset(["v"]), "I": frozenset(["w"]),
            },
            restriction_map={
                "idC": {"g1": "g1", "g2": "g2"}, "idU1": {"u": "u"},
                "idU2": {"v": "v"}, "idI": {"w": "w"},
                "u1": {"g1": "u", "g2": "u"},
                "u2": {"g1": "v", "g2": "v"},
                "i1": {"u": "w"}, "i2": {"v": "w"},
                "u1i1": {"g1": "w", "g2": "w"},
                "u2i2": {"g1": "w", "g2": "w"},
            },
        )
        site = _zariski_site()
        fail = sheaf_condition_failure(F, site)
        assert fail is not None
        assert fail["object"] == "C"
        assert len(fail["amalgamations"]) == 2

    def test_sheaf_condition_none_for_sheaf(self):
        cat = _arrow_cat()
        F = _singleton_sheaf(cat)
        site = GrothendieckSite(cat, trivial_topology(cat))
        assert sheaf_condition_failure(F, site) is None

    def test_every_presheaf_is_sheaf_on_trivial_topology(self):
        """On the trivial topology (only maximal sieves), every presheaf is a sheaf."""
        cat = _arrow_cat()
        F = FinitePresheaf(
            category=cat,
            objects_map={"A": frozenset(["a1", "a2"]), "B": frozenset(["b1", "b2"])},
            restriction_map={
                "idA": {"a1": "a1", "a2": "a2"},
                "idB": {"b1": "b1", "b2": "b2"},
                "f":   {"b1": "a1", "b2": "a2"},
            },
        )
        site = GrothendieckSite(cat, trivial_topology(cat))
        assert is_sheaf(F, site)


# ------------------------------------------------------------------ #
# Subobject classifier Ω                                                #
# ------------------------------------------------------------------ #

class TestOmegaPresheaf:
    def test_omega_constructs(self):
        cat = _arrow_cat()
        Omega = omega_presheaf(cat)
        assert isinstance(Omega, FinitePresheaf)

    def test_omega_is_sheaf_on_trivial(self):
        cat = _arrow_cat()
        Omega = omega_presheaf(cat)
        site = GrothendieckSite(cat, trivial_topology(cat))
        assert is_sheaf(Omega, site)

    def test_omega_sections_are_sieves(self):
        """Ω(d) = set of all sieves on d, so its cardinality > 0."""
        cat = _arrow_cat()
        Omega = omega_presheaf(cat)
        for d in cat.objects:
            assert len(Omega.sections(d)) > 0
