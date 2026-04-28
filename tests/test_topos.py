"""
Tests for topos_ai.topos — Elementary Topos Structures on FinSet.
"""
import pytest
from topos_ai.topos import (
    finset_product,
    product_projection_1,
    product_projection_2,
    product_morphism,
    finset_exponential,
    evaluation_morphism,
    curry,
    uncurry,
    all_finset_morphisms,
    verify_ccc,
    SubobjectClassifier,
    verify_subobject_classifier,
    TRUE, FALSE, OMEGA,
)


# ------------------------------------------------------------------ #
# Small test sets                                                        #
# ------------------------------------------------------------------ #

A2 = frozenset(["a1", "a2"])
B2 = frozenset(["b1", "b2"])
C3 = frozenset(["c1", "c2", "c3"])
SING = frozenset(["s"])
EMPTY: frozenset = frozenset()


# ------------------------------------------------------------------ #
# Products                                                              #
# ------------------------------------------------------------------ #

class TestFinSetProduct:
    def test_product_size(self):
        assert len(finset_product(A2, B2)) == 4

    def test_product_elements(self):
        P = finset_product(A2, B2)
        assert ("a1","b1") in P and ("a2","b2") in P

    def test_product_with_singleton(self):
        P = finset_product(A2, SING)
        assert len(P) == 2

    def test_product_with_empty(self):
        assert finset_product(A2, EMPTY) == frozenset()

    def test_projection_1(self):
        pi1 = dict(product_projection_1(A2, B2))
        for (a, b), img in pi1.items():
            assert img == a

    def test_projection_2(self):
        pi2 = dict(product_projection_2(A2, B2))
        for (a, b), img in pi2.items():
            assert img == b

    def test_product_morphism_pairing(self):
        # <id, const_b1>: A2 -> A2 x B2
        f = frozenset((a, a) for a in A2)        # id_A
        g = frozenset((a, "b1") for a in A2)     # const b1
        pair = dict(product_morphism(f, g, A2, B2))
        for a in A2:
            assert pair[a] == (a, "b1")


# ------------------------------------------------------------------ #
# Exponential objects                                                    #
# ------------------------------------------------------------------ #

class TestExponential:
    def test_exponential_size(self):
        ZY = finset_exponential(B2, C3)
        # |C3^B2| = 3^2 = 9
        assert len(ZY) == 9

    def test_exponential_elements_are_functions(self):
        ZY = finset_exponential(B2, C3)
        for func in ZY:
            fd = dict(func)
            assert set(fd.keys()) == B2
            assert all(v in C3 for v in fd.values())

    def test_exponential_singleton_domain(self):
        ZY = finset_exponential(SING, C3)
        # 3^1 = 3
        assert len(ZY) == 3

    def test_exponential_empty_domain(self):
        ZY = finset_exponential(EMPTY, C3)
        # One function: the empty function
        assert len(ZY) == 1

    def test_evaluation_morphism_applies_function(self):
        ev = dict(evaluation_morphism(B2, C3))
        ZY = finset_exponential(B2, C3)
        for h in ZY:
            h_dict = dict(h)
            for b in B2:
                assert ev[(h, b)] == h_dict[b]


# ------------------------------------------------------------------ #
# Curry and uncurry                                                      #
# ------------------------------------------------------------------ #

class TestCurryUncurry:
    def _build_f(self, X, Y, Z):
        """Build a specific morphism f: X×Y -> Z."""
        XY = finset_product(X, Y)
        Z_sorted = sorted(Z, key=str)
        XY_sorted = sorted(XY, key=str)
        # Map each pair deterministically
        assignments = [Z_sorted[i % len(Z_sorted)] for i in range(len(XY_sorted))]
        return frozenset(zip(XY_sorted, assignments))

    def test_curry_produces_function_to_ZY(self):
        f = self._build_f(A2, B2, C3)
        g = dict(curry(f, A2, B2, C3))
        ZY = finset_exponential(B2, C3)
        for a in A2:
            assert g[a] in ZY

    def test_uncurry_curry_roundtrip(self):
        f = self._build_f(A2, B2, C3)
        g = curry(f, A2, B2, C3)
        f_back = uncurry(g, A2, B2, C3)
        assert f_back == f

    def test_curry_uncurry_roundtrip(self):
        # Build a g: A2 -> C3^B2 first
        ZY = finset_exponential(B2, C3)
        ZY_sorted = sorted(ZY, key=str)
        A2_sorted = sorted(A2, key=str)
        g = frozenset(zip(A2_sorted, ZY_sorted[:len(A2_sorted)]))
        f = uncurry(g, A2, B2, C3)
        g_back = curry(f, A2, B2, C3)
        assert g_back == g

    def test_curry_constant_function(self):
        """Constant f: X×Y -> {c} should curry to constant g: X -> (Y -> {c})."""
        Z1 = frozenset(["c"])
        f = frozenset(((a, b), "c") for a in A2 for b in B2)
        g = dict(curry(f, A2, B2, Z1))
        for a in A2:
            for b in B2:
                assert dict(g[a])[b] == "c"


# ------------------------------------------------------------------ #
# verify_ccc                                                             #
# ------------------------------------------------------------------ #

class TestVerifyCCC:
    def test_frontier3_scenario(self):
        """Reproduce Frontier 3: |Hom(X×Y,Z)| = |Hom(X,Z^Y)|."""
        X = frozenset(["x1","x2"])
        Y = frozenset(["y1","y2"])
        Z = frozenset(["z1","z2","z3"])
        assert verify_ccc(X, Y, Z)

    def test_singleton_sets(self):
        assert verify_ccc(SING, SING, SING)

    def test_asymmetric_sizes(self):
        assert verify_ccc(A2, C3, B2)

    def test_correct_cardinality(self):
        X = frozenset(["x1","x2"])
        Y = frozenset(["y1","y2"])
        Z = frozenset(["z1","z2","z3"])
        hom_XY_Z = all_finset_morphisms(finset_product(X, Y), Z)
        hom_X_ZY = all_finset_morphisms(X, finset_exponential(Y, Z))
        # Both should be 3^4 = 81
        assert len(hom_XY_Z) == len(hom_X_ZY) == 81


# ------------------------------------------------------------------ #
# SubobjectClassifier                                                    #
# ------------------------------------------------------------------ #

class TestSubobjectClassifier:
    def test_omega_is_two_element(self):
        sc = SubobjectClassifier()
        assert sc.Omega == frozenset({TRUE, FALSE})

    def test_characteristic_morphism_marks_subset(self):
        sc = SubobjectClassifier()
        X = frozenset(["x1","x2","x3","x4"])
        A = frozenset(["x2","x4"])
        mono = frozenset((a, a) for a in A)
        chi = dict(sc.characteristic_morphism(X, mono))
        assert chi["x2"] == TRUE
        assert chi["x4"] == TRUE
        assert chi["x1"] == FALSE
        assert chi["x3"] == FALSE

    def test_characteristic_full_subset(self):
        sc = SubobjectClassifier()
        X = frozenset(["x"])
        A = X
        mono = frozenset((a, a) for a in A)
        chi = dict(sc.characteristic_morphism(X, mono))
        assert chi["x"] == TRUE

    def test_characteristic_empty_subset(self):
        sc = SubobjectClassifier()
        X = frozenset(["x","y"])
        A = frozenset()
        mono = frozenset()
        chi = dict(sc.characteristic_morphism(X, mono))
        assert all(v == FALSE for v in chi.values())

    def test_verify_pullback_valid(self):
        sc = SubobjectClassifier()
        X = frozenset(["x1","x2","x3","x4"])
        A = frozenset(["x2","x4"])
        mono = frozenset((a, a) for a in A)
        assert sc.verify_pullback(X, A, mono)

    def test_verify_pullback_unique(self):
        """There must be exactly one valid χ for any inclusion."""
        sc = SubobjectClassifier()
        X = frozenset(["p","q","r"])
        A = frozenset(["p","r"])
        mono = frozenset((a, a) for a in A)
        assert sc.verify_pullback(X, A, mono)

    def test_verify_pullback_singleton_subset(self):
        sc = SubobjectClassifier()
        X = frozenset(["a","b","c"])
        A = frozenset(["b"])
        mono = frozenset((a, a) for a in A)
        assert sc.verify_pullback(X, A, mono)

    def test_repr(self):
        sc = SubobjectClassifier()
        assert "SubobjectClassifier" in repr(sc)


# ------------------------------------------------------------------ #
# verify_subobject_classifier                                            #
# ------------------------------------------------------------------ #

class TestVerifySubobjectClassifier:
    def test_frontier4_scenario(self):
        """Reproduce Frontier 4: A={x2,x4} ⊂ X={x1,x2,x3,x4}."""
        X = frozenset(["x1","x2","x3","x4"])
        A = frozenset(["x2","x4"])
        assert verify_subobject_classifier(X, A)

    def test_full_subset(self):
        X = frozenset(["a","b"])
        assert verify_subobject_classifier(X, X)

    def test_empty_subset(self):
        X = frozenset(["a","b"])
        assert verify_subobject_classifier(X, frozenset())

    def test_singleton_A(self):
        X = frozenset(["a","b","c"])
        A = frozenset(["b"])
        assert verify_subobject_classifier(X, A)

    def test_A_not_subset_raises(self):
        X = frozenset(["a","b"])
        A = frozenset(["c"])   # c not in X
        with pytest.raises(ValueError):
            verify_subobject_classifier(X, A)
