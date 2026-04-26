"""
Tests for topos_ai.formal_yoneda — Yoneda Lemma.
"""
import pytest
from topos_ai.formal_category import FiniteCategory
from topos_ai.formal_kan import FiniteSetFunctor
from topos_ai.formal_yoneda import (
    representable_functor,
    yoneda_evaluate,
    yoneda_inverse,
    verify_yoneda,
    verify_yoneda_naturality_in_A,
)


# ------------------------------------------------------------------ #
# Shared fixtures                                                        #
# ------------------------------------------------------------------ #

def _chain_cat():
    """Category 0 -f-> 1 -g-> 2 with composite h = g∘f."""
    return FiniteCategory(
        objects=["0", "1", "2"],
        morphisms={
            "id_0": ("0","0"), "id_1": ("1","1"), "id_2": ("2","2"),
            "f": ("0","1"), "g": ("1","2"), "h": ("0","2"),
        },
        identities={"0": "id_0", "1": "id_1", "2": "id_2"},
        composition={
            ("id_0","id_0"): "id_0", ("id_1","id_1"): "id_1", ("id_2","id_2"): "id_2",
            ("f","id_0"): "f", ("id_1","f"): "f",
            ("g","id_1"): "g", ("id_2","g"): "g",
            ("h","id_0"): "h", ("id_2","h"): "h",
            ("g","f"): "h",
        },
    )


def _F_functor(cat):
    """F: C -> FinSet from the Yoneda experiment."""
    return FiniteSetFunctor(
        category=cat,
        objects_map={
            "0": frozenset(["a","b"]),
            "1": frozenset(["x","y","z"]),
            "2": frozenset(["u","v"]),
        },
        morphism_map={
            "id_0": {"a":"a","b":"b"},
            "id_1": {"x":"x","y":"y","z":"z"},
            "id_2": {"u":"u","v":"v"},
            "f": {"a":"x","b":"y"},
            "g": {"x":"u","y":"v","z":"v"},
            "h": {"a":"u","b":"v"},
        },
    )


def _arrow_cat():
    return FiniteCategory(
        objects=["A","B"],
        morphisms={"idA":("A","A"),"idB":("B","B"),"f":("A","B")},
        identities={"A":"idA","B":"idB"},
        composition={("idA","idA"):"idA",("idB","idB"):"idB",
                     ("f","idA"):"f",("idB","f"):"f"},
    )


# ------------------------------------------------------------------ #
# representable_functor                                                  #
# ------------------------------------------------------------------ #

class TestRepresentableFunctor:
    def test_constructs(self):
        cat = _chain_cat()
        h0 = representable_functor(cat, "0")
        assert isinstance(h0, FiniteSetFunctor)

    def test_hom_sets_at_objects(self):
        cat = _chain_cat()
        h0 = representable_functor(cat, "0")
        # C(0, 0) = {id_0}
        assert h0.apply_obj("0") == frozenset(["id_0"])
        # C(0, 1) = {f}
        assert h0.apply_obj("1") == frozenset(["f"])
        # C(0, 2) = {h}
        assert h0.apply_obj("2") == frozenset(["h"])

    def test_morphism_action_is_postcomposition(self):
        cat = _chain_cat()
        h0 = representable_functor(cat, "0")
        # h0(f: 0->1)(id_0: 0->0) = f ∘ id_0 = f
        assert h0.apply_mor("f")["id_0"] == "f"

    def test_hom_at_A_contains_identity(self):
        cat = _chain_cat()
        h0 = representable_functor(cat, "0")
        assert "id_0" in h0.apply_obj("0")

    def test_unknown_object_raises(self):
        cat = _chain_cat()
        with pytest.raises(ValueError, match="not an object"):
            representable_functor(cat, "X")

    def test_hom_arrow_cat(self):
        cat = _arrow_cat()
        hA = representable_functor(cat, "A")
        # C(A, A) = {idA}, C(A, B) = {f}
        assert hA.apply_obj("A") == frozenset(["idA"])
        assert hA.apply_obj("B") == frozenset(["f"])


# ------------------------------------------------------------------ #
# yoneda_evaluate and yoneda_inverse                                    #
# ------------------------------------------------------------------ #

class TestYonedaMap:
    def test_evaluate_extracts_correct_element(self):
        cat = _chain_cat()
        F = _F_functor(cat)
        # Natural transformation mapping id_0 ↦ a (Phi^{-1}(a))
        alpha_a = yoneda_inverse("a", cat, F, "0")
        assert yoneda_evaluate(alpha_a, cat, "0") == "a"

    def test_evaluate_extracts_b(self):
        cat = _chain_cat()
        F = _F_functor(cat)
        alpha_b = yoneda_inverse("b", cat, F, "0")
        assert yoneda_evaluate(alpha_b, cat, "0") == "b"

    def test_inverse_maps_to_correct_F_values(self):
        cat = _chain_cat()
        F = _F_functor(cat)
        alpha_a = yoneda_inverse("a", cat, F, "0")
        # alpha_a at object 1: C(0,1) = {f}; alpha_a_1(f) = F(f)(a) = x
        assert alpha_a["1"]["f"] == "x"

    def test_inverse_wrong_element_raises(self):
        cat = _chain_cat()
        F = _F_functor(cat)
        with pytest.raises(ValueError):
            yoneda_inverse("INVALID", cat, F, "0")

    def test_roundtrip_eval_inverse(self):
        cat = _chain_cat()
        F = _F_functor(cat)
        for x in F.apply_obj("0"):
            alpha = yoneda_inverse(x, cat, F, "0")
            assert yoneda_evaluate(alpha, cat, "0") == x


# ------------------------------------------------------------------ #
# verify_yoneda                                                          #
# ------------------------------------------------------------------ #

class TestVerifyYoneda:
    def test_verify_yoneda_A0(self):
        """Reproduce Frontier 4: |Nat(C(0,-), F)| = |F(0)| = 2."""
        cat = _chain_cat()
        F = _F_functor(cat)
        assert verify_yoneda(cat, F, "0")

    def test_verify_yoneda_A1(self):
        cat = _chain_cat()
        F = _F_functor(cat)
        assert verify_yoneda(cat, F, "1")

    def test_verify_yoneda_A2(self):
        cat = _chain_cat()
        F = _F_functor(cat)
        assert verify_yoneda(cat, F, "2")

    def test_verify_yoneda_arrow_cat(self):
        cat = _arrow_cat()
        F = FiniteSetFunctor(
            category=cat,
            objects_map={"A": frozenset(["p","q"]), "B": frozenset(["r"])},
            morphism_map={"idA":{"p":"p","q":"q"},"idB":{"r":"r"},"f":{"p":"r","q":"r"}},
        )
        assert verify_yoneda(cat, F, "A")
        assert verify_yoneda(cat, F, "B")

    def test_verify_singleton_functor(self):
        cat = _arrow_cat()
        F = FiniteSetFunctor(
            category=cat,
            objects_map={"A": frozenset(["s"]), "B": frozenset(["t"])},
            morphism_map={"idA":{"s":"s"},"idB":{"t":"t"},"f":{"s":"t"}},
        )
        assert verify_yoneda(cat, F, "A")


# ------------------------------------------------------------------ #
# verify_yoneda_naturality_in_A                                         #
# ------------------------------------------------------------------ #

class TestYonedaNaturality:
    def test_naturality_f_0_to_1(self):
        cat = _chain_cat()
        F = _F_functor(cat)
        assert verify_yoneda_naturality_in_A(cat, F, "0", "1", "f")

    def test_naturality_g_1_to_2(self):
        cat = _chain_cat()
        F = _F_functor(cat)
        assert verify_yoneda_naturality_in_A(cat, F, "1", "2", "g")

    def test_naturality_h_0_to_2(self):
        cat = _chain_cat()
        F = _F_functor(cat)
        assert verify_yoneda_naturality_in_A(cat, F, "0", "2", "h")

    def test_wrong_direction_raises(self):
        cat = _chain_cat()
        F = _F_functor(cat)
        with pytest.raises(ValueError):
            verify_yoneda_naturality_in_A(cat, F, "1", "0", "f")  # f goes 0->1 not 1->0
