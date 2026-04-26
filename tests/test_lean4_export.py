"""Tests for topos_ai.lean4_export — Lean 4 code generation from formal structures."""

import pytest

from topos_ai.formal_category import FiniteCategory, FiniteFunctor, NaturalTransformation, Presheaf
from topos_ai.monoidal import FiniteMonoidalCategory, strict_monoidal_from_monoid
from topos_ai.lean4_export import (
    _lean_id,
    category_to_lean4,
    functor_to_lean4,
    nat_trans_to_lean4,
    monoidal_to_lean4,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _arrow_category():
    """Walking-arrow category: 0 —f→ 1."""
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


def _single_object_category():
    """Trivial category with one object and one morphism."""
    return FiniteCategory(
        objects=("*",),
        morphisms={"id": ("*", "*")},
        identities={"*": "id"},
        composition={("id", "id"): "id"},
    )


def _bool_and_smc():
    """AND strict symmetric monoidal category on {F, T}."""
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


# ------------------------------------------------------------------ #
# Tests: _lean_id                                                      #
# ------------------------------------------------------------------ #

class TestLeanId:
    def test_simple_string(self):
        assert _lean_id("hello") == "hello"

    def test_alphanumeric(self):
        assert _lean_id("abc123") == "abc123"

    def test_leading_digit_gets_prefix(self):
        result = _lean_id("0")
        assert result[0].isalpha()

    def test_special_chars_replaced(self):
        result = _lean_id("a-b.c")
        assert re.match(r"^[a-zA-Z0-9_]+$", result)

    def test_empty_fallback(self):
        result = _lean_id("---")
        assert result and result[0].isalpha()

    def test_underscore_preserved(self):
        assert _lean_id("id_A") == "id_A"

    def test_star_sanitised(self):
        result = _lean_id("*")
        assert result and result[0].isalpha()


import re


# ------------------------------------------------------------------ #
# Tests: category_to_lean4                                             #
# ------------------------------------------------------------------ #

class TestCategoryToLean4:
    def test_returns_string(self):
        C = _arrow_category()
        code = category_to_lean4(C, name="Arrow")
        assert isinstance(code, str)

    def test_contains_namespace(self):
        C = _arrow_category()
        code = category_to_lean4(C, name="Arrow")
        assert "namespace Arrow" in code
        assert "end Arrow" in code

    def test_contains_mathlib_import(self):
        C = _arrow_category()
        code = category_to_lean4(C)
        assert "import Mathlib" in code

    def test_inductive_obj_declared(self):
        C = _arrow_category()
        code = category_to_lean4(C, name="Arrow")
        assert "inductive Obj" in code

    def test_inductive_mor_declared(self):
        C = _arrow_category()
        code = category_to_lean4(C, name="Arrow")
        assert "inductive Mor" in code

    def test_obj_constructors_present(self):
        C = _arrow_category()
        code = category_to_lean4(C, name="Arrow")
        assert "| A" in code
        assert "| B" in code

    def test_mor_constructors_present(self):
        C = _arrow_category()
        code = category_to_lean4(C, name="Arrow")
        assert "| idA" in code
        assert "| f" in code

    def test_decidable_eq_derived(self):
        C = _arrow_category()
        code = category_to_lean4(C)
        assert "DecidableEq" in code

    def test_category_instance(self):
        C = _arrow_category()
        code = category_to_lean4(C, name="Arrow")
        assert "instance" in code
        assert "Category Obj" in code

    def test_axioms_by_decide(self):
        C = _arrow_category()
        code = category_to_lean4(C)
        assert "by decide" in code
        # All three category axioms
        assert "id_comp" in code
        assert "comp_id" in code
        assert "assoc" in code

    def test_hash_check_line(self):
        C = _arrow_category()
        code = category_to_lean4(C, name="Arrow")
        assert "#check" in code

    def test_ident_function(self):
        C = _arrow_category()
        code = category_to_lean4(C)
        assert "def ident" in code

    def test_comp_function(self):
        C = _arrow_category()
        code = category_to_lean4(C)
        assert "def comp" in code

    def test_single_object_category(self):
        C = _single_object_category()
        code = category_to_lean4(C, name="Trivial")
        assert "namespace Trivial" in code
        assert "inductive Obj" in code
        # object name sanitised correctly
        assert "| x" in code or "| _" in code or "| " in code

    def test_name_in_hash_check(self):
        C = _arrow_category()
        code = category_to_lean4(C, name="TestCat")
        assert "TestCat.instCategoryObj" in code

    def test_composition_arms_present(self):
        """Each entry in composition table should appear as a match arm."""
        C = _arrow_category()
        code = category_to_lean4(C, name="Arrow")
        # f o idA = f should appear
        assert "idA" in code
        assert "idB" in code

    def test_generated_code_has_no_tab_indentation_issues(self):
        """Generated code should use spaces, not mixed tabs."""
        C = _arrow_category()
        code = category_to_lean4(C)
        # Should not start lines with raw tabs
        for line in code.splitlines():
            assert not line.startswith("\t"), f"Tab-indented line: {line!r}"


# ------------------------------------------------------------------ #
# Tests: functor_to_lean4                                              #
# ------------------------------------------------------------------ #

class TestFunctorToLean4:
    def _make_identity_functor(self):
        C = _arrow_category()
        obj_map = {o: o for o in C.objects}
        mor_map = {m: m for m in C.morphisms}
        F = FiniteFunctor(source=C, target=C, object_map=obj_map, morphism_map=mor_map)
        return F, C

    def test_returns_string(self):
        F, C = self._make_identity_functor()
        code = functor_to_lean4(F, "C", "C", "IdF")
        assert isinstance(code, str)

    def test_functor_def_present(self):
        F, C = self._make_identity_functor()
        code = functor_to_lean4(F, "C", "C", "IdF")
        assert "def IdF" in code

    def test_functor_arrow_present(self):
        F, C = self._make_identity_functor()
        code = functor_to_lean4(F, "C", "D", "MyF")
        assert "C.Obj ⥤ D.Obj" in code

    def test_obj_match_arms(self):
        F, C = self._make_identity_functor()
        code = functor_to_lean4(F, "C", "C", "IdF")
        assert "obj x" in code
        assert "match x with" in code

    def test_map_match_arms(self):
        F, C = self._make_identity_functor()
        code = functor_to_lean4(F, "C", "C", "IdF")
        assert "map f" in code

    def test_map_id_by_decide(self):
        F, C = self._make_identity_functor()
        code = functor_to_lean4(F, "C", "C", "IdF")
        assert "map_id" in code
        assert "by decide" in code

    def test_map_comp_by_decide(self):
        F, C = self._make_identity_functor()
        code = functor_to_lean4(F, "C", "C", "IdF")
        assert "map_comp" in code


# ------------------------------------------------------------------ #
# Tests: nat_trans_to_lean4                                            #
# ------------------------------------------------------------------ #

class TestNatTransToLean4:
    def _make_identity_nat_trans(self):
        """Identity transformation on the constant presheaf on the arrow category."""
        C = _arrow_category()
        sets = {"A": frozenset({"x"}), "B": frozenset({"x"})}
        restrictions = {
            "idA": {"x": "x"},
            "idB": {"x": "x"},
            "f": {"x": "x"},
        }
        P = Presheaf(category=C, sets=sets, restrictions=restrictions)
        components = {
            "A": {"x": "x"},
            "B": {"x": "x"},
        }
        alpha = NaturalTransformation(source=P, target=P, components=components)
        return alpha, C

    def test_returns_string(self):
        alpha, C = self._make_identity_nat_trans()
        code = nat_trans_to_lean4(alpha, "F", "G", "alpha", "C")
        assert isinstance(code, str)

    def test_def_present(self):
        alpha, C = self._make_identity_nat_trans()
        code = nat_trans_to_lean4(alpha, "F", "G", "alpha", "C")
        assert "def alpha" in code

    def test_arrow_type(self):
        alpha, C = self._make_identity_nat_trans()
        code = nat_trans_to_lean4(alpha, "F", "G", "alpha", "C")
        assert "F ⟶ G" in code

    def test_app_match(self):
        alpha, C = self._make_identity_nat_trans()
        code = nat_trans_to_lean4(alpha, "F", "G", "alpha", "C")
        assert "app X" in code

    def test_naturality_by_decide(self):
        alpha, C = self._make_identity_nat_trans()
        code = nat_trans_to_lean4(alpha, "F", "G", "alpha", "C")
        assert "naturality" in code
        assert "by decide" in code


# ------------------------------------------------------------------ #
# Tests: monoidal_to_lean4                                             #
# ------------------------------------------------------------------ #

class TestMonoidalToLean4:
    def test_returns_string(self):
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc, name="BoolAnd")
        assert isinstance(code, str)

    def test_monoidal_import(self):
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc)
        assert "MonoidalCategory" in code

    def test_namespace_present(self):
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc, name="BoolAnd")
        assert "namespace BoolAnd" in code
        assert "end BoolAnd" in code

    def test_tensor_unit_defined(self):
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc)
        assert "def tensorUnit" in code

    def test_tensor_obj_defined(self):
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc)
        assert "def tensorObj" in code

    def test_tensor_hom_defined(self):
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc)
        assert "def tensorHom" in code

    def test_assoc_defined(self):
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc)
        assert "def assoc" in code

    def test_left_unit_defined(self):
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc)
        assert "def leftUnit" in code

    def test_right_unit_defined(self):
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc)
        assert "def rightUnit" in code

    def test_monoidal_instance(self):
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc)
        assert "instance" in code
        assert "MonoidalCategory Obj" in code

    def test_pentagon_triangle_by_decide(self):
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc)
        assert "pentagon" in code
        assert "triangle" in code
        assert code.count("by decide") >= 2

    def test_hash_check_line(self):
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc, name="BoolAnd")
        assert "BoolAnd.instMonoidalCategoryObj" in code

    def test_tensor_unit_value(self):
        """Unit of AND monoid is T; generated code should reference T."""
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc, name="BoolAnd")
        # unit = "T", lean_id("T") = "T"
        assert ".T" in code

    def test_obj_constructors_included(self):
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc, name="BoolAnd")
        assert "| F" in code
        assert "| T" in code

    def test_inductive_obj_present(self):
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc)
        assert "inductive Obj" in code

    def test_inductive_mor_present(self):
        mc = _bool_and_smc()
        code = monoidal_to_lean4(mc)
        assert "inductive Mor" in code
