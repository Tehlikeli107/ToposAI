"""
test_coverage_boost5.py — Cover remaining gaps in monoidal, infinity_categories,
formal_category via direct construction and targeted mock patches.

Covered lines:
  infinity_categories.py : 59, 125, 245, 433
  monoidal.py            : 152, 200, 236, 332, 358, 389
  formal_category.py     : 1953, 2289, 2335, 2378
"""
from __future__ import annotations

import pytest
from unittest.mock import patch


# ===========================================================================
# Helpers
# ===========================================================================

def _make_single_obj_cat():
    from topos_ai.formal_category import FiniteCategory
    return FiniteCategory(
        objects=("A",),
        morphisms={"id_A": ("A", "A")},
        identities={"A": "id_A"},
        composition={("id_A", "id_A"): "id_A"},
    )


def _make_walking_arrow_cat():
    from topos_ai.formal_category import FiniteCategory
    return FiniteCategory(
        objects=("0", "1"),
        morphisms={"id0": ("0", "0"), "id1": ("1", "1"), "f": ("0", "1")},
        identities={"0": "id0", "1": "id1"},
        composition={
            ("id0", "id0"): "id0", ("id1", "id1"): "id1",
            ("f",   "id0"): "f",  ("id1", "f"):   "f",
        },
    )


def _walking_arrow_monoidal():
    """Re-create from test_misc_coverage to keep this module self-contained."""
    from topos_ai.formal_category import FiniteCategory
    from topos_ai.monoidal import FiniteMonoidalCategory

    C = _make_walking_arrow_cat()

    def to(A, B):
        return "1" if A == "1" or B == "1" else "0"

    tensor_objects  = {(A, B): to(A, B) for A in ("0", "1") for B in ("0", "1")}
    tensor_morphisms = {
        ("id0","id0"): "id0", ("id0","id1"): "id1", ("id0","f"): "f",
        ("id1","id0"): "id1", ("id1","id1"): "id1", ("id1","f"): "id1",
        ("f",  "id0"): "f",  ("f",  "id1"): "id1", ("f",  "f"):  "f",
    }

    def alpha_mor(A, B, Cv):
        ABC = to(to(A, B), Cv)
        return "id0" if ABC == "0" else "id1"

    associator = {(A,B,Cv): alpha_mor(A,B,Cv) for A in ("0","1") for B in ("0","1") for Cv in ("0","1")}
    return FiniteMonoidalCategory(
        category=C, tensor_objects=tensor_objects, tensor_morphisms=tensor_morphisms,
        unit="0", associator=associator,
        left_unitor={"0": "id0", "1": "id1"},
        right_unitor={"0": "id0", "1": "id1"},
    )


def _z2_sym_monoidal():
    """Z/2Z-style symmetric monoidal (0⊕0=0, 0⊕1=1, 1⊕0=1, 1⊕1=0)."""
    from topos_ai.monoidal import strict_monoidal_from_monoid
    return strict_monoidal_from_monoid(
        objects=["0", "1"],
        tensor_table={
            ("0","0"): "0", ("0","1"): "1",
            ("1","0"): "1", ("1","1"): "0",
        },
        unit="0",
    )


# ===========================================================================
# infinity_categories.py
# ===========================================================================

class TestInfinityCategories:
    def test_validate_no_0simplex_raises(self):
        """Line 59: FiniteSimplicialSet({}, {}) raises 'needs a 0-simplex level'."""
        from topos_ai.infinity_categories import FiniteSimplicialSet
        with pytest.raises(ValueError, match="0-simplex level"):
            FiniteSimplicialSet({}, {})

    def test_validate_degeneracy_identities_empty_returns_true(self):
        """Line 125: validate_degeneracy_identities() returns True when degeneracies={}."""
        from topos_ai.infinity_categories import FiniteSimplicialSet
        # Build a valid 1-simplex without degeneracies
        ss = FiniteSimplicialSet(
            simplices={0: ("v0",), 1: ("e0",)},
            faces={(1, "e0", 0): "v0", (1, "e0", 1): "v0"},
        )
        assert ss.degeneracies == {}
        result = ss.validate_degeneracy_identities()
        assert result is True  # line 125

    def test_has_unique_inner_horn_fillers_true(self):
        """Line 245: returns True for a simplex with max_dimension < 2 (no inner horns)."""
        from topos_ai.infinity_categories import FiniteSimplicialSet
        ss = FiniteSimplicialSet(
            simplices={0: ("v0",), 1: ("e0",)},
            faces={(1, "e0", 0): "v0", (1, "e0", 1): "v0"},
        )
        # max_dimension=1 → no inner horns → trivially True
        assert ss.has_unique_inner_horn_fillers() is True  # line 245

    def test_infinity_category_layer_no_torch_raises(self):
        """Line 433: InfinityCategoryLayer raises ImportError when torch is None."""
        import topos_ai.infinity_categories as ic_mod
        with patch.object(ic_mod, "torch", None):
            with pytest.raises(ImportError, match="requires PyTorch"):
                ic_mod.InfinityCategoryLayer(4, 4, 4)


# ===========================================================================
# monoidal.py
# ===========================================================================

class TestMonoidalViolations:
    def test_bifunctoriality_interchange_violation(self):
        """Line 152: corrupt tensor_morphisms so (h∘f)⊗(k∘g) ≠ (h⊗k)∘(f⊗g)."""
        mc = _walking_arrow_monoidal()
        # Corrupt f⊗f from "f" to "id0".
        # For (h,f)=(f,id0), (k,g)=(id1,f):
        #   LHS = C.compose(tensor_mor(f,id1)="id1", tensor_mor(id0,f)="f") = "f"
        #   RHS = tensor_mor(C.compose(f,id0)="f", C.compose(id1,f)="f")
        #       = tensor_mor("f","f") = "id0"  ← corrupted
        #   "f" ≠ "id0" → ValueError
        mc.tensor_morphisms[("f", "f")] = "id0"
        with pytest.raises(ValueError, match="Bifunctoriality interchange"):
            mc._check_bifunctoriality()

    def test_associator_naturality_violation(self):
        """Line 200: patch compose to make LHS ≠ RHS in associator naturality check."""
        mc = _walking_arrow_monoidal()
        original = mc.category.compose
        call_count = [0]

        def bad_compose(after, before):
            call_count[0] += 1
            if call_count[0] == 1:
                return "id1"  # first call: lie (should be "id0")
            return original(after, before)

        with patch.object(mc.category, "compose", side_effect=bad_compose):
            with pytest.raises(ValueError, match="Associator naturality"):
                mc._check_associator_naturality()

    def test_pentagon_violation(self):
        """Line 236: patch compose so LHS ≠ RHS in pentagon check."""
        mc = _walking_arrow_monoidal()
        original = mc.category.compose
        call_count = [0]

        def bad_compose(after, before):
            call_count[0] += 1
            # Call 1 is the LHS compose α ∘ α.  Returning "id1" instead of "id0"
            # makes lhs="id1" while rhs="id0" → raises "Pentagon coherence fails".
            if call_count[0] == 1:
                real = original(after, before)
                return "id1" if real == "id0" else "id0"
            return original(after, before)

        with patch.object(mc.category, "compose", side_effect=bad_compose):
            with pytest.raises(ValueError, match="Pentagon coherence"):
                mc._check_pentagon()

    def test_braiding_naturality_violation(self):
        """Line 332: patch compose so LHS ≠ RHS in braiding naturality check."""
        mc = _z2_sym_monoidal()
        original = mc.category.compose
        call_count = [0]

        def bad_compose(after, before):
            call_count[0] += 1
            if call_count[0] == 1:
                return "WRONG_BRAID"
            return original(after, before)

        with patch.object(mc.category, "compose", side_effect=bad_compose):
            with pytest.raises(ValueError, match="Braiding naturality"):
                mc._check_braiding_naturality()

    def test_hexagon_violation(self):
        """Line 358: patch compose (call 2 = outer LHS compose) to fail hexagon."""
        mc = _z2_sym_monoidal()
        original = mc.category.compose
        call_count = [0]

        def bad_compose(after, before):
            call_count[0] += 1
            if call_count[0] == 2:
                return "WRONG_HEX"
            return original(after, before)

        with patch.object(mc.category, "compose", side_effect=bad_compose):
            with pytest.raises(ValueError, match="Hexagon coherence"):
                mc._check_hexagon()

    def test_strict_monoidal_default_name_tensor(self):
        """Line 389: default name_tensor lambda is defined and called when None is passed."""
        from topos_ai.monoidal import strict_monoidal_from_monoid
        # Passing no name_tensor hits the `def name_tensor(f, g): return f"{f}_x_{g}"` branch
        mc = strict_monoidal_from_monoid(
            objects=["x"],
            tensor_table={("x", "x"): "x"},
            unit="x",
            name_tensor=None,  # triggers line 388-389
        )
        assert mc is not None
        # The default name_tensor is called during tensor_morphisms construction
        # id_x ⊗ id_x → "id_x_x_id_x"
        assert mc.tensor_morphisms.get(("id_x", "id_x")) == "id_x"


# ===========================================================================
# formal_category.py
# ===========================================================================

class TestPresheafToposEdgeCases:
    def _single_obj_topos(self):
        from topos_ai.formal_category import PresheafTopos
        cat = _make_single_obj_cat()
        return PresheafTopos(cat), cat

    def _walking_arrow_topos(self):
        from topos_ai.formal_category import PresheafTopos
        cat = _make_walking_arrow_cat()
        return PresheafTopos(cat), cat

    def test_validate_effective_epimorphism_non_epi_returns_false(self):
        """Line 1953: non-epimorphism transformation → validate_effective_epimorphism returns False."""
        from topos_ai.formal_category import Presheaf, NaturalTransformation
        topos, cat = self._single_obj_topos()

        P = Presheaf(cat, {"A": {"p1", "p2"}}, {"id_A": {"p1": "p1", "p2": "p2"}})
        Q = Presheaf(cat, {"A": {"q1", "q2"}}, {"id_A": {"q1": "q1", "q2": "q2"}})
        # Non-surjective: both p1, p2 → q1 (q2 never reached)
        alpha = NaturalTransformation(
            source=P, target=Q,
            components={"A": {"p1": "q1", "p2": "q1"}},
        )
        result = topos.validate_effective_epimorphism(alpha)
        assert result is False  # line 1953

    def test_require_topology_wrong_category_raises(self):
        """Line 2289: j_operator_on_sieve with topology from different category raises."""
        from topos_ai.formal_category import PresheafTopos, GrothendieckTopology
        from topos_ai.formal_category import FiniteCategory

        # topos is over cat_A
        topos, cat_A = self._single_obj_topos()

        # topology is over a *different* category object cat_B
        cat_B = _make_single_obj_cat()  # new object, different identity
        max_sieve_B = frozenset({"id_A"})
        gt_B = GrothendieckTopology(cat_B, {"A": [frozenset({"id_A"})]})

        # j_operator_on_sieve calls _require_topology which checks `topology.category is self.category`
        with pytest.raises(ValueError, match="Topology must be defined on this topos base category"):
            topos.j_operator_on_sieve(gt_B, "A", frozenset({"id_A"}))

    def test_sieve_implication_non_sieve_raises(self):
        """Line 2378: sieve_implication raises when antecedent is not a sieve."""
        topos, cat = self._walking_arrow_topos()
        # {"id1"} is NOT a sieve on "1" because compose(id1, f) = f ∉ {"id1"}
        with pytest.raises(ValueError, match="Sieve implication requires sieves"):
            topos.sieve_implication("1", frozenset({"id1"}), frozenset({"id1", "f"}))

    def test_validate_lawvere_tierney_lt1_fails(self):
        """Line 2335: validate_lawvere_tierney_axioms returns False when j(true) ≠ true."""
        from topos_ai.formal_category import PresheafTopos, GrothendieckTopology
        topos, cat = self._single_obj_topos()
        gt = GrothendieckTopology(cat, {"A": [frozenset({"id_A"})]})

        # Mock j_operator_on_sieve to always return empty sieve → j(true) = {} ≠ {id_A}
        with patch.object(topos, "j_operator_on_sieve", return_value=frozenset()):
            result = topos.validate_lawvere_tierney_axioms(gt)
        assert result is False  # line 2335
