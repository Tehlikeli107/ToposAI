"""
test_coverage_boost4.py — Target the remaining single-line misses and module gaps.

Covers:
  - lazy/free_category.py line 68  (BFS skip when edge is in exceptions)
  - sheaf_nn.py line 372           (skip edge when src/dst not in vertex_objects)
  - sites.py line 251              (GrothendieckTopology transitivity violation)
  - topology/sheaf_computer.py line 110  (return None when path exhausted)
"""
from __future__ import annotations

import pytest
from unittest.mock import patch


# ===========================================================================
# Helpers: minimal FiniteCategory
# ===========================================================================

def _make_cat_AB():
    """A → B category with id_A, id_B, f."""
    from topos_ai.formal_category import FiniteCategory
    return FiniteCategory(
        objects=("A", "B"),
        morphisms={
            "id_A": ("A", "A"),
            "id_B": ("B", "B"),
            "f": ("A", "B"),
        },
        identities={"A": "id_A", "B": "id_B"},
        composition={
            ("id_A", "id_A"): "id_A",
            ("id_B", "id_B"): "id_B",
            ("f",   "id_A"): "f",
            ("id_B", "f"  ): "f",
        },
    )


def _make_cat_A():
    """Single-object category {A} with only id_A."""
    from topos_ai.formal_category import FiniteCategory
    return FiniteCategory(
        objects=("A",),
        morphisms={"id_A": ("A", "A")},
        identities={"A": "id_A"},
        composition={("id_A", "id_A"): "id_A"},
    )


# ===========================================================================
# lazy/free_category.py  line 68
# ===========================================================================

class TestFreeCategoryExceptions:
    def test_start_obj_exception_blocks_path(self):
        """Line 68: (start_obj, next_node) in exceptions → continue, path returns None."""
        from topos_ai.lazy.free_category import FreeCategoryGenerator

        fc = FreeCategoryGenerator()
        fc.add_morphism("f", "A", "B")
        fc.add_morphism("g", "B", "C")

        # Block the only edge out of A: BFS hits line 68 and can't find C.
        result = fc.find_morphism_path_lazy("A", "C", exceptions=[("A", "B")])
        assert result is None

    def test_current_node_exception_blocks_path(self):
        """Line 68: (current_node, next_node) in exceptions → continue."""
        from topos_ai.lazy.free_category import FreeCategoryGenerator

        fc = FreeCategoryGenerator()
        fc.add_morphism("f", "A", "B")
        fc.add_morphism("g", "B", "C")
        fc.add_morphism("h", "C", "D")

        # Block the B→C edge mid-path; A→B is fine, but B→C is blocked.
        result = fc.find_morphism_path_lazy("A", "D", exceptions=[("B", "C")])
        assert result is None

    def test_exception_does_not_block_other_paths(self):
        """Exception blocks one route but the BFS finds an alternative."""
        from topos_ai.lazy.free_category import FreeCategoryGenerator

        fc = FreeCategoryGenerator()
        fc.add_morphism("f", "A", "B")
        fc.add_morphism("g", "A", "C")  # direct A→C as alternative
        fc.add_morphism("h", "B", "C")

        # Block A→B; but A→C exists directly.
        result = fc.find_morphism_path_lazy("A", "C", exceptions=[("A", "B")])
        assert result == "g"

    def test_no_exceptions_normal_bfs(self):
        """Without exceptions BFS finds the normal composed path."""
        from topos_ai.lazy.free_category import FreeCategoryGenerator

        fc = FreeCategoryGenerator()
        fc.add_morphism("f", "A", "B")
        fc.add_morphism("g", "B", "C")

        result = fc.find_morphism_path_lazy("A", "C")
        assert result == "g o f"

    def test_identity_shortcut(self):
        """start == target returns identity without BFS."""
        from topos_ai.lazy.free_category import FreeCategoryGenerator

        fc = FreeCategoryGenerator()
        fc.add_morphism("f", "A", "B")
        assert fc.find_morphism_path_lazy("A", "A") == "id_A"

    def test_direct_start_target_exception(self):
        """If (start, target) is in exceptions, return None immediately (line 49-50)."""
        from topos_ai.lazy.free_category import FreeCategoryGenerator

        fc = FreeCategoryGenerator()
        fc.add_morphism("f", "A", "B")
        result = fc.find_morphism_path_lazy("A", "B", exceptions=[("A", "B")])
        assert result is None


# ===========================================================================
# sheaf_nn.py  line 372
# ===========================================================================

class TestSheafLaplacianFromPresheaf:
    def _build_presheaf(self):
        """A→B presheaf: F(A)={a1,a2}, F(B)={b1}, F(f)={b1→a1}."""
        from topos_ai.formal_category import FiniteCategory, Presheaf

        cat = FiniteCategory(
            objects=("A", "B"),
            morphisms={
                "id_A": ("A", "A"),
                "id_B": ("B", "B"),
                "f":    ("A", "B"),
            },
            identities={"A": "id_A", "B": "id_B"},
            composition={
                ("id_A", "id_A"): "id_A",
                ("id_B", "id_B"): "id_B",
                ("f",    "id_A"): "f",
                ("id_B", "f"  ): "f",
            },
        )
        sets_map   = {"A": {"a1", "a2"}, "B": {"b1"}}
        restrictions = {
            "id_A": {"a1": "a1", "a2": "a2"},
            "id_B": {"b1": "b1"},
            "f":    {"b1": "a1"},
        }
        presheaf = Presheaf(category=cat, sets=sets_map, restrictions=restrictions)
        return cat, presheaf

    def test_skips_edge_with_missing_vertex(self):
        """Line 372: morphism g: B→? where ? ∉ vertex_objects is skipped silently."""
        from topos_ai.sheaf_nn import sheaf_laplacian_from_presheaf

        cat, presheaf = self._build_presheaf()

        # vertex_objects = [A, B].  f: A→B is valid.
        # We introduce a mock morphism whose source or target is outside vertex_objects.
        # We can do this by patching the category's source/target to lie outside.
        # Easier: add a third object C with an extra morphism "g": B→C,
        # but keep vertex_objects as [A, B].  We do that by patching cat.source/target.

        class _FakeCategory:
            """Delegates everything to cat but reports that 'g' goes B→C."""
            def __init__(self, real):
                self._r = real

            def source(self, m):
                return "B" if m == "ghost" else self._r.source(m)

            def target(self, m):
                return "C" if m == "ghost" else self._r.target(m)

            @property
            def objects(self):
                return self._r.objects

        # Wrap presheaf so its .category is our fake one
        fake_cat = _FakeCategory(cat)
        import types
        fake_presheaf = types.SimpleNamespace(
            category=fake_cat,
            sets={"A": frozenset({"a1", "a2"}), "B": frozenset({"b1"})},
            restrictions={
                "id_A": {"a1": "a1", "a2": "a2"},
                "id_B": {"b1": "b1"},
                "f":    {"b1": "a1"},
                "ghost": {"b1": "b1"},   # "ghost" goes B→C in fake_cat
            },
        )

        # edge_morphisms includes 'ghost' which goes to C ∉ vertex_objects → line 372
        lap = sheaf_laplacian_from_presheaf(
            presheaf=fake_presheaf,
            vertex_objects=["A", "B"],
            edge_morphisms=["f", "ghost"],
            stalk_dim=2,
        )
        assert lap is not None  # returned a valid SheafLaplacian (ghost was skipped)

    def test_normal_presheaf_works(self):
        """Sanity: presheaf with valid vertex_objects returns a SheafLaplacian."""
        from topos_ai.sheaf_nn import sheaf_laplacian_from_presheaf

        _, presheaf = self._build_presheaf()
        lap = sheaf_laplacian_from_presheaf(
            presheaf=presheaf,
            vertex_objects=["A", "B"],
            edge_morphisms=["f"],
            stalk_dim=2,
        )
        assert lap is not None


# ===========================================================================
# sites.py  line 251
# ===========================================================================

class TestGrothendieckTopologyTransitivity:
    def test_transitivity_violation_raises(self):
        """Line 251: transitivity check raises when sieve should be covering but isn't."""
        from topos_ai.sites import GrothendieckTopology, Sieve, maximal_sieve

        cat = _make_cat_A()

        max_s   = maximal_sieve(cat, "A")           # {id_A}
        empty_s = Sieve(on="A", morphisms=set(), category=cat, validate=False)

        # Build the topology with validate=False so we can insert a deliberate
        # violation: J(A) = [max_s] only (empty_s is not covering).
        gt = GrothendieckTopology(cat, {"A": [max_s]}, validate=False)

        # Patch 'covers' so that:
        #   1st call  covers(A, empty_s) → False  (R is not covering → don't skip)
        #   2nd call  covers(A, empty_s) → True   (pullback of R along id_A IS covering)
        # This forces all_pull_cover=True while R is not covering → ValueError.
        original_covers = gt.covers
        call_counts: dict = {}

        def patched_covers(d, sieve):
            if sieve == empty_s:
                key = (d, "empty")
                call_counts[key] = call_counts.get(key, 0) + 1
                # First call (non-covering check) → False; subsequent (pullback) → True
                return call_counts[key] > 1
            return original_covers(d, sieve)

        with patch.object(gt, "covers", side_effect=patched_covers):
            with pytest.raises(ValueError, match="transitivity"):
                gt._check_transitivity()

    def test_valid_topology_passes_transitivity(self):
        """A proper trivial topology does not raise during transitivity check."""
        from topos_ai.sites import GrothendieckTopology, maximal_sieve

        cat = _make_cat_A()
        max_s = maximal_sieve(cat, "A")
        # Valid: only the maximal sieve covers everything; there are no non-covering
        # sieves whose pullbacks are all covering.
        gt = GrothendieckTopology(cat, {"A": [max_s]}, validate=False)
        # Direct call should not raise
        gt._check_transitivity()  # no error

    def test_full_validate_passes(self):
        """validate=True on a correct Grothendieck topology succeeds."""
        from topos_ai.sites import GrothendieckTopology, maximal_sieve

        cat = _make_cat_A()
        max_s = maximal_sieve(cat, "A")
        gt = GrothendieckTopology(cat, {"A": [max_s]}, validate=True)
        assert gt is not None


# ===========================================================================
# topology/sheaf_computer.py  line 110
# ===========================================================================

class TestToposSheafComputer:
    def test_returns_none_when_dst_not_in_any_patch(self):
        """Line 110: return None when destination node doesn't exist in any patch."""
        from topos_ai.topology.sheaf_computer import ToposSheafComputer

        sc = ToposSheafComputer(n_nodes=5, patch_size=5, overlap=2)
        # Node_100 is not in any patch; the engine exhausts all patches → line 110
        result = sc.global_morphism_query_via_gluing(0, 100)
        assert result is None

    def test_returns_path_for_valid_query(self):
        """Sanity: a reachable destination returns a non-None path string."""
        from topos_ai.topology.sheaf_computer import ToposSheafComputer

        sc = ToposSheafComputer(n_nodes=10, patch_size=10, overlap=3)
        result = sc.global_morphism_query_via_gluing(0, 5)
        assert result is not None
        assert isinstance(result, str)

    def test_returns_none_for_negative_dst(self):
        """Destination index far outside the sharded patches returns None."""
        from topos_ai.topology.sheaf_computer import ToposSheafComputer

        sc = ToposSheafComputer(n_nodes=8, patch_size=4, overlap=2)
        result = sc.global_morphism_query_via_gluing(0, 999)
        assert result is None
