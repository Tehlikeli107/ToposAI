"""
Coverage tests for:
  - topos_ai.storage.cql_database.CategoricalDatabase
  - topos_ai.topology.sheaf_computer.ToposSheafComputer
"""

import pytest

from topos_ai.storage.cql_database import CategoricalDatabase
from topos_ai.topology.sheaf_computer import ToposSheafComputer


# ------------------------------------------------------------------ #
# CategoricalDatabase                                                  #
# ------------------------------------------------------------------ #

class TestCategoricalDatabase:
    def test_constructs_in_memory(self):
        db = CategoricalDatabase(":memory:")
        assert db is not None
        db.close()

    def test_add_object_returns_id(self):
        db = CategoricalDatabase(":memory:")
        oid = db.add_object("A")
        assert isinstance(oid, int)
        assert oid > 0
        db.close()

    def test_add_object_idempotent(self):
        db = CategoricalDatabase(":memory:")
        id1 = db.add_object("X")
        id2 = db.add_object("X")
        assert id1 == id2
        db.close()

    def test_get_object_id_returns_none_for_missing(self):
        db = CategoricalDatabase(":memory:")
        assert db.get_object_id("missing") is None
        db.close()

    def test_add_morphism_returns_true(self):
        db = CategoricalDatabase(":memory:")
        result = db.add_morphism("f", "A", "B")
        assert result is True
        db.close()

    def test_add_morphism_duplicate_returns_false(self):
        db = CategoricalDatabase(":memory:")
        db.add_morphism("f", "A", "B")
        result = db.add_morphism("f", "A", "B")
        assert result is False
        db.close()

    def test_count_morphisms(self):
        db = CategoricalDatabase(":memory:")
        assert db.count_morphisms() == 0
        db.add_morphism("f", "A", "B")
        db.add_morphism("g", "B", "C")
        assert db.count_morphisms() == 2
        db.close()

    def test_transitive_closure_creates_composed_morphism(self):
        db = CategoricalDatabase(":memory:")
        db.add_morphism("f", "A", "B")
        db.add_morphism("g", "B", "C")
        before = db.count_morphisms()
        db.compute_transitive_closure_sql_join(max_depth=2)
        after = db.count_morphisms()
        # g∘f: A→C should be added
        assert after > before
        db.close()

    def test_transitive_closure_no_new_morphisms_on_chain(self):
        """A→B, B→C, A→C already present: closure adds nothing."""
        db = CategoricalDatabase(":memory:")
        db.add_morphism("f", "A", "B")
        db.add_morphism("g", "B", "C")
        db.add_morphism("h", "A", "C")
        before = db.count_morphisms()
        db.compute_transitive_closure_sql_join(max_depth=3)
        after = db.count_morphisms()
        assert after == before
        db.close()

    def test_transitive_closure_verbose(self, capsys):
        db = CategoricalDatabase(":memory:")
        db.add_morphism("f", "A", "B")
        db.add_morphism("g", "B", "C")
        db.compute_transitive_closure_sql_join(max_depth=2, verbose=True)
        db.close()
        # Should not raise; verbose flag exercises the print paths

    def test_object_auto_created_by_morphism(self):
        db = CategoricalDatabase(":memory:")
        db.add_morphism("f", "X", "Y")
        assert db.get_object_id("X") is not None
        assert db.get_object_id("Y") is not None
        db.close()

    def test_multiple_morphisms_same_endpoints(self):
        db = CategoricalDatabase(":memory:")
        db.add_morphism("f", "A", "B")
        db.add_morphism("g", "A", "B")  # distinct label, same src/dst
        assert db.count_morphisms() == 2
        db.close()

    def test_single_object_loop_morphism(self):
        db = CategoricalDatabase(":memory:")
        db.add_morphism("id", "A", "A")
        assert db.count_morphisms() == 1
        db.close()


# ------------------------------------------------------------------ #
# ToposSheafComputer                                                   #
# ------------------------------------------------------------------ #

class TestToposSheafComputer:
    def test_constructs(self):
        sc = ToposSheafComputer(n_nodes=10, patch_size=5, overlap=1)
        assert sc is not None

    def test_patches_non_empty(self):
        sc = ToposSheafComputer(n_nodes=20, patch_size=10, overlap=2)
        assert len(sc.patches) > 0

    def test_single_patch_for_small_n(self):
        sc = ToposSheafComputer(n_nodes=5, patch_size=10, overlap=2)
        assert len(sc.patches) >= 1

    def test_each_patch_is_finite_category(self):
        from topos_ai.formal_category import FiniteCategory
        sc = ToposSheafComputer(n_nodes=15, patch_size=8, overlap=2)
        for _s, _e, patch in sc.patches:
            assert isinstance(patch, FiniteCategory)

    def test_global_morphism_query_same_node(self):
        sc = ToposSheafComputer(n_nodes=10, patch_size=5, overlap=1)
        result = sc.global_morphism_query_via_gluing(0, 0)
        # Identity path
        assert result is not None

    def test_global_morphism_query_adjacent_nodes(self):
        sc = ToposSheafComputer(n_nodes=10, patch_size=5, overlap=1)
        result = sc.global_morphism_query_via_gluing(0, 1)
        assert result is not None

    def test_global_morphism_query_cross_patch(self):
        """Query across patch boundary should return a glued path."""
        sc = ToposSheafComputer(n_nodes=20, patch_size=8, overlap=2)
        result = sc.global_morphism_query_via_gluing(0, 15)
        assert result is not None

    def test_global_morphism_query_full_range(self):
        sc = ToposSheafComputer(n_nodes=15, patch_size=8, overlap=2)
        result = sc.global_morphism_query_via_gluing(0, 14)
        assert result is not None

    def test_patch_coverage_covers_all_nodes(self):
        """Every node index should appear in at least one patch."""
        n = 25
        sc = ToposSheafComputer(n_nodes=n, patch_size=10, overlap=3)
        covered = set()
        for start, end, patch in sc.patches:
            for i in range(start, end):
                covered.add(i)
        assert all(i in covered for i in range(n))

    def test_overlap_creates_multiple_patches(self):
        sc = ToposSheafComputer(n_nodes=30, patch_size=10, overlap=3)
        assert len(sc.patches) >= 3

    def test_local_patch_has_transitive_morphisms(self):
        """Each patch should include composed morphisms (transitive closure)."""
        sc = ToposSheafComputer(n_nodes=10, patch_size=6, overlap=1)
        for _s, _e, patch in sc.patches:
            # At least as many morphisms as objects (identity + steps + composed)
            assert len(patch.morphisms) >= len(patch.objects)

    def test_query_returns_string_path(self):
        sc = ToposSheafComputer(n_nodes=10, patch_size=5, overlap=1)
        result = sc.global_morphism_query_via_gluing(0, 3)
        if result is not None:
            assert isinstance(result, str)
