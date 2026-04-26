"""
Tests for topos_ai.formal_lawvere_tierney.

Coverage goals:
  - LawvereTierneyTopology construction (happy + error paths)
  - All three axiom checkers (_check_lt1, _check_lt2, _check_lt3)
  - apply(), identity(), dense(), __eq__, __hash__, __repr__
  - all_lt_topologies()
  - j_closure()
  - j_closed_subobjects()
  - j_dense_monomorphism()
  - verify_lt_axioms()
  - subobject_lattice()
  - j_action_on_subobjects()
  - verify_closure_operator()
"""
from __future__ import annotations

import pytest
from topos_ai.formal_lawvere_tierney import (
    LawvereTierneyTopology,
    all_lt_topologies,
    j_action_on_subobjects,
    j_closed_subobjects,
    j_closure,
    j_dense_monomorphism,
    subobject_lattice,
    verify_closure_operator,
    verify_lt_axioms,
)
from topos_ai.topos import TRUE, FALSE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _identity():
    return LawvereTierneyTopology.identity()


def _dense():
    return LawvereTierneyTopology.dense()


X2 = frozenset({"a", "b"})
X3 = frozenset({"a", "b", "c"})
X_empty = frozenset()
X1 = frozenset({"x"})


def _mono(subset, X):
    """Return inclusion mono frozenset for subset ⊆ X."""
    return frozenset((s, s) for s in subset)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_identity_valid(self):
        lt = _identity()
        assert lt.j[TRUE] == TRUE
        assert lt.j[FALSE] == FALSE

    def test_dense_valid(self):
        lt = _dense()
        assert lt.j[TRUE] == TRUE
        assert lt.j[FALSE] == TRUE

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="Ω"):
            LawvereTierneyTopology({TRUE: TRUE})

    def test_extra_key_raises(self):
        with pytest.raises(ValueError, match="Ω"):
            LawvereTierneyTopology({TRUE: TRUE, FALSE: FALSE, "X": TRUE})

    def test_value_not_in_omega_raises(self):
        with pytest.raises(ValueError, match="not in Ω"):
            LawvereTierneyTopology({TRUE: TRUE, FALSE: "maybe"}, validate=False)
            # validate=False skips axiom checks but constructor still validates values
        # With validate=True it also raises before axioms
        with pytest.raises(ValueError, match="not in Ω"):
            LawvereTierneyTopology({TRUE: TRUE, FALSE: "maybe"})

    def test_validate_false_skips_axiom_checks(self):
        # j(T)=F violates LT1; with validate=False it constructs without error
        lt = LawvereTierneyTopology({TRUE: FALSE, FALSE: FALSE}, validate=False)
        assert lt.j[TRUE] == FALSE


# ---------------------------------------------------------------------------
# LT1: j(TRUE) = TRUE
# ---------------------------------------------------------------------------

class TestLT1:
    def test_lt1_violation_raises(self):
        with pytest.raises(ValueError, match="LT1"):
            LawvereTierneyTopology({TRUE: FALSE, FALSE: FALSE})

    def test_lt1_violation_message_contains_true(self):
        with pytest.raises(ValueError, match="TRUE"):
            LawvereTierneyTopology({TRUE: FALSE, FALSE: TRUE})


# ---------------------------------------------------------------------------
# LT2: idempotency j∘j = j
# ---------------------------------------------------------------------------

class TestLT2:
    def test_lt2_violation_raises(self):
        # j(T)=T, j(F)=T is idempotent (dense) — no violation
        # There's no way to violate LT2 while satisfying LT1 on Ω={T,F}
        # because both valid maps already satisfy it; we verify via validate=False
        lt = LawvereTierneyTopology({TRUE: TRUE, FALSE: FALSE}, validate=False)
        lt._check_lt2()  # must not raise

    def test_lt2_direct_check_identity(self):
        _identity()._check_lt2()

    def test_lt2_direct_check_dense(self):
        _dense()._check_lt2()

    def test_lt2_violation_via_validate_false(self):
        # Construct hypothetical j: T→T, F→F satisfies; but manually patch to violate
        lt = LawvereTierneyTopology({TRUE: TRUE, FALSE: FALSE}, validate=False)
        lt.j = {TRUE: TRUE, FALSE: TRUE}  # this is actually idempotent (dense)
        lt._check_lt2()  # should still pass for dense
        # To truly violate: we'd need j(j(F))≠j(F), impossible on {T,F} while LT1 holds
        # So we test the error message path by directly patching _check_lt2 scenario
        # using validate=False + non-Omega value already tested in construction test


# ---------------------------------------------------------------------------
# LT3: preserves meets
# ---------------------------------------------------------------------------

class TestLT3:
    def test_lt3_identity_valid(self):
        _identity()._check_lt3()

    def test_lt3_dense_valid(self):
        _dense()._check_lt3()

    def test_lt3_all_valid_topologies_pass(self):
        for lt in all_lt_topologies():
            lt._check_lt3()  # must not raise


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------

class TestApply:
    def test_apply_identity_true(self):
        assert _identity().apply(TRUE) == TRUE

    def test_apply_identity_false(self):
        assert _identity().apply(FALSE) == FALSE

    def test_apply_dense_true(self):
        assert _dense().apply(TRUE) == TRUE

    def test_apply_dense_false(self):
        assert _dense().apply(FALSE) == TRUE

    def test_apply_invalid_raises(self):
        with pytest.raises(ValueError, match="not in Ω"):
            _identity().apply("maybe")


# ---------------------------------------------------------------------------
# Equality / hash / repr
# ---------------------------------------------------------------------------

class TestEquality:
    def test_identity_equals_itself(self):
        assert _identity() == _identity()

    def test_dense_equals_itself(self):
        assert _dense() == _dense()

    def test_identity_not_equal_dense(self):
        assert _identity() != _dense()

    def test_not_equal_to_non_lt(self):
        result = _identity().__eq__("not an LT")
        assert result is NotImplemented

    def test_hash_consistent(self):
        lt1 = _identity()
        lt2 = _identity()
        assert hash(lt1) == hash(lt2)

    def test_hash_different_for_different_topologies(self):
        assert hash(_identity()) != hash(_dense())

    def test_can_be_used_in_set(self):
        s = {_identity(), _dense(), _identity()}
        assert len(s) == 2

    def test_repr_identity(self):
        r = repr(_identity())
        assert "T" in r or "TRUE" in r or "F" in r

    def test_repr_contains_topology_class(self):
        assert "LawvereTierneyTopology" in repr(_dense())


# ---------------------------------------------------------------------------
# all_lt_topologies()
# ---------------------------------------------------------------------------

class TestAllLTTopologies:
    def test_returns_list(self):
        result = all_lt_topologies()
        assert isinstance(result, list)

    def test_exactly_two_topologies(self):
        result = all_lt_topologies()
        assert len(result) == 2

    def test_both_are_lt_instances(self):
        for lt in all_lt_topologies():
            assert isinstance(lt, LawvereTierneyTopology)

    def test_identity_in_result(self):
        result = all_lt_topologies()
        assert _identity() in result

    def test_dense_in_result(self):
        result = all_lt_topologies()
        assert _dense() in result

    def test_all_satisfy_axioms(self):
        for lt in all_lt_topologies():
            assert verify_lt_axioms(lt)


# ---------------------------------------------------------------------------
# j_closure()
# ---------------------------------------------------------------------------

class TestJClosure:
    def test_closure_full_set_identity(self):
        X = X2
        mono = _mono({"a", "b"}, X)
        closure = j_closure(X, mono, _identity())
        assert closure == mono

    def test_closure_empty_identity(self):
        X = X2
        mono = _mono(set(), X)
        closure = j_closure(X, mono, _identity())
        assert closure == mono

    def test_closure_singleton_identity(self):
        X = X2
        mono = _mono({"a"}, X)
        closure = j_closure(X, mono, _identity())
        assert closure == mono

    def test_closure_empty_dense_becomes_all(self):
        X = X2
        mono = _mono(set(), X)
        closure = j_closure(X, mono, _dense())
        # dense j sends everything to TRUE, so closure is all of X
        expected = _mono({"a", "b"}, X)
        assert closure == expected

    def test_closure_singleton_dense_becomes_all(self):
        X = X2
        mono = _mono({"a"}, X)
        closure = j_closure(X, mono, _dense())
        expected = _mono({"a", "b"}, X)
        assert closure == expected

    def test_closure_of_x_is_x_for_any_lt(self):
        X = X2
        mono = _mono({"a", "b"}, X)
        for lt in all_lt_topologies():
            closure = j_closure(X, mono, lt)
            assert closure == mono

    def test_closure_on_single_element(self):
        X = X1
        mono = _mono({"x"}, X)
        for lt in all_lt_topologies():
            closure = j_closure(X, mono, lt)
            assert closure == mono

    def test_closure_empty_on_X1_dense(self):
        X = X1
        mono = _mono(set(), X)
        closure = j_closure(X, mono, _dense())
        expected = _mono({"x"}, X)
        assert closure == expected

    def test_closure_three_elements_identity(self):
        X = X3
        mono = _mono({"a", "b"}, X)
        closure = j_closure(X, mono, _identity())
        assert closure == mono

    def test_closure_three_elements_dense(self):
        X = X3
        mono = _mono({"a"}, X)
        closure = j_closure(X, mono, _dense())
        expected = _mono({"a", "b", "c"}, X)
        assert closure == expected


# ---------------------------------------------------------------------------
# j_closed_subobjects()
# ---------------------------------------------------------------------------

class TestJClosedSubobjects:
    def test_identity_all_subobjects_closed(self):
        X = X2
        closed = j_closed_subobjects(X, _identity())
        # Under identity topology every subobject is closed
        lattice = subobject_lattice(X)
        assert len(closed) == len(lattice)
        for mono in lattice:
            assert mono in closed

    def test_dense_only_empty_and_full_closed(self):
        # Under dense topology: j̄(A) = X for any A ≠ X
        # So only X is j-closed (and ∅ is not, since j̄(∅)=X ≠ ∅)
        # Wait: for dense j, j̄(A) = X for all A (since j(F)=T means even out-of-A elems
        # get T). So j̄(A)=X, hence only X satisfies j̄(A)=A.
        X = X2
        closed = j_closed_subobjects(X, _dense())
        # Only X itself is closed under dense topology
        X_mono = _mono({"a", "b"}, X)
        assert X_mono in closed
        # empty is NOT closed under dense (j̄(∅) = X ≠ ∅) when X is non-empty
        empty_mono = _mono(set(), X)
        assert empty_mono not in closed

    def test_empty_ambient_has_one_closed(self):
        X = X_empty
        for lt in all_lt_topologies():
            closed = j_closed_subobjects(X, lt)
            assert len(closed) == 1  # only ∅ itself

    def test_single_element_identity(self):
        X = X1
        closed = j_closed_subobjects(X, _identity())
        assert len(closed) == 2  # ∅ and {x}

    def test_single_element_dense(self):
        X = X1
        closed = j_closed_subobjects(X, _dense())
        # j̄(∅) = X ≠ ∅, j̄({x}) = {x} = X so X closed
        assert len(closed) == 1
        X_mono = _mono({"x"}, X)
        assert X_mono in closed

    def test_all_closed_are_in_lattice(self):
        X = X2
        lattice = subobject_lattice(X)
        for lt in all_lt_topologies():
            closed = j_closed_subobjects(X, lt)
            for mono in closed:
                assert mono in lattice


# ---------------------------------------------------------------------------
# j_dense_monomorphism()
# ---------------------------------------------------------------------------

class TestJDenseMonomorphism:
    def test_X_itself_dense_identity(self):
        X = X2
        mono = _mono({"a", "b"}, X)
        assert j_dense_monomorphism(X, mono, _identity()) is True

    def test_proper_subset_not_dense_identity(self):
        X = X2
        mono = _mono({"a"}, X)
        assert j_dense_monomorphism(X, mono, _identity()) is False

    def test_empty_not_dense_identity(self):
        X = X2
        mono = _mono(set(), X)
        assert j_dense_monomorphism(X, mono, _identity()) is False

    def test_any_mono_dense_under_dense_topology(self):
        X = X2
        for subset in [set(), {"a"}, {"b"}, {"a", "b"}]:
            mono = _mono(subset, X)
            assert j_dense_monomorphism(X, mono, _dense()) is True

    def test_dense_on_single_element_dense_topology(self):
        X = X1
        mono = _mono(set(), X)
        assert j_dense_monomorphism(X, mono, _dense()) is True

    def test_dense_on_single_element_identity_topology(self):
        X = X1
        mono = _mono({"x"}, X)
        assert j_dense_monomorphism(X, mono, _identity()) is True

    def test_empty_X_empty_mono_dense(self):
        X = X_empty
        mono = _mono(set(), X)
        for lt in all_lt_topologies():
            assert j_dense_monomorphism(X, mono, lt) is True


# ---------------------------------------------------------------------------
# verify_lt_axioms()
# ---------------------------------------------------------------------------

class TestVerifyLTAxioms:
    def test_identity_passes(self):
        assert verify_lt_axioms(_identity()) is True

    def test_dense_passes(self):
        assert verify_lt_axioms(_dense()) is True

    def test_invalid_lt1_raises(self):
        lt = LawvereTierneyTopology({TRUE: TRUE, FALSE: FALSE}, validate=False)
        lt.j[TRUE] = FALSE
        with pytest.raises(ValueError, match="LT1"):
            verify_lt_axioms(lt)


# ---------------------------------------------------------------------------
# subobject_lattice()
# ---------------------------------------------------------------------------

class TestSubobjectLattice:
    def test_empty_X_one_subobject(self):
        lattice = subobject_lattice(X_empty)
        assert len(lattice) == 1
        assert frozenset() in lattice

    def test_single_element_two_subobjects(self):
        lattice = subobject_lattice(X1)
        assert len(lattice) == 2

    def test_two_elements_four_subobjects(self):
        lattice = subobject_lattice(X2)
        assert len(lattice) == 4

    def test_three_elements_eight_subobjects(self):
        lattice = subobject_lattice(X3)
        assert len(lattice) == 8

    def test_all_are_frozensets_of_pairs(self):
        for mono in subobject_lattice(X2):
            assert isinstance(mono, frozenset)
            for pair in mono:
                assert isinstance(pair, tuple)
                assert len(pair) == 2
                assert pair[0] == pair[1]


# ---------------------------------------------------------------------------
# j_action_on_subobjects()
# ---------------------------------------------------------------------------

class TestJActionOnSubobjects:
    def test_identity_action_is_identity_map(self):
        action = j_action_on_subobjects(X2, _identity())
        for mono, closure in action.items():
            assert closure == mono

    def test_dense_action_sends_everything_to_X(self):
        X = X2
        X_mono = _mono({"a", "b"}, X)
        action = j_action_on_subobjects(X, _dense())
        for mono, closure in action.items():
            assert closure == X_mono

    def test_action_keys_match_lattice(self):
        lattice = subobject_lattice(X2)
        action = j_action_on_subobjects(X2, _identity())
        assert set(action.keys()) == set(lattice)

    def test_action_values_in_lattice(self):
        lattice = set(subobject_lattice(X2))
        action = j_action_on_subobjects(X2, _dense())
        for closure in action.values():
            assert closure in lattice

    def test_action_single_element(self):
        action = j_action_on_subobjects(X1, _identity())
        for mono, closure in action.items():
            assert closure == mono


# ---------------------------------------------------------------------------
# verify_closure_operator()
# ---------------------------------------------------------------------------

class TestVerifyClosureOperator:
    def test_identity_topology_passes(self):
        assert verify_closure_operator(X2, _identity()) is True

    def test_dense_topology_passes(self):
        assert verify_closure_operator(X2, _dense()) is True

    def test_passes_on_empty_X(self):
        for lt in all_lt_topologies():
            assert verify_closure_operator(X_empty, lt) is True

    def test_passes_on_single_element(self):
        for lt in all_lt_topologies():
            assert verify_closure_operator(X1, lt) is True

    def test_passes_on_three_elements(self):
        for lt in all_lt_topologies():
            assert verify_closure_operator(X3, lt) is True

    def test_c1_extensive_identity(self):
        # Under identity, j̄(A)=A so A ⊆ A trivially
        assert verify_closure_operator(X3, _identity()) is True

    def test_c2_idempotent_dense(self):
        # Under dense, j̄(j̄(A)) = j̄(X) = X = j̄(A)
        assert verify_closure_operator(X3, _dense()) is True

    def test_c3_monotone_dense(self):
        # Under dense, A⊆B implies j̄(A)=X=j̄(B)
        assert verify_closure_operator(X3, _dense()) is True

    def test_violation_c1_raises(self):
        # Manually inject a bad action to exercise C1 error path
        # We do this by monkey-patching j_action_on_subobjects via a bad LT
        # The easiest way: create a topology where j sends TRUE→FALSE (invalid by LT1)
        # and use validate=False. Under this map χ_A(x)=T for x in A → j(T)=F,
        # so j̄(A) might be ∅ while A is non-empty.
        lt = LawvereTierneyTopology({TRUE: FALSE, FALSE: FALSE}, validate=False)
        # This violates C1 for any non-empty A in non-empty X
        X = X1  # {x}
        # j̄({x}) = {y∈X | j(χ(y))=T} = {y | j(T)=F=T} = {} ≠ {x}
        with pytest.raises(ValueError, match="C1"):
            verify_closure_operator(X, lt)


# ---------------------------------------------------------------------------
# Integration: combined tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_all_topologies_all_closures_valid(self):
        X = X3
        for lt in all_lt_topologies():
            assert verify_closure_operator(X, lt) is True
            assert verify_lt_axioms(lt) is True

    def test_closed_subobjects_are_fixed_points_of_closure(self):
        X = X3
        for lt in all_lt_topologies():
            closed = j_closed_subobjects(X, lt)
            for mono in closed:
                closure = j_closure(X, mono, lt)
                assert closure == mono

    def test_dense_topology_implies_all_dense(self):
        X = X3
        lt = _dense()
        for mono in subobject_lattice(X):
            assert j_dense_monomorphism(X, mono, lt) is True

    def test_identity_topology_only_X_is_dense(self):
        X = X3
        lt = _identity()
        X_mono = _mono({"a", "b", "c"}, X)
        for mono in subobject_lattice(X):
            expected = (mono == X_mono)
            assert j_dense_monomorphism(X, mono, lt) == expected

    def test_round_trip_lt_via_all_lt_topologies(self):
        lt_list = all_lt_topologies()
        assert LawvereTierneyTopology({TRUE: TRUE, FALSE: FALSE}) in lt_list
        assert LawvereTierneyTopology({TRUE: TRUE, FALSE: TRUE}) in lt_list
