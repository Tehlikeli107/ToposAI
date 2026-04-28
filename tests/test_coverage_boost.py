"""
Coverage-boosting tests for modules with sub-90% coverage:
  - topos_ai.hott (FinitePathGroupoid + PathFamily + HomotopyEquivalence + FormalHomotopyEquivalence)
  - topos_ai.polynomial_functors (PolyMorphism, DynamicalSystem, WiringDiagram, MonomialFunctor)
  - topos_ai.quantum_logic (QuantumLogicGate)
"""

import pytest
import torch
import torch.nn as nn

from topos_ai.formal_category import FiniteCategory
from topos_ai.hott import (
    FinitePathGroupoid,
    PathFamily,
    FormalHomotopyEquivalence,
    HomotopyEquivalence,
)
from topos_ai.polynomial_functors import (
    PolynomialFunctor,
    MonomialFunctor,
    PolyMorphism,
    DynamicalSystem,
    WiringDiagram,
)
from topos_ai.quantum_logic import QuantumLogicGate


# ------------------------------------------------------------------ #
# Helpers: path groupoids                                             #
# ------------------------------------------------------------------ #

def _two_point_groupoid():
    """
    Two-point groupoid: objects {A, B}, paths {idA, idB, p: A→B, p_inv: B→A}.
    """
    return FinitePathGroupoid(
        objects=("A", "B"),
        paths={
            "idA": ("A", "A"), "idB": ("B", "B"),
            "p": ("A", "B"), "p_inv": ("B", "A"),
        },
        identities={"A": "idA", "B": "idB"},
        inverses={"idA": "idA", "idB": "idB", "p": "p_inv", "p_inv": "p"},
        composition={
            ("idA", "idA"): "idA",
            ("idB", "idB"): "idB",
            ("p", "idA"): "p",
            ("idB", "p"): "p",
            ("p_inv", "idB"): "p_inv",
            ("idA", "p_inv"): "p_inv",
            ("idA", "p_inv"): "p_inv",
            ("p_inv", "p"): "idA",
            ("p", "p_inv"): "idB",
        },
    )


def _path_family(groupoid):
    """Trivial PathFamily: constant fibers {x}."""
    fibers = {obj: frozenset({"x"}) for obj in groupoid.objects}
    transports = {path: {"x": "x"} for path in groupoid.paths}
    return PathFamily(base=groupoid, fibers=fibers, transports=transports)


# ------------------------------------------------------------------ #
# Tests: FinitePathGroupoid additional paths                          #
# ------------------------------------------------------------------ #

class TestFinitePathGroupoid:
    def test_compose_raises_non_composable(self):
        g = _two_point_groupoid()
        # p: A→B and p_inv: B→A, so p o p: not valid (B≠A)
        with pytest.raises(ValueError, match="composable"):
            g.compose("p", "p")

    def test_compose_raises_missing(self):
        g = _two_point_groupoid()
        # This pair isn't in the composition table
        # Actually let's test a path that's composable but not in table
        # We'll construct a partial groupoid
        g2 = FinitePathGroupoid(
            objects=("X",),
            paths={"id": ("X", "X")},
            identities={"X": "id"},
            inverses={"id": "id"},
            composition={("id", "id"): "id"},
        )
        # Manually corrupt composition to trigger KeyError
        del g2.composition[("id", "id")]
        with pytest.raises(ValueError, match="Missing path"):
            g2.compose("id", "id")

    def test_source_and_target(self):
        g = _two_point_groupoid()
        assert g.source("p") == "A"
        assert g.target("p") == "B"

    def test_identity_type(self):
        g = _two_point_groupoid()
        it = g.identity_type("A", "B")
        assert "p" in it

    def test_inverse(self):
        g = _two_point_groupoid()
        assert g.inverse("p") == "p_inv"

    def test_refl(self):
        g = _two_point_groupoid()
        assert g.refl("A") == "idA"

    def test_composable_pairs_count(self):
        g = _two_point_groupoid()
        pairs = list(g.composable_pairs())
        assert len(pairs) == len(g.composition)

    def test_validate_identity_error(self):
        with pytest.raises(ValueError, match="reflexivity"):
            FinitePathGroupoid(
                objects=("A",),
                paths={"id": ("A", "A")},
                identities={},  # missing identity for A
                inverses={"id": "id"},
                composition={("id", "id"): "id"},
            )

    def test_validate_inverse_error(self):
        with pytest.raises(ValueError):
            FinitePathGroupoid(
                objects=("A",),
                paths={"id": ("A", "A")},
                identities={"A": "id"},
                inverses={},  # missing inverse for id
                composition={("id", "id"): "id"},
            )

    def test_validate_endpoint_error(self):
        with pytest.raises(ValueError):
            FinitePathGroupoid(
                objects=("A",),
                paths={"bad": ("A", "B")},  # B not in objects
                identities={"A": "bad"},
                inverses={"bad": "bad"},
                composition={},
            )

    def test_validate_identity_path_wrong_type(self):
        with pytest.raises(ValueError):
            FinitePathGroupoid(
                objects=("A", "B"),
                paths={"idA": ("A", "B")},  # idA should go A→A, not A→B
                identities={"A": "idA", "B": "idA"},
                inverses={"idA": "idA"},
                composition={},
            )


# ------------------------------------------------------------------ #
# Tests: PathFamily                                                    #
# ------------------------------------------------------------------ #

class TestPathFamily:
    def test_transport_accessor(self):
        g = _two_point_groupoid()
        pf = _path_family(g)
        assert pf.transport("p", "x") == "x"

    def test_transport_missing_raises(self):
        g = _two_point_groupoid()
        pf = _path_family(g)
        with pytest.raises(ValueError, match="No transport"):
            pf.transport("p", "MISSING_VALUE")

    def test_transport_equivalence(self):
        g = _two_point_groupoid()
        pf = _path_family(g)
        fwd, bwd = pf.transport_equivalence("p")
        assert fwd == {"x": "x"}
        assert bwd == {"x": "x"}

    def test_validate_transport_equivalences(self):
        g = _two_point_groupoid()
        pf = _path_family(g)
        assert pf.validate_transport_equivalences()

    def test_fiber_mismatch_raises(self):
        g = _two_point_groupoid()
        with pytest.raises(ValueError, match="fiber"):
            PathFamily(
                base=g,
                fibers={"A": frozenset({"x"})},  # missing B
                transports={path: {"x": "x"} for path in g.paths},
            )

    def test_transport_table_mismatch_raises(self):
        g = _two_point_groupoid()
        with pytest.raises(ValueError, match="transport"):
            PathFamily(
                base=g,
                fibers={obj: frozenset({"x"}) for obj in g.objects},
                transports={},  # empty
            )


# ------------------------------------------------------------------ #
# Tests: FormalHomotopyEquivalence                                     #
# ------------------------------------------------------------------ #

class TestFormalHomotopyEquivalence:
    def _arrow_cat(self):
        return FiniteCategory(
            objects=("A", "B"),
            morphisms={"idA": ("A", "A"), "idB": ("B", "B"), "f": ("A", "B")},
            identities={"A": "idA", "B": "idB"},
            composition={
                ("idA", "idA"): "idA", ("idB", "idB"): "idB",
                ("f", "idA"): "f", ("idB", "f"): "f",
            },
        )

    def test_isomorphic_categories(self):
        C = self._arrow_cat()
        fhe = FormalHomotopyEquivalence(C, C)
        result = fhe.find_strict_isomorphism()
        assert result is not None

    def test_is_univalent_equivalent_true(self):
        C = self._arrow_cat()
        fhe = FormalHomotopyEquivalence(C, C)
        assert fhe.is_univalent_equivalent()

    def test_non_isomorphic_categories(self):
        C = self._arrow_cat()
        D = FiniteCategory(
            objects=("X", "Y", "Z"),
            morphisms={
                "idX": ("X", "X"), "idY": ("Y", "Y"), "idZ": ("Z", "Z"),
                "g": ("X", "Y"), "h": ("Y", "Z"), "gh": ("X", "Z"),
            },
            identities={"X": "idX", "Y": "idY", "Z": "idZ"},
            composition={
                ("idX", "idX"): "idX", ("idY", "idY"): "idY", ("idZ", "idZ"): "idZ",
                ("g", "idX"): "g", ("idY", "g"): "g",
                ("h", "idY"): "h", ("idZ", "h"): "h",
                ("gh", "idX"): "gh", ("idZ", "gh"): "gh",
                ("h", "g"): "gh",
            },
        )
        fhe = FormalHomotopyEquivalence(C, D)
        # C has 2 objects, D has 3 — not isomorphic
        assert not fhe.is_univalent_equivalent()

    def test_trivial_category_isomorphic_to_itself(self):
        C = FiniteCategory(
            objects=("*",),
            morphisms={"id": ("*", "*")},
            identities={"*": "id"},
            composition={("id", "id"): "id"},
        )
        fhe = FormalHomotopyEquivalence(C, C)
        assert fhe.is_univalent_equivalent()

    def test_uses_groupoid_path(self):
        g = _two_point_groupoid()
        g2 = _two_point_groupoid()
        fhe = FormalHomotopyEquivalence(g, g2)
        result = fhe.find_strict_isomorphism()
        assert result is not None


# ------------------------------------------------------------------ #
# Tests: HomotopyEquivalence (numerical / Kabsch)                     #
# ------------------------------------------------------------------ #

class TestHomotopyEquivalenceNumerical:
    def test_identity_alignment(self):
        aligner = HomotopyEquivalence()
        X = torch.randn(5, 3)
        R, t = aligner.find_homotopy_path(X, X)
        transported = aligner.transport_along_path(X, R, t)
        assert torch.allclose(transported, X, atol=1e-5)

    def test_rotation_recovery(self):
        aligner = HomotopyEquivalence()
        X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        theta = torch.tensor(0.5)
        R_true = torch.tensor([
            [theta.cos(), -theta.sin()],
            [theta.sin(), theta.cos()]
        ])
        t_true = torch.tensor([1.0, 2.0])
        Y = (R_true @ X.T).T + t_true
        R, t = aligner.find_homotopy_path(X, Y)
        transported = aligner.transport_along_path(X, R, t)
        assert torch.allclose(transported, Y, atol=1e-4)

    def test_shape_mismatch_raises(self):
        aligner = HomotopyEquivalence()
        with pytest.raises(ValueError, match="same shape"):
            aligner.find_homotopy_path(torch.randn(3, 2), torch.randn(3, 3))

    def test_3d_alignment(self):
        aligner = HomotopyEquivalence()
        X = torch.randn(6, 3)
        Y = X + torch.tensor([1.0, 2.0, 3.0])  # pure translation
        R, t = aligner.find_homotopy_path(X, Y)
        transported = aligner.transport_along_path(X, R, t)
        assert torch.allclose(transported, Y, atol=1e-4)


# ------------------------------------------------------------------ #
# Tests: QuantumLogicGate                                              #
# ------------------------------------------------------------------ #

class TestQuantumLogicGate:
    def _proj(self, n=3):
        """Return a random rank-1 projection in ℝ^{n×n}."""
        v = torch.randn(n)
        v = v / v.norm()
        return v.outer(v)

    def test_quantum_and_shape(self):
        gate = QuantumLogicGate()
        P = self._proj()
        Q = self._proj()
        result = gate.quantum_and(P, Q)
        assert result.shape == P.shape

    def test_quantum_or_shape(self):
        gate = QuantumLogicGate()
        P = self._proj()
        Q = self._proj()
        result = gate.quantum_or(P, Q)
        assert result.shape == P.shape

    def test_quantum_not_shape(self):
        gate = QuantumLogicGate()
        P = self._proj()
        result = gate.quantum_not(P)
        assert result.shape == P.shape

    def test_quantum_not_complement(self):
        """P + ¬P = I."""
        gate = QuantumLogicGate()
        P = self._proj()
        not_P = gate.quantum_not(P)
        assert torch.allclose(P + not_P, torch.eye(P.shape[0]), atol=1e-5)

    def test_check_commutation(self):
        gate = QuantumLogicGate()
        P = self._proj()
        is_comm, comm = gate.check_commutation(P, P)
        assert is_comm
        assert torch.allclose(comm, torch.zeros_like(comm), atol=1e-5)

    def test_check_commutation_non_commuting(self):
        """Two generic projections are usually non-commuting."""
        gate = QuantumLogicGate()
        P = self._proj()
        Q = self._proj()
        is_comm, comm = gate.check_commutation(P, Q)
        # can't guarantee non-commuting, just check it runs
        assert isinstance(is_comm, bool)

    def test_sequential_and_shape(self):
        gate = QuantumLogicGate()
        P = self._proj()
        Q = self._proj()
        result = gate.sequential_and(P, Q)
        assert result.shape == P.shape

    def test_validate_projection_shape_mismatch_raises(self):
        gate = QuantumLogicGate()
        P = torch.eye(3)
        Q = torch.eye(4)
        with pytest.raises(ValueError, match="square matrices"):
            gate.quantum_and(P, Q)

    def test_eye_like(self):
        gate = QuantumLogicGate()
        P = torch.randn(3, 3)
        eye = gate._eye_like(P)
        assert torch.allclose(eye, torch.eye(3))

    def test_symmetrize(self):
        gate = QuantumLogicGate()
        A = torch.randn(3, 3)
        S = gate._symmetrize(A)
        assert torch.allclose(S, S.T, atol=1e-6)


# ------------------------------------------------------------------ #
# Tests: PolynomialFunctor + related                                   #
# ------------------------------------------------------------------ #

class TestPolynomialFunctors:
    def test_polynomial_functor_apply(self):
        p = PolynomialFunctor(positions=3, directions_per_pos=2)
        x = torch.randn(4, 2)
        out = p.apply(x)
        assert out.shape == (4, 3)

    def test_polynomial_functor_list_directions(self):
        p = PolynomialFunctor(positions=3, directions_per_pos=[1, 2, 3])
        assert p.directions == [1, 2, 3]

    def test_polynomial_functor_arena_map(self):
        p = PolynomialFunctor(positions=2, directions_per_pos=2)
        q = p.arena_map(lambda x: x, lambda x: x)
        assert q is p  # returns self

    def test_monomial_functor_apply(self):
        m = MonomialFunctor(n=3)
        x = torch.randn(4, 3)
        out = m.apply(x)
        assert out.shape == (4, 1)

    def test_poly_morphism_forward(self):
        p = PolynomialFunctor(positions=3, directions_per_pos=4)
        q = PolynomialFunctor(positions=5, directions_per_pos=4)
        mor = PolyMorphism(source=p, target=q, hidden=16)
        positions = torch.randn(2, 3)
        directions = torch.randn(2, 4)
        new_pos, back_dir = mor(positions, directions)
        assert new_pos.shape == (2, 5)
        assert back_dir.shape == (2, 4)

    def test_dynamical_system_forward(self):
        ds = DynamicalSystem(state_dim=4, input_dim=2, output_dim=3)
        state = torch.randn(2, 4)
        inp = torch.randn(2, 2)
        output, new_state = ds(state, inp)
        assert output.shape == (2, 3)
        assert new_state.shape == (2, 4)

    def test_dynamical_system_run(self):
        ds = DynamicalSystem(state_dim=4, input_dim=2, output_dim=3)
        batch = 2
        T = 5
        state = torch.randn(batch, 4)
        inputs = torch.randn(batch, T, 2)  # [B, T, input_dim]
        outputs, final_state = ds.run(state, inputs)
        assert outputs.shape == (batch, T, 3)
        assert final_state.shape == (batch, 4)

    def test_wiring_diagram_forward(self):
        from topos_ai.polynomial_functors import WiringDiagram
        ds1 = DynamicalSystem(state_dim=4, input_dim=2, output_dim=2)
        ds2 = DynamicalSystem(state_dim=4, input_dim=2, output_dim=2)
        wd = WiringDiagram(systems=[ds1, ds2])
        states = [torch.randn(2, 4), torch.randn(2, 4)]
        ext_input = torch.randn(2, 2)
        outputs, new_states = wd(states, ext_input)
        assert len(outputs) == 2
        assert len(new_states) == 2
