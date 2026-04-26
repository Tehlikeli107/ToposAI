"""
Tests for topos_ai.adjunction — Formal Adjoint Functors.
"""
import pytest
from topos_ai.formal_category import FiniteCategory
from topos_ai.adjunction import FiniteAdjunction


# ------------------------------------------------------------------ #
# Shared category fixtures (poset adjunction from the experiment)      #
# ------------------------------------------------------------------ #

def _C():
    return FiniteCategory(
        objects=["0", "1"],
        morphisms={"id_0": ("0","0"), "id_1": ("1","1"), "f": ("0","1")},
        identities={"0": "id_0", "1": "id_1"},
        composition={
            ("id_0","id_0"): "id_0", ("id_1","id_1"): "id_1",
            ("f","id_0"): "f", ("id_1","f"): "f",
        },
    )


def _D():
    return FiniteCategory(
        objects=["A", "B"],
        morphisms={"id_A": ("A","A"), "id_B": ("B","B"), "g": ("A","B")},
        identities={"A": "id_A", "B": "id_B"},
        composition={
            ("id_A","id_A"): "id_A", ("id_B","id_B"): "id_B",
            ("g","id_A"): "g", ("id_B","g"): "g",
        },
    )


def _adjunction():
    """Build the F ⊣ G adjunction from formal_proof_adjoint_functors.py."""
    return FiniteAdjunction(
        C=_C(), D=_D(),
        F_obj={"0": "A", "1": "A"},
        F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
        G_obj={"A": "1", "B": "1"},
        G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
        unit={"0": "f", "1": "id_1"},
        counit={"A": "id_A", "B": "g"},
    )


# ------------------------------------------------------------------ #
# Construction and repr                                                 #
# ------------------------------------------------------------------ #

class TestAdjunctionConstruction:
    def test_constructs(self):
        adj = _adjunction()
        assert isinstance(adj, FiniteAdjunction)

    def test_repr(self):
        adj = _adjunction()
        assert "FiniteAdjunction" in repr(adj)

    def test_identity_adjunction_constructs(self):
        """Identity functor is self-adjoint on any category."""
        C = _C()
        adj = FiniteAdjunction(
            C=C, D=C,
            F_obj={"0": "0", "1": "1"},
            F_mor={"id_0": "id_0", "id_1": "id_1", "f": "f"},
            G_obj={"0": "0", "1": "1"},
            G_mor={"id_0": "id_0", "id_1": "id_1", "f": "f"},
            unit={"0": "id_0", "1": "id_1"},
            counit={"0": "id_0", "1": "id_1"},
        )
        assert isinstance(adj, FiniteAdjunction)


# ------------------------------------------------------------------ #
# Axiom checks                                                          #
# ------------------------------------------------------------------ #

class TestAdjunctionAxioms:
    def test_unit_naturality(self):
        adj = _adjunction()
        adj._check_unit_naturality()  # should not raise

    def test_counit_naturality(self):
        adj = _adjunction()
        adj._check_counit_naturality()

    def test_triangle_1(self):
        adj = _adjunction()
        adj._check_triangle_1()

    def test_triangle_2(self):
        adj = _adjunction()
        adj._check_triangle_2()

    def test_wrong_unit_raises(self):
        """Unit component with wrong target should raise at construction."""
        with pytest.raises(ValueError, match="unit"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
                unit={"0": "id_0", "1": "id_1"},   # η_0 should be f: 0→G(F(0))=1
                counit={"A": "id_A", "B": "g"},
            )

    def test_wrong_counit_raises(self):
        """Counit with wrong type should raise."""
        with pytest.raises(ValueError, match="counit"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
                unit={"0": "f", "1": "id_1"},
                counit={"A": "g", "B": "g"},   # ε_A should be id_A not g (g has wrong src)
            )

    def test_bad_F_morphism_type_raises(self):
        """F mapping morphism to wrong-typed morphism in D should raise."""
        C, D = _C(), _D()
        # F(0)=A, F(1)=A, but F(f)=g where g: A->B has target B != F(1)=A
        with pytest.raises(ValueError):
            FiniteAdjunction(
                C=C, D=D,
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "g"},  # g: A->B, target B != A=F(1)
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
                unit={"0": "f", "1": "id_1"},
                counit={"A": "id_A", "B": "g"},
            )


# ------------------------------------------------------------------ #
# Hom-set bijection                                                     #
# ------------------------------------------------------------------ #

class TestHomBijection:
    def test_hom_bijection_verified(self):
        adj = _adjunction()
        assert adj.verify_hom_bijection()

    def test_phi_maps_C_to_D(self):
        adj = _adjunction()
        # C(0, G(B)) = C(0, 1) = {f}, so phi(f, B) should be in D(F(0), B) = D(A, B) = {g}
        result = adj.phi("f", "B")
        assert result in adj.D.hom(adj.F_obj["0"], "B")

    def test_psi_maps_D_to_C(self):
        adj = _adjunction()
        # D(F(0), B) = D(A, B) = {g}; psi(g, 0) should be in C(0, G(B)) = C(0,1) = {f}
        result = adj.psi("g", "0")
        assert result in adj.C.hom("0", adj.G_obj["B"])

    def test_phi_psi_inverse(self):
        adj = _adjunction()
        phi_f = adj.phi("f", "B")
        assert adj.psi(phi_f, "0") == "f"

    def test_psi_phi_inverse(self):
        adj = _adjunction()
        psi_g = adj.psi("g", "0")
        assert adj.phi(psi_g, "B") == "g"

    def test_hom_C_accessor(self):
        adj = _adjunction()
        hom = adj.hom_C("0", "B")
        # C(0, G(B)) = C(0, 1) = morphisms from 0 to 1 = {f}
        assert "f" in hom

    def test_hom_D_accessor(self):
        adj = _adjunction()
        hom = adj.hom_D("0", "B")
        # D(F(0), B) = D(A, B) = {g}
        assert "g" in hom


# ------------------------------------------------------------------ #
# Free-forgetful style (identity functor adjunction)                    #
# ------------------------------------------------------------------ #

class TestIdentityAdjunction:
    def test_identity_hom_bijection(self):
        C = _C()
        adj = FiniteAdjunction(
            C=C, D=C,
            F_obj={"0": "0", "1": "1"},
            F_mor={"id_0": "id_0", "id_1": "id_1", "f": "f"},
            G_obj={"0": "0", "1": "1"},
            G_mor={"id_0": "id_0", "id_1": "id_1", "f": "f"},
            unit={"0": "id_0", "1": "id_1"},
            counit={"0": "id_0", "1": "id_1"},
        )
        assert adj.verify_hom_bijection()
