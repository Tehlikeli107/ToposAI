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

    def test_F_obj_missing_object_raises(self):
        with pytest.raises(ValueError, match="F_obj missing"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A"},          # missing "1"
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
                unit={"0": "f", "1": "id_1"},
                counit={"A": "id_A", "B": "g"},
            )

    def test_F_obj_not_in_D_raises(self):
        with pytest.raises(ValueError, match="not in D"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "NOWHERE"},   # "NOWHERE" not in D
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
                unit={"0": "f", "1": "id_1"},
                counit={"A": "id_A", "B": "g"},
            )

    def test_F_mor_not_in_D_raises(self):
        with pytest.raises(ValueError, match="not in D"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "GHOST"},  # not in D
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
                unit={"0": "f", "1": "id_1"},
                counit={"A": "id_A", "B": "g"},
            )

    def test_G_obj_not_in_C_raises(self):
        with pytest.raises(ValueError, match="not in C"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
                G_obj={"A": "NOWHERE", "B": "1"},   # "NOWHERE" not in C
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
                unit={"0": "f", "1": "id_1"},
                counit={"A": "id_A", "B": "g"},
            )

    def test_G_mor_not_in_C_raises(self):
        with pytest.raises(ValueError, match="not in C"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "GHOST", "g": "id_1"},  # GHOST not in C
                unit={"0": "f", "1": "id_1"},
                counit={"A": "id_A", "B": "g"},
            )

    def test_unit_component_not_morphism_raises(self):
        with pytest.raises(ValueError, match="unit"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
                unit={"0": "GHOST", "1": "id_1"},   # GHOST not a morphism in C
                counit={"A": "id_A", "B": "g"},
            )

    def test_counit_component_not_morphism_raises(self):
        with pytest.raises(ValueError, match="counit"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
                unit={"0": "f", "1": "id_1"},
                counit={"A": "GHOST", "B": "g"},   # GHOST not a morphism in D
            )

    def test_F_mor_missing_morphism_raises(self):
        with pytest.raises(ValueError, match="F_mor missing"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A"},   # missing "f"
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
                unit={"0": "f", "1": "id_1"},
                counit={"A": "id_A", "B": "g"},
            )

    def test_G_obj_missing_object_raises(self):
        with pytest.raises(ValueError, match="G_obj missing"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
                G_obj={"A": "1"},     # missing "B"
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
                unit={"0": "f", "1": "id_1"},
                counit={"A": "id_A", "B": "g"},
            )

    def test_G_mor_missing_morphism_raises(self):
        with pytest.raises(ValueError, match="G_mor missing"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "id_1"},   # missing "g"
                unit={"0": "f", "1": "id_1"},
                counit={"A": "id_A", "B": "g"},
            )

    def test_G_mor_type_mismatch_raises(self):
        """G morphism mapping to wrong-typed C morphism raises."""
        with pytest.raises(ValueError):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "f"},  # g:A->B mapped to f:0->1, G(B)=1 but f goes 0->1 not 1->1
                unit={"0": "f", "1": "id_1"},
                counit={"A": "id_A", "B": "g"},
            )

    def test_unit_missing_component_raises(self):
        with pytest.raises(ValueError, match="unit missing"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
                unit={"1": "id_1"},    # missing "0"
                counit={"A": "id_A", "B": "g"},
            )

    def test_unit_source_wrong_raises(self):
        """Unit component with wrong source (≠ c) raises."""
        with pytest.raises(ValueError, match="unit"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
                unit={"0": "id_1", "1": "id_1"},  # id_1 has source "1" ≠ "0"
                counit={"A": "id_A", "B": "g"},
            )

    def test_counit_missing_component_raises(self):
        with pytest.raises(ValueError, match="counit missing"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
                unit={"0": "f", "1": "id_1"},
                counit={"B": "g"},    # missing "A"
            )

    def test_counit_source_wrong_raises(self):
        """Counit component with source ≠ F(G(d)) raises."""
        with pytest.raises(ValueError, match="counit"):
            FiniteAdjunction(
                C=_C(), D=_D(),
                F_obj={"0": "A", "1": "A"},
                F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
                G_obj={"A": "1", "B": "1"},
                G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
                unit={"0": "f", "1": "id_1"},
                counit={"A": "id_B", "B": "g"},  # id_B: B->B, expected source F(G(A))=F(1)=A
            )

    def test_F_composition_violation_raises(self):
        """F that maps composite to a different morphism than F(after)∘F(before) raises."""
        # Use a larger category: 0 -f-> 1 -h-> 2 with composite k = h∘f
        # C3 and D2 where F constant except we corrupt F(k)
        C3 = FiniteCategory(
            objects=["0","1","2"],
            morphisms={
                "id0":("0","0"), "id1":("1","1"), "id2":("2","2"),
                "f":("0","1"), "h":("1","2"), "k":("0","2"),
            },
            identities={"0":"id0","1":"id1","2":"id2"},
            composition={
                ("id0","id0"):"id0", ("id1","id1"):"id1", ("id2","id2"):"id2",
                ("f","id0"):"f", ("id1","f"):"f",
                ("h","id1"):"h", ("id2","h"):"h",
                ("k","id0"):"k", ("id2","k"):"k",
                ("h","f"):"k",
            },
        )
        D2 = FiniteCategory(
            objects=["X","Y"],
            morphisms={"idX":("X","X"),"idY":("Y","Y"),"p":("X","Y")},
            identities={"X":"idX","Y":"idY"},
            composition={
                ("idX","idX"):"idX",("idY","idY"):"idY",
                ("p","idX"):"p",("idY","p"):"p",
            },
        )
        adj = FiniteAdjunction(
            C=C3, D=D2,
            F_obj={"0":"X","1":"X","2":"X"},
            F_mor={"id0":"idX","id1":"idX","id2":"idX","f":"idX","h":"idX","k":"p"},
            G_obj={"X":"0","Y":"0"},
            G_mor={"idX":"id0","idY":"id0","p":"id0"},
            unit={"0":"id0","1":"id0","2":"id0"},
            counit={"X":"idX","Y":"idX"},
            validate=False,
        )
        # F(k)=p but F(h)∘F(f)=idX∘idX=idX ≠ p → composition violation
        with pytest.raises(ValueError):
            adj._check_functor_F()

    def test_G_composition_violation_raises(self):
        """G that maps composite to a different morphism raises."""
        C3 = FiniteCategory(
            objects=["0","1","2"],
            morphisms={
                "id0":("0","0"),"id1":("1","1"),"id2":("2","2"),
                "f":("0","1"),"h":("1","2"),"k":("0","2"),
            },
            identities={"0":"id0","1":"id1","2":"id2"},
            composition={
                ("id0","id0"):"id0",("id1","id1"):"id1",("id2","id2"):"id2",
                ("f","id0"):"f",("id1","f"):"f",
                ("h","id1"):"h",("id2","h"):"h",
                ("k","id0"):"k",("id2","k"):"k",
                ("h","f"):"k",
            },
        )
        D2 = FiniteCategory(
            objects=["X","Y"],
            morphisms={"idX":("X","X"),"idY":("Y","Y"),"p":("X","Y")},
            identities={"X":"idX","Y":"idY"},
            composition={
                ("idX","idX"):"idX",("idY","idY"):"idY",
                ("p","idX"):"p",("idY","p"):"p",
            },
        )
        adj = FiniteAdjunction(
            C=C3, D=D2,
            F_obj={"0":"X","1":"X","2":"X"},
            F_mor={"id0":"idX","id1":"idX","id2":"idX","f":"idX","h":"idX","k":"idX"},
            G_obj={"X":"0","Y":"0"},
            G_mor={"idX":"id0","idY":"id0","p":"f"},  # G(p)=f: 0->1, but G(Y)=0, G(X)=0 → type mismatch
            unit={"0":"id0","1":"id0","2":"id0"},
            counit={"X":"idX","Y":"idX"},
            validate=False,
        )
        with pytest.raises(ValueError):
            adj._check_functor_G()

    def test_unit_naturality_violation_raises(self):
        """Unit component that makes naturality square fail raises."""
        adj = FiniteAdjunction(
            C=_C(), D=_D(),
            F_obj={"0": "A", "1": "A"},
            F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
            G_obj={"A": "1", "B": "1"},
            G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
            unit={"0": "f", "1": "id_1"},
            counit={"A": "id_A", "B": "g"},
            validate=False,
        )
        # G(F(m)) ∘ η_X = η_Y ∘ m.  For m=f: 0→1, LHS = id_1 ∘ f = f, RHS = id_1 ∘ f = f.
        # Change η_1 to f (0→1) — then RHS = C.compose("f","f") which requires target(f)=1=source(f)=0 FAIL
        # Instead corrupt G_mor so G(F(f)) gives something different:
        # G(F(f)) = G_mor[F_mor["f"]] = G_mor["id_A"]. Change G_mor["id_A"] so
        # compose(G_mor["id_A"], unit["0"]) ≠ compose(unit["1"], "f").
        # unit["0"]="f": 0→1, unit["1"]="id_1": 1→1.
        # RHS = C.compose("id_1","f") = f.
        # Make G_mor["id_A"] = "id_1" but unit["1"] = "f" → RHS = C.compose("f","f") → not composable.
        # Better: keep unit[1]=id_1, change unit[0] to id_0 (0→0):
        # LHS = C.compose("id_1", "id_0") → not composable (target(id_0)=0 ≠ source(id_1)=1).
        # So just assert any ValueError is raised:
        adj.unit["0"] = "id_0"
        with pytest.raises(ValueError):
            adj._check_unit_naturality()

    def test_counit_naturality_violation_raises(self):
        """Counit component that makes naturality square fail raises."""
        adj = FiniteAdjunction(
            C=_C(), D=_D(),
            F_obj={"0": "A", "1": "A"},
            F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
            G_obj={"A": "1", "B": "1"},
            G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
            unit={"0": "f", "1": "id_1"},
            counit={"A": "id_A", "B": "g"},
            validate=False,
        )
        # For m=g: A→B, LHS = D.compose("g","id_A")=g, RHS = D.compose("g",F_mor[G_mor["g"]])
        # Change counit["B"] = "id_A" (A→A) so D.compose("g","id_A")=g but
        # eps_Y∘F(G(m)) with Y=B, eps_B="id_A" and F(G(g))=F("id_1")="id_A":
        # RHS = D.compose("id_A","id_A")=id_A.  LHS = D.compose("g","id_A")=g ≠ id_A.
        adj.counit["B"] = "id_A"
        with pytest.raises(ValueError):
            adj._check_counit_naturality()

    def test_triangle_1_violation_raises(self):
        """ε_{F(c)} ∘ F(η_c) ≠ id_{F(c)} raises during _check_triangle_1."""
        # Build a richer example: C=arrow, D=arrow, F and G as identity
        C = _C();  D = _D()
        # Use the valid adjunction but swap counit so triangle fails:
        # Triangle 1: ε_{F(0)}∘F(η_0) = ε_A∘F(f) = id_A∘id_A = id_A ✓
        # Force F(η_0)=g (A→B) and ε_A=id_A → id_A∘g = g ≠ id_A.
        adj = FiniteAdjunction(
            C=C, D=D,
            F_obj={"0": "A", "1": "A"},
            F_mor={"id_0": "id_A", "id_1": "id_A", "f": "g"},  # F(f)=g: A→B
            G_obj={"A": "1", "B": "1"},
            G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
            unit={"0": "f", "1": "id_1"},
            counit={"A": "id_A", "B": "g"},
            validate=False,
        )
        # ε_{F(0)}=ε_A=id_A, F(η_0)=F(unit[0])=F(f)=g: A→B
        # D.compose(id_A, g): target(g)=B, source(id_A)=A — not composable → raises
        with pytest.raises(ValueError):
            adj._check_triangle_1()

    def test_triangle_2_violation_raises(self):
        """G(ε_d) ∘ η_{G(d)} ≠ id_{G(d)} raises during _check_triangle_2."""
        adj = FiniteAdjunction(
            C=_C(), D=_D(),
            F_obj={"0": "A", "1": "A"},
            F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
            G_obj={"A": "1", "B": "1"},
            G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
            unit={"0": "f", "1": "id_1"},
            counit={"A": "id_A", "B": "g"},
            validate=False,
        )
        # For d=B: G(ε_B)∘η_{G(B)} = G(g)∘η_1 = id_1∘id_1 = id_1 = id_{G(B)} ✓
        # Change unit so η_{G(B)}=η_1=f (0→1): target(f)=1≠source(G(eps_B)=id_1 source=1...
        # compose(id_1, f) → target(f)=1, source(id_1)=1 ✓ → id_1 ≠ id_1? That's equal!
        # So change G_mor["g"]="f" (0→1), then G(ε_B)=f: 0→1, η_{G(B)}=η_1="id_1": 1→1
        # C.compose("f","id_1"): target(id_1)=1, source(f)=0 — NOT composable → raises
        adj.G_mor["g"] = "f"
        with pytest.raises(ValueError):
            adj._check_triangle_2()


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
# Deep coverage: composition, naturality, triangle, and bijection       #
# ------------------------------------------------------------------ #

def _z2():
    """Z/2Z as a one-object FiniteCategory (monoid with a² = e)."""
    return FiniteCategory(
        objects=["*"],
        morphisms={"e": ("*", "*"), "a": ("*", "*")},
        identities={"*": "e"},
        composition={
            ("e", "e"): "e", ("a", "e"): "a",
            ("e", "a"): "a", ("a", "a"): "e",
        },
    )


class TestAdjunctionDeepCoverage:
    """Tests targeting specific error branches not covered by standard tests."""

    def test_F_composition_preserved_raises(self):
        """F(composite) ≠ F(after)∘F(before) raises line 128."""
        Z2 = _z2()
        # F: Z2 → Z2 where F(e)=e, F(a)=a  but we deliberately give F(a∘a=e)=a ≠ F(a)∘F(a)=e
        adj = FiniteAdjunction(
            C=Z2, D=Z2,
            F_obj={"*": "*"},
            F_mor={"e": "a", "a": "a"},   # F(e)=a, F(a)=a: F(a∘a=e)=a but F(a)∘F(a)=a∘a=e ≠ a
            G_obj={"*": "*"},
            G_mor={"e": "e", "a": "e"},
            unit={"*": "e"},
            counit={"*": "e"},
            validate=False,
        )
        with pytest.raises(ValueError):
            adj._check_functor_F()

    def test_G_composition_violation_line_157(self):
        """G(composite) ≠ G(after)∘G(before) raises line 157."""
        Z2 = _z2()
        adj = FiniteAdjunction(
            C=Z2, D=Z2,
            F_obj={"*": "*"},
            F_mor={"e": "e", "a": "e"},
            G_obj={"*": "*"},
            G_mor={"e": "a", "a": "a"},   # G(e)=a, G(a)=a: G(a∘a=e)=a but G(a)∘G(a)=a∘a=e ≠ a
            unit={"*": "e"},
            counit={"*": "e"},
            validate=False,
        )
        with pytest.raises(ValueError):
            adj._check_functor_G()

    def test_unit_naturality_failure_message(self):
        """Unit naturality failure raises with 'naturality' in message (line 224)."""
        Z2 = _z2()
        # Constant F and G, unit η_* = a.
        # For m=a: GFm = G(F(a))=G(e)=e, lhs=compose(e,a)=a, rhs=compose(a,a)=e → a≠e
        adj = FiniteAdjunction(
            C=Z2, D=Z2,
            F_obj={"*": "*"},
            F_mor={"e": "e", "a": "e"},
            G_obj={"*": "*"},
            G_mor={"e": "e", "a": "e"},
            unit={"*": "a"},
            counit={"*": "a"},
            validate=False,
        )
        with pytest.raises(ValueError, match="naturality"):
            adj._check_unit_naturality()

    def test_counit_naturality_failure_message(self):
        """Counit naturality failure raises with 'naturality' in message (line 241)."""
        Z2 = _z2()
        # For m=a: FGm=F(G(a))=F(e)=e, lhs=compose(a,a)=e, rhs=compose(a,e)=a → e≠a
        adj = FiniteAdjunction(
            C=Z2, D=Z2,
            F_obj={"*": "*"},
            F_mor={"e": "e", "a": "e"},
            G_obj={"*": "*"},
            G_mor={"e": "e", "a": "e"},
            unit={"*": "a"},
            counit={"*": "a"},
            validate=False,
        )
        with pytest.raises(ValueError, match="naturality"):
            adj._check_counit_naturality()

    def test_triangle_1_failure_message(self):
        """Triangle 1 failure raises with 'triangle' in message (line 268)."""
        Z2 = _z2()
        # F constant (F(a)=e), counit ε_*=a, unit η_*=a.
        # result = D.compose(a, F(unit[*]))=D.compose(a, F(a))=D.compose(a,e)=a ≠ e=id_{F(*)}
        adj = FiniteAdjunction(
            C=Z2, D=Z2,
            F_obj={"*": "*"},
            F_mor={"e": "e", "a": "e"},
            G_obj={"*": "*"},
            G_mor={"e": "e", "a": "e"},
            unit={"*": "a"},
            counit={"*": "a"},
            validate=False,
        )
        with pytest.raises(ValueError, match="triangle"):
            adj._check_triangle_1()

    def test_triangle_2_failure_message(self):
        """Triangle 2 failure raises with 'triangle' in message (line 282)."""
        Z2 = _z2()
        # G constant (G(a)=e), counit ε_*=a, unit η_*=a.
        # result = C.compose(G(counit[*]), unit[G(*)])=C.compose(G(a),a)=C.compose(e,a)=a ≠ e=id_{G(*)}
        adj = FiniteAdjunction(
            C=Z2, D=Z2,
            F_obj={"*": "*"},
            F_mor={"e": "e", "a": "e"},
            G_obj={"*": "*"},
            G_mor={"e": "e", "a": "e"},
            unit={"*": "a"},
            counit={"*": "a"},
            validate=False,
        )
        with pytest.raises(ValueError, match="triangle"):
            adj._check_triangle_2()

    def test_verify_hom_bijection_returns_false(self):
        """verify_hom_bijection returns False when Ψ∘Φ ≠ id (lines 353/356)."""
        Z2 = _z2()
        # phi(phi_mor, *) = D.compose(counit[*]=a, F_mor[phi_mor]=e) = compose(a,e)=a
        # psi(a, *) = C.compose(G_mor[a]=e, unit[*]=a) = compose(e,a)=a
        # psi(phi(e,*),*) = psi(a,*) = a ≠ e → returns False
        adj = FiniteAdjunction(
            C=Z2, D=Z2,
            F_obj={"*": "*"},
            F_mor={"e": "e", "a": "e"},
            G_obj={"*": "*"},
            G_mor={"e": "e", "a": "e"},
            unit={"*": "a"},
            counit={"*": "a"},
            validate=False,
        )
        assert adj.verify_hom_bijection() is False


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
