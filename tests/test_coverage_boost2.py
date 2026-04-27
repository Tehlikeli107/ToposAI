"""
Coverage tests for:
  - topos_ai.optim      (raises, closure, grad=None)
  - topos_ai.lawvere_tierney (neural; closed_topology, check_axioms branch)
  - topos_ai.kan        (mode="right", bad mode, universality_loss)
  - topos_ai.adjoint    (hom_isomorphism, NeuralAdjoint)
  - topos_ai.verification (empty-entity → "entity", filename=None, Lean succeed mock)
  - topos_ai.formal_yoneda (cardinality mismatch, Phi roundtrip errors, naturality False)
  - topos_ai.formal_lawvere_tierney (LT2/LT3 direct violations, C2/C3 mock violations)
  - topos_ai.topos (verify_ccc error paths, verify_pullback error paths via mock)
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# topos_ai.optim
# ---------------------------------------------------------------------------
from topos_ai.optim import ToposAdam


def _make_params():
    p = nn.Parameter(torch.tensor([1.0, 2.0]))
    return [p]


class TestToposAdamRaises:
    def test_negative_lr_raises(self):
        with pytest.raises(ValueError, match="learning rate"):
            ToposAdam(_make_params(), lr=-0.001)

    def test_negative_eps_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            ToposAdam(_make_params(), eps=-1e-8)

    def test_bad_beta0_raises(self):
        with pytest.raises(ValueError, match="beta"):
            ToposAdam(_make_params(), betas=(-0.1, 0.999))

    def test_bad_beta1_too_large_raises(self):
        with pytest.raises(ValueError, match="beta"):
            ToposAdam(_make_params(), betas=(0.9, 1.1))


class TestToposAdamStep:
    def test_step_with_closure(self):
        p = nn.Parameter(torch.tensor([1.0, 2.0]))
        opt = ToposAdam([p], lr=1e-3)
        loss_tensor = torch.tensor(0.5)

        def closure():
            return loss_tensor

        loss = opt.step(closure=closure)
        assert loss is loss_tensor

    def test_step_skips_param_with_no_grad(self):
        p = nn.Parameter(torch.tensor([1.0, 2.0]))
        opt = ToposAdam([p], lr=1e-3)
        # p.grad is None → line 41: continue (no update)
        loss = opt.step()
        assert loss is None

    def test_step_updates_param_with_grad(self):
        p = nn.Parameter(torch.tensor([0.5, -0.5]))
        opt = ToposAdam([p], lr=1e-3)
        p.grad = torch.tensor([0.1, -0.1])
        old = p.data.clone()
        opt.step()
        assert not torch.equal(p.data, old)


# ---------------------------------------------------------------------------
# topos_ai.lawvere_tierney (neural)
# ---------------------------------------------------------------------------
from topos_ai.lawvere_tierney import LawvereTierneyTopology as NeuralLT


class TestNeuralLawvereTierney:
    def _lt(self):
        return NeuralLT()

    def test_closed_topology_line27(self):
        lt = self._lt()
        P = torch.tensor([0.3, 0.7])
        C = torch.tensor([0.5, 0.2])
        result = lt.closed_topology(P, C)
        assert result.shape == P.shape

    def test_check_axioms_closed_topology_branch(self):
        """Lines 41-42: check_axioms with a closed_topology function."""
        lt = self._lt()
        P = torch.tensor([0.3, 0.7])
        Q = torch.tensor([0.4, 0.6])
        C = torch.tensor([0.5, 0.5])
        # topology_func.__name__ == "closed_topology" → enters if-branch
        ax1, ax2, ax3 = lt.check_axioms(P, Q, C, lt.closed_topology)
        assert isinstance(ax1, float)
        assert isinstance(ax2, float)
        assert isinstance(ax3, float)

    def test_check_axioms_double_negation_branch(self):
        """Else branch: check_axioms with double_negation_topology."""
        lt = self._lt()
        P = torch.tensor([0.3, 0.7])
        Q = torch.tensor([0.4, 0.6])
        C = torch.zeros(2)
        ax1, ax2, ax3 = lt.check_axioms(P, Q, C, lt.double_negation_topology)
        assert isinstance(ax1, float)


# ---------------------------------------------------------------------------
# topos_ai.kan
# ---------------------------------------------------------------------------
from topos_ai.kan import NeuralKanExtension, KanAdjunction


class TestNeuralKanExtension:
    def test_right_mode(self):
        proxy = NeuralKanExtension(num_source=3, dim_c=4, dim_e=4, dim_d=4, mode="right")
        tgt = torch.randn(2, 4)
        out = proxy(tgt)
        assert out.shape[-1] == 4

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            NeuralKanExtension(num_source=3, dim_c=4, dim_e=4, dim_d=4, mode="unknown")


class TestKanAdjunctionUniversalityLoss:
    def test_universality_loss(self):
        adj = KanAdjunction(dim_c=4, dim_e=4, dim_d=4)
        source = torch.randn(3, 4)
        target = torch.randn(2, 4)
        true_vals = torch.randn(2, 4)
        loss = adj.universality_loss(source, target, true_vals)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# topos_ai.adjoint
# ---------------------------------------------------------------------------
from topos_ai.adjoint import AdjointPair, NeuralAdjoint


class TestAdjointHomIsomorphism:
    def test_hom_isomorphism_line34(self):
        F_map = nn.Linear(4, 4)
        G_map = nn.Linear(4, 4)
        pair = AdjointPair(F_map, G_map)
        f_mor = torch.randn(4)
        g_mor = torch.randn(4)
        result = pair.hom_isomorphism(f_mor, g_mor)
        assert isinstance(result.item(), float)


class TestNeuralAdjoint:
    def test_construction_lines74_84(self):
        adj = NeuralAdjoint(dim_c=8, dim_d=4, hidden=16)
        x = torch.randn(2, 8)
        y = torch.randn(2, 4)
        loss_u, loss_c = adj.triangle_loss(x, y)
        assert loss_u.item() >= 0.0


# ---------------------------------------------------------------------------
# topos_ai.verification
# ---------------------------------------------------------------------------
from topos_ai.verification import Lean4VerificationBridge


class TestLean4VerificationBridge:
    def test_empty_entity_becomes_entity_keyword(self):
        """Line 22: cleaned='' after stripping → cleaned='entity'."""
        bridge = Lean4VerificationBridge(["!!!"])
        assert bridge.entities[0] == "entity"

    def test_digit_prefix_gets_v_prefix(self):
        """Line 24: cleaned starts with digit → 'v_' + cleaned."""
        bridge = Lean4VerificationBridge(["1abc"])
        assert bridge.entities[0] == "v_1abc"

    def test_prove_theorem_no_filename_uses_timestamp(self):
        """Line 90: filename=None → generates timestamp filename."""
        bridge = Lean4VerificationBridge(["A", "B"])
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError
            result, script, reason = bridge.prove_theorem([0, 1], 0.9)
        assert result is None
        assert "Compiler Not Found" in reason

    def test_prove_theorem_lean_succeeds(self):
        """Lines 103-106: returncode=0 → returns (True, script, output)."""
        bridge = Lean4VerificationBridge(["A", "B"])
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            ok, script, output = bridge.prove_theorem([0, 1], 0.95, filename="test.lean")
        assert ok is True
        assert "topos_proof" in script

    def test_prove_theorem_lean_fails(self):
        """Line 108-110: returncode!=0 → returns (False, script, output)."""
        bridge = Lean4VerificationBridge(["A", "B"])
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error"
        with patch("subprocess.run", return_value=mock_result):
            ok, script, output = bridge.prove_theorem([0, 1], 0.5, filename="test.lean")
        assert ok is False

    def test_prove_theorem_lean_timeout(self):
        """Lines 116-117: TimeoutExpired → returns (None, script, 'Timeout')."""
        import subprocess
        bridge = Lean4VerificationBridge(["A", "B"])
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("lean", 10)):
            ok, script, reason = bridge.prove_theorem([0, 1], 0.5, filename="test.lean")
        assert ok is None
        assert reason == "Timeout"


# ---------------------------------------------------------------------------
# topos_ai.formal_yoneda
# ---------------------------------------------------------------------------
from topos_ai.formal_category import FiniteCategory
from topos_ai.formal_kan import FiniteSetFunctor
from topos_ai.formal_yoneda import (
    verify_yoneda,
    verify_yoneda_naturality_in_A,
    yoneda_inverse,
    yoneda_evaluate,
)


def _arrow_cat():
    return FiniteCategory(
        objects=["A", "B"],
        morphisms={"idA": ("A", "A"), "idB": ("B", "B"), "f": ("A", "B")},
        identities={"A": "idA", "B": "idB"},
        composition={
            ("idA", "idA"): "idA", ("idB", "idB"): "idB",
            ("f", "idA"): "f", ("idB", "f"): "f",
        },
    )


def _simple_functor(cat):
    """F: A↦{"x"}, B↦{"y"}, f↦{x→y}."""
    return FiniteSetFunctor(
        category=cat,
        objects_map={"A": frozenset(["x"]), "B": frozenset(["y"])},
        morphism_map={
            "idA": {"x": "x"},
            "idB": {"y": "y"},
            "f": {"x": "y"},
        },
    )


def _two_elem_functor(cat):
    """F: A↦{"x1","x2"}, B↦{"y1","y2"}, f preserving."""
    return FiniteSetFunctor(
        category=cat,
        objects_map={"A": frozenset(["x1", "x2"]), "B": frozenset(["y1", "y2"])},
        morphism_map={
            "idA": {"x1": "x1", "x2": "x2"},
            "idB": {"y1": "y1", "y2": "y2"},
            "f": {"x1": "y1", "x2": "y2"},
        },
    )


class TestVerifyYonedaErrors:
    def test_cardinality_mismatch_line188(self):
        """Line 188: len(nats) != len(FA) → ValueError."""
        cat = _arrow_cat()
        F = _simple_functor(cat)
        # FA has 1 element; mock nats to be empty
        with patch("topos_ai.formal_yoneda.all_natural_transformations", return_value=[]):
            with pytest.raises(ValueError, match="cardinality mismatch"):
                verify_yoneda(cat, F, "A")

    def test_phi_phi_inv_not_identity_line198(self):
        """Line 198: Φ(Φ⁻¹(x)) ≠ x → ValueError."""
        cat = _arrow_cat()
        F = _simple_functor(cat)
        # Fake nat with correct cardinality (1 element)
        fake_nat = {"A": {"idA": "x"}, "B": {"f": "y"}}
        # Mock nats to return [fake_nat], and yoneda_inverse to return wrong nat
        wrong_nat = {"A": {"idA": "WRONG"}, "B": {"f": "y"}}
        with patch("topos_ai.formal_yoneda.all_natural_transformations", return_value=[fake_nat]):
            with patch("topos_ai.formal_yoneda.yoneda_inverse", return_value=wrong_nat):
                # yoneda_evaluate(wrong_nat, cat, "A") = wrong_nat["A"]["idA"] = "WRONG"
                # "WRONG" != "x" → line 198
                with pytest.raises(ValueError, match="Φ"):
                    verify_yoneda(cat, F, "A")

    def test_phi_inv_phi_not_identity_line207(self):
        """Line 207: Φ⁻¹(Φ(α)) ≠ α → ValueError."""
        cat = _arrow_cat()
        F = _simple_functor(cat)
        fake_nat = {"A": {"idA": "x"}, "B": {"f": "y"}}
        different_nat = {"A": {"idA": "x"}, "B": {"f": "WRONG"}}
        # Check 2 passes (real yoneda_inverse), check 3 returns different_nat ≠ fake_nat
        real_inv = yoneda_inverse("x", cat, F, "A")
        with patch("topos_ai.formal_yoneda.all_natural_transformations", return_value=[fake_nat]):
            with patch(
                "topos_ai.formal_yoneda.yoneda_inverse",
                side_effect=[real_inv, different_nat],
            ):
                with pytest.raises(ValueError, match="Φ"):
                    verify_yoneda(cat, F, "A")


class TestVerifyYonedaNaturalityFalse:
    def test_naturality_returns_false_line255(self):
        """Line 255: lhs != rhs → return False."""
        cat = _arrow_cat()
        F = _two_elem_functor(cat)
        # Bad nat: alpha_A(idA)="x1" but alpha_B(f)="WRONG" instead of "y1"
        bad_nat = {"A": {"idA": "x1"}, "B": {"f": "WRONG"}}
        with patch("topos_ai.formal_yoneda.all_natural_transformations", return_value=[bad_nat]):
            result = verify_yoneda_naturality_in_A(cat, F, "A", "B", "f")
        assert result is False


# ---------------------------------------------------------------------------
# topos_ai.formal_lawvere_tierney  (lines 120, 132, 383, 398)
# ---------------------------------------------------------------------------
from topos_ai.formal_lawvere_tierney import (
    LawvereTierneyTopology as FormalLT,
    verify_closure_operator,
)
from topos_ai.topos import TRUE, FALSE


class TestFormalLTAxiomViolations:
    def test_lt2_violation_line120(self):
        """j = {T:F, F:T} violates LT2: j(j(T))=T≠F=j(T)."""
        lt = FormalLT(j={TRUE: FALSE, FALSE: TRUE}, validate=False)
        with pytest.raises(ValueError, match="LT2"):
            lt._check_lt2()

    def test_lt3_violation_line132(self):
        """j = {T:F, F:T} violates LT3: j(T∧F)=T ≠ F∧T=F."""
        lt = FormalLT(j={TRUE: FALSE, FALSE: TRUE}, validate=False)
        with pytest.raises(ValueError, match="LT3"):
            lt._check_lt3()


class TestVerifyClosureOperatorViolations:
    def _make_monos(self):
        m_empty = frozenset()
        m_a = frozenset({("a", "a")})
        m_ab = frozenset({("a", "a"), ("b", "b")})
        return m_empty, m_a, m_ab

    def test_c2_idempotent_violation_line383(self):
        """Line 383: j̄(j̄(A)) ≠ j̄(A) → ValueError."""
        m_empty, m_a, m_ab = self._make_monos()
        # j̄(∅) = m_a, but j̄(m_a) = m_ab ≠ m_a → C2 fails
        bad_action = {m_empty: m_a, m_a: m_ab, m_ab: m_ab}
        lt = FormalLT.identity()
        X = frozenset({"a", "b"})
        with patch(
            "topos_ai.formal_lawvere_tierney.j_action_on_subobjects",
            return_value=bad_action,
        ):
            with pytest.raises(ValueError, match="C2"):
                verify_closure_operator(X, lt)

    def test_c3_monotone_violation_line398(self):
        """Line 398: A⊆B but j̄(A)⊄j̄(B) → ValueError (C1 and C2 pass)."""
        m_empty, m_a, m_ab = self._make_monos()
        # j̄(∅) = m_ab ({a,b}), j̄(m_a) = m_a ({a}), j̄(m_ab) = m_ab
        # C1: {} ⊆ {a,b} ✓, {a} ⊆ {a} ✓, {a,b} ⊆ {a,b} ✓
        # C2: j̄(j̄(∅)) = j̄(m_ab) = m_ab = j̄(∅) ✓, others ✓
        # C3: {} ⊆ {a} but {a,b} ⊄ {a} → fails
        bad_action = {m_empty: m_ab, m_a: m_a, m_ab: m_ab}
        lt = FormalLT.identity()
        X = frozenset({"a", "b"})
        with patch(
            "topos_ai.formal_lawvere_tierney.j_action_on_subobjects",
            return_value=bad_action,
        ):
            with pytest.raises(ValueError, match="C3"):
                verify_closure_operator(X, lt)


# ---------------------------------------------------------------------------
# topos_ai.topos  (verify_ccc + verify_pullback error paths)
# ---------------------------------------------------------------------------
from topos_ai.topos import verify_ccc, SubobjectClassifier, all_finset_morphisms


class TestVerifyCCCErrors:
    def test_cardinality_mismatch_line208(self):
        """Line 208: |Hom(X×Y,Z)| ≠ |Hom(X,Z^Y)| → ValueError."""
        X = frozenset({"a"})
        Y = frozenset({"b"})
        Z = frozenset({"c"})
        # Return lists so len() gives different values on the two calls
        with patch(
            "topos_ai.topos.all_finset_morphisms",
            side_effect=[[object(), object()], [object()]],
        ):
            with pytest.raises(ValueError, match="cardinality mismatch"):
                verify_ccc(X, Y, Z)

    def test_uncurry_curry_not_identity_line218(self):
        """Line 218: uncurry(curry(f)) ≠ f → ValueError."""
        X = frozenset({"a"})
        Y = frozenset({"b"})
        Z = frozenset({"c"})
        # Build a real single morphism f: {a}×{b} → {c}
        f = frozenset({(("a", "b"), "c")})
        ZY = frozenset([frozenset({("b", "c")})])
        real_g = frozenset({("a", frozenset({("b", "c")}))})
        # Mock: all_finset_morphisms returns [f] and [real_g] (same size)
        # Then curry(f) → real_g, uncurry(real_g) → something ≠ f
        wrong_f = frozenset({(("a", "b"), "WRONG")})
        with patch("topos_ai.topos.all_finset_morphisms", side_effect=[frozenset([f]), frozenset([real_g])]):
            with patch("topos_ai.topos.curry", return_value=real_g):
                with patch("topos_ai.topos.uncurry", return_value=wrong_f):
                    with pytest.raises(ValueError, match="uncurry"):
                        verify_ccc(X, Y, Z)

    def test_curry_uncurry_not_identity_line226(self):
        """Line 226: curry(uncurry(g)) ≠ g → ValueError."""
        X = frozenset({"a"})
        Y = frozenset({"b"})
        Z = frozenset({"c"})
        f = frozenset({(("a", "b"), "c")})
        real_g = frozenset({("a", frozenset({("b", "c")}))})
        wrong_g = frozenset({("a", frozenset({("b", "WRONG")}))})
        with patch("topos_ai.topos.all_finset_morphisms", side_effect=[frozenset([f]), frozenset([real_g])]):
            with patch("topos_ai.topos.curry", side_effect=[real_g, wrong_g]):
                with patch("topos_ai.topos.uncurry", return_value=f):
                    with pytest.raises(ValueError, match="curry"):
                        verify_ccc(X, Y, Z)


class TestVerifyPullbackErrors:
    def _sc(self):
        return SubobjectClassifier()

    def test_commutativity_fails_line301(self):
        """Line 301: chi_dict[ma] != TRUE → ValueError."""
        sc = self._sc()
        X = frozenset({"a", "b"})
        mono = frozenset({("a", "a")})  # valid injection
        # Mock characteristic_morphism to return wrong chi: a→F instead of a→T
        wrong_chi = frozenset({("a", FALSE), ("b", FALSE)})
        with patch.object(sc, "characteristic_morphism", return_value=wrong_chi):
            with pytest.raises(ValueError, match="commutativity"):
                sc.verify_pullback(X, frozenset({"a"}), mono)

    def test_universality_fails_line308(self):
        """Line 308: preimage_T ≠ image → ValueError."""
        sc = self._sc()
        X = frozenset({"a", "b"})
        mono = frozenset({("a", "a")})
        # chi: a→T, b→T → preimage_T = {"a","b"} ≠ {"a"} = image → universality fails
        wrong_chi = frozenset({("a", TRUE), ("b", TRUE)})
        with patch.object(sc, "characteristic_morphism", return_value=wrong_chi):
            with pytest.raises(ValueError, match="universality"):
                sc.verify_pullback(X, frozenset({"a"}), mono)

    def test_uniqueness_fails_line325(self):
        """Line 325: len(valid) != 1 → ValueError."""
        sc = self._sc()
        X = frozenset({"a", "b"})
        mono = frozenset({("a", "a")})
        # Correct chi so C1/C2 pass, but mock all_finset_morphisms to return two valid candidates
        # Both: cand[a]=T, preimage={a}=image → both valid
        cand1 = frozenset({("a", TRUE), ("b", FALSE)})
        cand2 = frozenset({("a", TRUE), ("b", FALSE)})
        # We need two *distinct* candidates that both pass
        cand2 = frozenset({("a", TRUE), ("b", TRUE)})
        # Wait, cand2 has preimage={a,b} ≠ {a}, so it won't pass the second check.
        # Both must have preimage = {a}. Only cand1 does. Let me use a different approach:
        # Use X={a} and mono={("a","a")} so image={a}
        # The single valid candidate maps a→T, which is the correct chi.
        # To get 2 valid ones, mock all_finset_morphisms to return [cand1, cand1_dup]
        # where both have a→T
        X2 = frozenset({"a"})
        mono2 = frozenset({("a", "a")})
        correct_chi = frozenset({("a", TRUE)})
        # two identical morphisms that both pass
        dup1 = frozenset({("a", TRUE)})
        dup2 = frozenset({("a", TRUE)})
        # frozensets are equal, so a frozenset of [dup1, dup2] would have only 1 element
        # Use a list to keep two "candidates"
        with patch.object(sc, "characteristic_morphism", return_value=correct_chi):
            with patch(
                "topos_ai.topos.all_finset_morphisms",
                return_value=[dup1, dup2],  # list so iteration yields 2, both identical
            ):
                with pytest.raises(ValueError, match="found 2 valid"):
                    sc.verify_pullback(X2, frozenset({"a"}), mono2)
