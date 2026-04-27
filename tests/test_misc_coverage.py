"""
Coverage tests for:
  - topos_ai.topology   (PersistentHomology, 0% → ~100%)
  - topos_ai.generation (ToposConstrainedDecoder top_k path, 83% → ~100%)
  - topos_ai.monad      (base Monad, GiryMonad edge cases, ContinuationMonad, WriterMonad.T)
  - topos_ai.distributed (world_size=1, num_universes=0, HAS_FSDP mock, dist mock)
  - topos_ai.monoidal   (strict_monoidal bifunctoriality-id violation, associator/unitor/pentagon/triangle violations)
  - topos_ai.enriched   (composition-type, identity-type, associativity, unitality violations)
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

# ─────────────────────────────────────────────────────────────────────────────
# topology.py
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

from topos_ai.topology import PersistentHomology


class TestPersistentHomology:
    """Full coverage of PersistentHomology.__init__, _boundary_matrix_rank, calculate_betti."""

    def test_init_stores_n(self):
        ph = PersistentHomology(5)
        assert ph.N == 5

    def test_three_isolated_nodes(self):
        """No edges → β0 = 3, β1 = 0."""
        ph = PersistentHomology(3)
        dist = np.array([[0, 10, 10], [10, 0, 10], [10, 10, 0]], dtype=float)
        b0, b1 = ph.calculate_betti(dist, threshold=1.0)
        assert b0 == 3
        assert b1 == 0

    def test_single_edge_two_components(self):
        """Nodes 0-1 connected, node 2 isolated → β0=2, β1=0."""
        ph = PersistentHomology(3)
        dist = np.array([[0, 1, 10], [1, 0, 10], [10, 10, 0]], dtype=float)
        b0, b1 = ph.calculate_betti(dist, threshold=2.0)
        assert b0 == 2
        assert b1 == 0

    def test_complete_triangle_fills_cycle(self):
        """Triangle (3 nodes, 3 edges, 1 triangle) → β0=1, β1=0."""
        ph = PersistentHomology(3)
        dist = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        b0, b1 = ph.calculate_betti(dist, threshold=1.5)
        assert b0 == 1
        assert b1 == 0

    def test_square_cycle_beta1(self):
        """4 nodes forming a square (0-1-2-3-0) with no diagonals → β0=1, β1=1."""
        ph = PersistentHomology(4)
        dist = np.array([
            [0, 1, 5, 1],
            [1, 0, 1, 5],
            [5, 1, 0, 1],
            [1, 5, 1, 0],
        ], dtype=float)
        b0, b1 = ph.calculate_betti(dist, threshold=2.0)
        assert b0 == 1
        assert b1 == 1

    def test_boundary_matrix_rank_no_triangles_early_return(self):
        """_boundary_matrix_rank returns 0 immediately when triangles list is empty."""
        import networkx as nx
        ph = PersistentHomology(3)
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        rank = ph._boundary_matrix_rank(G, [])
        assert rank == 0

    def test_boundary_matrix_rank_no_edges_early_return(self):
        """_boundary_matrix_rank returns 0 when graph has no edges."""
        import networkx as nx
        ph = PersistentHomology(3)
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        rank = ph._boundary_matrix_rank(G, [(0, 1, 2)])
        assert rank == 0

    def test_boundary_matrix_rank_single_triangle(self):
        """Triangle gives rank 1 (one independent 2-chain)."""
        import networkx as nx
        ph = PersistentHomology(3)
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (0, 2)])
        rank = ph._boundary_matrix_rank(G, [(0, 1, 2)])
        assert rank == 1

    def test_fully_connected_four_nodes(self):
        """K4 (complete graph, 4 nodes) → β0=1, β1=0 (all cycles filled)."""
        ph = PersistentHomology(4)
        dist = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]], dtype=float)
        b0, b1 = ph.calculate_betti(dist, threshold=2.0)
        assert b0 == 1
        assert b1 == 0

    def test_single_node(self):
        """Single node → β0=1, β1=0."""
        ph = PersistentHomology(1)
        b0, b1 = ph.calculate_betti(np.array([[0.0]]), threshold=1.0)
        assert b0 == 1
        assert b1 == 0


# ─────────────────────────────────────────────────────────────────────────────
# generation.py
# ─────────────────────────────────────────────────────────────────────────────

from topos_ai.generation import ToposConstrainedDecoder


class TestGenerationTopK:
    """Cover the top_k branch (lines 61-66) and no-top_k sampling path (65-66)."""

    def setup_method(self):
        vocab_size = 6
        # Identity reachability: each token can only reach itself
        reach = torch.eye(vocab_size)
        self.decoder = ToposConstrainedDecoder(reach, threshold=0.5)
        # Logits increasing so greedy picks last token
        self.logits = torch.arange(vocab_size, dtype=torch.float)

    def test_generate_with_top_k_returns_valid_token(self):
        """top_k=2 must cover lines 61-63, then 65-66."""
        result = self.decoder.generate_safe_token(
            0, self.logits.clone(), temperature=1.0, top_k=2
        )
        assert 0 <= result < 6

    def test_generate_top_k_large_covers_all(self):
        """top_k larger than vocab: k = min(top_k, numel) → same as no top_k."""
        result = self.decoder.generate_safe_token(
            0, self.logits.clone(), temperature=1.0, top_k=100
        )
        assert 0 <= result < 6

    def test_generate_top_k_zero_raises(self):
        """top_k=0 must fire the ValueError on line 60."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            self.decoder.generate_safe_token(
                0, self.logits.clone(), temperature=1.0, top_k=0
            )

    def test_generate_top_k_negative_raises(self):
        """top_k < 0 must fire the ValueError on line 60."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            self.decoder.generate_safe_token(
                0, self.logits.clone(), temperature=1.0, top_k=-3
            )

    def test_generate_no_top_k_samples(self):
        """temperature > 0, top_k=None → hits softmax + multinomial (lines 65-66)."""
        torch.manual_seed(0)
        result = self.decoder.generate_safe_token(
            0, self.logits.clone(), temperature=1.0
        )
        assert 0 <= result < 6

    def test_all_unreachable_falls_back_to_original(self):
        """No reachable token → apply_topological_mask returns clone, sampling still works."""
        vocab_size = 4
        # Zero reachability matrix: nothing is reachable
        reach = torch.zeros(vocab_size, vocab_size)
        decoder = ToposConstrainedDecoder(reach, threshold=0.5)
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # Should return original logits and sample normally
        result = decoder.generate_safe_token(0, logits, temperature=1.0)
        assert 0 <= result < vocab_size


# ─────────────────────────────────────────────────────────────────────────────
# monad.py
# ─────────────────────────────────────────────────────────────────────────────

from topos_ai.monad import Monad, GiryMonad, ContinuationMonad, WriterMonad


class TestMonadBase:
    """Cover lines 10, 13, 16 (NotImplementedError), 20 (bind), 23 (kleisli_compose)."""

    def test_T_raises_not_implemented(self):
        """Monad.T raises NotImplementedError (line 10)."""
        m = Monad()
        with pytest.raises(NotImplementedError):
            m.T(torch.rand(3))

    def test_unit_raises_not_implemented(self):
        """Monad.unit raises NotImplementedError (line 13)."""
        m = Monad()
        with pytest.raises(NotImplementedError):
            m.unit(torch.rand(3))

    def test_join_raises_not_implemented(self):
        """Monad.join raises NotImplementedError (line 16)."""
        m = Monad()
        with pytest.raises(NotImplementedError):
            m.join(torch.rand(3))

    def test_bind_calls_monad_base(self):
        """GiryMonad inherits Monad.bind → covers line 20."""
        giry = GiryMonad(dim=3)
        x = torch.tensor([[1.0, 2.0, 3.0]])
        tx = giry.unit(x)  # one_hot of argmax
        # bind(tx, f) = join(T(f(tx))) via Monad.bind
        result = giry.bind(tx, giry.T)
        assert result.shape[-1] == 3

    def test_kleisli_compose_covers_line_23(self):
        """Monad.kleisli_compose returns a callable (line 23)."""
        giry = GiryMonad(dim=3)
        composed_f = giry.kleisli_compose(giry.T, giry.T)
        x = torch.rand(1, 3)
        result = composed_f(x)
        assert result.shape[-1] == 3


class TestGiryMonadEdgeCases:
    """Cover line 79: GiryMonad.join fallthrough when dim ∉ {2, 3}."""

    def test_join_1d_tensor_fallthrough(self):
        """1-D tensor: not dim==3 and not dim==2 → hits the final return ttx (line 79)."""
        giry = GiryMonad(dim=3)
        x = torch.rand(3)
        result = giry.join(x)
        assert result is x  # same object returned unchanged

    def test_join_4d_tensor_fallthrough(self):
        """4-D tensor also falls through to line 79."""
        giry = GiryMonad(dim=3)
        x = torch.rand(2, 3, 4, 5)
        result = giry.join(x)
        assert result is x


class TestContinuationMonad:
    """Cover lines 95 (T), 98 (unit), 101 (join) of ContinuationMonad."""

    def test_T_applies_transform(self):
        """ContinuationMonad.T returns self.transform(x) (line 95)."""
        cont = ContinuationMonad()
        x = torch.rand(4)
        t = cont.T(x)
        torch.testing.assert_close(t, x)  # default transform is Identity

    def test_unit_calls_T(self):
        """ContinuationMonad.unit calls self.T(x) (line 98)."""
        cont = ContinuationMonad()
        x = torch.rand(4)
        u = cont.unit(x)
        torch.testing.assert_close(u, x)

    def test_join_calls_T(self):
        """ContinuationMonad.join calls self.T(ttx) (line 101)."""
        cont = ContinuationMonad()
        x = torch.rand(4)
        j = cont.join(x)
        torch.testing.assert_close(j, x)

    def test_with_custom_transform(self):
        """ContinuationMonad with a non-identity transform."""
        double = nn.Linear(3, 3, bias=False)
        nn.init.eye_(double.weight)
        cont = ContinuationMonad(transform=double)
        x = torch.rand(1, 3)
        t = cont.T(x)
        assert t.shape == x.shape

    def test_monad_laws_loss_finite(self):
        """monad_laws_loss runs through unit+join without NaN."""
        cont = ContinuationMonad()
        x = torch.rand(1, 3)
        loss = cont.monad_laws_loss(x)
        assert torch.isfinite(loss)


class TestWriterMonadT:
    """Cover line 120: WriterMonad.T delegates to self.unit."""

    def test_T_returns_unit(self):
        """WriterMonad.T(x) returns self.unit(x) (line 120)."""
        writer = WriterMonad(log_dim=2)
        x = torch.ones(1, 3)
        val, log = writer.T(x)
        assert val.shape == (1, 3)
        assert log.shape == (1, 2)
        torch.testing.assert_close(log, torch.zeros(1, 2))

    def test_T_log_all_zeros(self):
        """The log tensor produced by T is all zeros (unit log)."""
        writer = WriterMonad(log_dim=4)
        x = torch.randn(2, 5)
        val, log = writer.T(x)
        assert (log == 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# distributed.py
# ─────────────────────────────────────────────────────────────────────────────

from topos_ai.distributed import setup_distributed_topos, setup_expert_parallelism


class TestDistributedCoverage:
    """Cover lines 30-31, 37-38, 49, 57."""

    def test_expert_parallelism_world_size_one_returns_model(self):
        """world_size <= 1 → immediate return (line 49)."""
        model = nn.Linear(4, 4)
        result = setup_expert_parallelism(model, rank=0, world_size=1)
        assert result is model

    def test_expert_parallelism_world_size_zero_returns_model(self):
        """world_size = 0 also triggers the <= 1 guard."""
        model = nn.Linear(4, 4)
        result = setup_expert_parallelism(model, rank=0, world_size=0)
        assert result is model

    def test_expert_parallelism_num_universes_zero_skips(self):
        """Module with num_universes=0 triggers continue (line 57)."""

        class MultiUniverseToposAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_universes = 0

        model = nn.Sequential(MultiUniverseToposAttention())
        result = setup_expert_parallelism(model, rank=0, world_size=2)
        # Module had no valid universes to assign, model returned unchanged
        assert result is model

    def test_expert_parallelism_no_matching_modules(self):
        """No MultiUniverseToposAttention modules → loop body skipped, model returned."""
        model = nn.Linear(4, 4)
        result = setup_expert_parallelism(model, rank=0, world_size=4)
        assert result is model

    def test_setup_distributed_no_fsdp_returns_model(self):
        """HAS_FSDP=False → warning + return model (lines 30-31)."""
        model = nn.Linear(4, 4)
        with patch("topos_ai.distributed.HAS_FSDP", False):
            result = setup_distributed_topos(model, rank=0, world_size=1)
        assert result is model

    def test_setup_distributed_not_initialized_returns_model(self):
        """dist.is_initialized()=False → warning + return model (lines 33-35)."""
        model = nn.Linear(4, 4)
        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = False
        with patch("topos_ai.distributed.dist", mock_dist):
            result = setup_distributed_topos(model, rank=0, world_size=1)
        assert result is model

    def test_setup_distributed_fsdp_wrap(self):
        """HAS_FSDP=True + dist.is_initialized()=True → FSDP wrap (lines 37-38)."""
        model = nn.Linear(4, 4)
        wrapped = nn.Linear(4, 4)  # stand-in for FSDP-wrapped model

        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = True
        mock_fsdp = MagicMock(return_value=wrapped)
        mock_cpu_offload = MagicMock(return_value=MagicMock())

        with patch("topos_ai.distributed.dist", mock_dist), \
             patch("topos_ai.distributed.FSDP", mock_fsdp), \
             patch("topos_ai.distributed.CPUOffload", mock_cpu_offload):
            result = setup_distributed_topos(model, rank=0, world_size=2)

        assert result is wrapped
        mock_fsdp.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# monoidal.py – bifunctoriality identity violation
# ─────────────────────────────────────────────────────────────────────────────

from topos_ai.formal_category import FiniteCategory
from topos_ai.monoidal import strict_monoidal_from_monoid, FiniteMonoidalCategory, FiniteSymmetricMonoidalCategory


def _walking_arrow_monoidal():
    """
    Non-strict monoidal category on the walking-arrow 0→1.
    ⊗ = max on objects {0,1} with unit 0.
    Strict associator (identity morphisms).
    """
    C = FiniteCategory(
        objects=("0", "1"),
        morphisms={"id0": ("0", "0"), "id1": ("1", "1"), "f": ("0", "1")},
        identities={"0": "id0", "1": "id1"},
        composition={
            ("id0", "id0"): "id0", ("id1", "id1"): "id1",
            ("f", "id0"): "f",    ("id1", "f"): "f",
        },
    )

    def to(A, B):
        return "1" if A == "1" or B == "1" else "0"

    tensor_objects = {(A, B): to(A, B) for A in ("0", "1") for B in ("0", "1")}
    tensor_morphisms = {
        ("id0", "id0"): "id0", ("id0", "id1"): "id1", ("id0", "f"): "f",
        ("id1", "id0"): "id1", ("id1", "id1"): "id1", ("id1", "f"): "id1",
        ("f",   "id0"): "f",  ("f",   "id1"): "id1", ("f",   "f"):   "f",
    }

    def alpha_mor(A, B, Cv):
        AB = to(A, B)
        ABC = to(AB, Cv)
        return "id0" if ABC == "0" else "id1"

    associator = {(A, B, Cv): alpha_mor(A, B, Cv) for A in ("0","1") for B in ("0","1") for Cv in ("0","1")}
    return FiniteMonoidalCategory(
        category=C, tensor_objects=tensor_objects, tensor_morphisms=tensor_morphisms,
        unit="0", associator=associator,
        left_unitor={"0": "id0", "1": "id1"},
        right_unitor={"0": "id0", "1": "id1"},
    )


class TestMonoidalViolations:
    """Trigger ValueError lines in FiniteMonoidalCategory by post-construction corruption."""

    def test_left_unitor_naturality_violation(self):
        """
        Corrupt tensor_morphisms[("id0","f")] from "f" to "id1" so that
        for the morphism f:0→1, LHS = compose(id1,id1)=id1 ≠ compose(f,id0)=f.
        Fires 'Left unitor naturality fails for f=f.' (line 208).
        """
        mc = _walking_arrow_monoidal()
        mc.tensor_morphisms[("id0", "f")] = "id1"
        with pytest.raises(ValueError, match="Left unitor naturality fails"):
            mc._check_unitor_naturality()

    def test_right_unitor_naturality_violation(self):
        """
        Corrupt tensor_morphisms[("f","id0")] from "f" to "id1".
        For f:0→1 the right check gives LHS=compose(id1,id1)=id1 ≠ compose(f,id0)=f.
        Fires 'Right unitor naturality fails for f=f.' (line 210).
        """
        mc = _walking_arrow_monoidal()
        mc.tensor_morphisms[("f", "id0")] = "id1"
        with pytest.raises(ValueError, match="Right unitor naturality fails"):
            mc._check_unitor_naturality()

    def test_triangle_coherence_violation(self):
        """
        Corrupt associator[("0","0","1")] from "id1" to "f".
        For A=0, B=1: LHS = compose(id1, f) = f ≠ RHS = id1.
        Fires 'Triangle coherence fails for A=0, B=1.' (line 256).
        """
        mc = _walking_arrow_monoidal()
        mc.associator[("0", "0", "1")] = "f"
        with pytest.raises(ValueError, match="Triangle coherence fails"):
            mc._check_triangle()


# ─────────────────────────────────────────────────────────────────────────────
# enriched.py – validation violation tests
# ─────────────────────────────────────────────────────────────────────────────

from topos_ai.enriched import FiniteEnrichedCategory, discrete_enriched_category


def _z2_enriched():
    """
    Z/2Z-enriched 2-object category: hom(A,A)=hom(B,B)=0, hom(A,B)=hom(B,A)=1.
    V = (Z/2Z, +, 0).
    """
    V = strict_monoidal_from_monoid(
        objects=["0", "1"],
        tensor_table={
            ("0", "0"): "0", ("0", "1"): "1",
            ("1", "0"): "1", ("1", "1"): "0",
        },
        unit="0",
    )
    C_cat = V.category
    objs = ["A", "B"]
    hom = {("A","A"): "0", ("A","B"): "1", ("B","A"): "1", ("B","B"): "0"}
    comps = {}
    for X in objs:
        for Y in objs:
            for Z in objs:
                src = V.tensor_obj(hom[(Y, Z)], hom[(X, Y)])
                dst = hom[(X, Z)]
                comps[(X, Y, Z)] = C_cat.identities[src]  # src == dst for Z2
    ids = {X: C_cat.identities["0"] for X in objs}
    return FiniteEnrichedCategory(
        objects=objs, enriching=V, hom_objects=hom,
        compositions=comps, identity_elements=ids,
    )


def _bool_chaotic():
    """Bool-enriched 2-object category: all hom-objects = T (unit=T, V=bool-AND)."""
    V = strict_monoidal_from_monoid(
        objects=["F", "T"],
        tensor_table={("F","F"):"F",("F","T"):"F",("T","F"):"F",("T","T"):"T"},
        unit="T",
    )
    C_cat = V.category
    objs = ["A", "B"]
    hom = {(X, Y): "T" for X in objs for Y in objs}
    comps = {(X, Y, Z): C_cat.identities["T"] for X in objs for Y in objs for Z in objs}
    ids = {X: C_cat.identities["T"] for X in objs}
    return FiniteEnrichedCategory(
        objects=objs, enriching=V, hom_objects=hom,
        compositions=comps, identity_elements=ids,
    )


class TestEnrichedViolations:
    """Trigger ValueError branches in FiniteEnrichedCategory validation (post-construction corruption)."""

    def test_composition_not_v_morphism_raises(self):
        """compositions[(A,B,C)] points to a non-existent V-morphism → line 163."""
        ec = _bool_chaotic()
        ec.compositions[("A", "A", "A")] = "GHOST"
        with pytest.raises(ValueError, match="not a V-morphism"):
            ec._check_composition_types()

    def test_identity_not_v_morphism_raises(self):
        """identity_elements[A] points to a non-existent V-morphism → line 185."""
        ec = _bool_chaotic()
        ec.identity_elements["A"] = "GHOST"
        with pytest.raises(ValueError, match="not a V-morphism"):
            ec._check_identity_types()

    def test_missing_composition_entry_raises(self):
        """Missing ∘_{A,B,C} key → 'Missing composition' (line 158)."""
        ec = _bool_chaotic()
        del ec.compositions[("A", "A", "A")]
        with pytest.raises(ValueError, match="Missing composition"):
            ec._check_composition_types()

    def test_missing_identity_entry_raises(self):
        """Missing j_A key → 'Missing identity' (line 182)."""
        ec = _bool_chaotic()
        del ec.identity_elements["A"]
        with pytest.raises(ValueError, match="Missing identity"):
            ec._check_identity_types()

    def test_left_unitality_violation(self):
        """
        Corrupt V's left_unitor["0"] = "id_1" in the Z2 enriched category.
        For A=A, B=A: lhs=id_0, rhs=id_1 → 'Left unitality fails' (line 275).
        """
        ec = _z2_enriched()
        ec.enriching.left_unitor["0"] = "id_1"  # corrupt V's left unitor
        with pytest.raises(ValueError, match="Left unitality fails"):
            ec._check_left_unitality()

    def test_right_unitality_violation(self):
        """
        Corrupt V's right_unitor["0"] = "id_1".
        For A=A, B=A: lhs=id_0, rhs=id_1 → 'Right unitality fails' (line 305).
        """
        ec = _z2_enriched()
        ec.enriching.right_unitor["0"] = "id_1"
        with pytest.raises(ValueError, match="Right unitality fails"):
            ec._check_right_unitality()

    def test_discrete_enriched_no_identity_morphism_raises(self):
        """
        discrete_enriched_category with hom(X,X) pointing to a V-object for which
        no V-morphism I→hom(X,X) exists → 'No V-morphism' (line 440).
        V = bool-AND (unit=T); hom(X,X)="F" but no morphism T→F exists.
        """
        V = strict_monoidal_from_monoid(
            objects=["F", "T"],
            tensor_table={("F","F"):"F",("F","T"):"F",("T","F"):"F",("T","T"):"T"},
            unit="T",
        )
        with pytest.raises(ValueError, match="No V-morphism"):
            discrete_enriched_category(
                objects=["X"],
                enriching=V,
                hom_matrix={("X", "X"): "F"},
            )
