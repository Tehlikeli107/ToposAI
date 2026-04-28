"""
Coverage tests for:
  - topos_ai.tokenization  (early-break paths, merge_count print)
  - topos_ai.nn            (TopologicalLinear bias=True, kv_cache path)
  - topos_ai.math          (transitive_closure unknown composition)
  - topos_ai.lean4_export  (_lean_ids, export_to_file)
  - topos_ai.models        (mask=None for seq_len<=1)
  - topos_ai.polynomial_functors (WiringDiagram wire_map path)
  - topos_ai.__init__      (_has_dependency ImportError → False)
  - topos_ai.elementary_topos (subobject_classifier)
  - topos_ai.adjunction    (verify_hom_bijection line 356)
  - topos_ai.hott          (PathFamily invalid-transport, FormalHomotopyEquivalence)
"""
import os
import tempfile
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# topos_ai.tokenization
# ---------------------------------------------------------------------------
from topos_ai.tokenization import TopologicalTokenizer


class TestTokenizerBreakPaths:
    def test_no_morphisms_break_line50(self):
        """Line 50: empty morphisms dict → break."""
        tok = TopologicalTokenizer(vocab_size=10)
        tok.train("a")  # single char → no pairs → break immediately

    def test_low_strength_break_line55(self):
        """Line 55: best_strength < 0.01 → break."""
        tok = TopologicalTokenizer(vocab_size=10)
        with patch.object(
            tok, "_compute_topological_morphisms",
            return_value={("a", "b"): 0.005},
        ):
            tok.train("ab")

    def test_token_already_in_vocab_break_line59(self):
        """Line 59: new_token_str already in vocab → break."""
        tok = TopologicalTokenizer(vocab_size=10)
        # First call: merge ("a","b")→"ab" (adds to vocab)
        # Second call: same pair → "ab" already in vocab → break
        with patch.object(
            tok, "_compute_topological_morphisms",
            side_effect=[{("a", "b"): 1.0}, {("a", "b"): 1.0}],
        ):
            tok.train("ab")

    def test_merge_count_print_line80(self, capsys):
        """Line 80: every 100 merges prints progress."""
        tok = TopologicalTokenizer(vocab_size=200)
        _calls = [0]

        def mock_morph(tokens):
            n = _calls[0]
            _calls[0] += 1
            if n >= 105:
                return {}
            return {(f"X{n}a", f"X{n}b"): 1.0}

        with patch.object(tok, "_compute_topological_morphisms", side_effect=mock_morph):
            tok.train("abcde")

        out = capsys.readouterr().out
        assert "> 100 merges" in out


# ---------------------------------------------------------------------------
# topos_ai.nn
# ---------------------------------------------------------------------------
from topos_ai.nn import TopologicalLinear, MultiUniverseToposAttention


class TestTopologicalLinearBias:
    def test_bias_true_line63(self):
        """Line 63: bias=True → bias_raw parameter is created."""
        layer = TopologicalLinear(4, 8, bias=True)
        assert layer.bias_raw is not None

    def test_forward_with_bias_lines74_75(self):
        """Lines 74-75: bias_raw not None → B_topos applied in forward."""
        layer = TopologicalLinear(4, 8, bias=True)
        x = torch.randn(2, 4)
        out = layer(x)
        assert out.shape == (2, 8)


class TestMultiUniverseAttentionKVCache:
    def test_kv_cache_path_lines245_248(self):
        """Lines 245-248: kv_cache not None → cat with cached K,V."""
        from topos_ai.nn import precompute_freqs_cis
        d_model, num_universes = 8, 2
        attn = MultiUniverseToposAttention(d_model, num_universes, top_k=1)
        B, seq, past = 1, 2, 3
        x = torch.randn(B, seq, d_model)
        d_u = d_model // num_universes
        k_cache = torch.zeros(B, past, num_universes, d_u)
        v_cache = torch.zeros(B, past, num_universes, d_u)
        # freqs_cis: complex tensor of shape [seq, d_u//2]
        freqs_cis = precompute_freqs_cis(d_u, seq + past)[past: past + seq]
        # kv_cache not None → K_all = cat([k_cache, K_new], dim=1)
        out, new_kv = attn(x, freqs_cis, kv_cache=(k_cache, v_cache))
        assert out.shape[0] == B


# ---------------------------------------------------------------------------
# topos_ai.math
# ---------------------------------------------------------------------------
from topos_ai.math import transitive_closure


class TestTransitiveClosure:
    def test_unknown_composition_raises_lines75_78(self):
        """Lines 75-78: unknown composition mode → ValueError."""
        R = torch.eye(3)
        with pytest.raises(ValueError, match="lukasiewicz"):
            transitive_closure(R, composition="invalid_mode")

    def test_lukasiewicz_composition(self):
        """Cover the lukasiewicz branch."""
        R = torch.rand(3, 3)
        result = transitive_closure(R, composition="lukasiewicz")
        assert result.shape == (3, 3)


# ---------------------------------------------------------------------------
# topos_ai.lean4_export
# ---------------------------------------------------------------------------
from topos_ai.lean4_export import _lean_ids, export_to_file


class TestLean4Export:
    def test_lean_ids_line33(self):
        """Line 33: _lean_ids([...]) → list of clean identifiers."""
        result = _lean_ids(["hello", "world", "123"])
        assert result == ["hello", "world", "n123"]

    def test_export_to_file_lines357_358(self):
        """Lines 357-358: export_to_file writes code to path."""
        code = "-- generated lean code\n"
        with tempfile.NamedTemporaryFile(mode="r", suffix=".lean", delete=False) as f:
            path = f.name
        try:
            export_to_file(code, path)
            with open(path, encoding="utf-8") as f:
                assert f.read() == code
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# topos_ai.models
# ---------------------------------------------------------------------------
from topos_ai.models import ToposTransformer


class TestToposTransformerMask:
    def test_mask_none_when_seq_len_one_line74(self):
        """Line 74: SeqLen == 1 → mask = None."""
        model = ToposTransformer(vocab_size=32, d_model=8, num_universes=2, num_layers=1, max_seq_len=16)
        idx = torch.zeros(1, 1, dtype=torch.long)  # seq_len=1
        with torch.no_grad():
            logits, _ = model(idx)
        assert logits.shape[-1] == 32


# ---------------------------------------------------------------------------
# topos_ai.polynomial_functors
# ---------------------------------------------------------------------------
from topos_ai.polynomial_functors import DynamicalSystem, WiringDiagram


class TestWiringDiagramWireMap:
    def test_wire_map_path_lines251_252(self):
        """Lines 251-252: wire_map not None → src_box redirect."""
        # Two DynamicalSystems: output_dim of first = input_dim of second
        s1 = DynamicalSystem(state_dim=4, input_dim=3, output_dim=4)
        s2 = DynamicalSystem(state_dim=4, input_dim=4, output_dim=4)
        # wire_map: s2 takes output of s1 (box 0) instead of s1's output naturally
        wd = WiringDiagram([s1, s2], wire_map=[(0, 0)])
        B = 2
        state1 = torch.zeros(B, 4)
        state2 = torch.zeros(B, 4)
        inp = torch.randn(B, 3)
        outputs, new_states = wd([state1, state2], inp)
        assert len(outputs) == 2
        assert len(new_states) == 2


# ---------------------------------------------------------------------------
# topos_ai.__init__  — _has_dependency
# ---------------------------------------------------------------------------


class TestHasDependency:
    def test_nonexistent_module_returns_false_lines140_141(self):
        """Lines 140-141: ImportError → returns False."""
        from importlib import import_module as _orig_import
        import topos_ai as _tai

        # _has_dependency is a private function but we can test via indirect import
        # Alternatively test the helper directly if accessible
        try:
            result = _tai._has_dependency("_nonexistent_toposai_module_xyz_9999")
        except AttributeError:
            # If not exported, exercise via patch
            with patch("importlib.import_module", side_effect=ImportError):
                result = _tai._has_dependency("anything")
        assert result is False


# ---------------------------------------------------------------------------
# topos_ai.elementary_topos
# ---------------------------------------------------------------------------
from topos_ai.elementary_topos import ElementaryTopos


class TestElementaryToposSubobjectClassifier:
    def test_subobject_classifier_line43(self):
        """Line 43: subobject_classifier(X, Y) → exponential(X, Y)."""
        et = ElementaryTopos(dim=4)
        X = torch.tensor([0.3, 0.7])
        Y = torch.tensor([0.8, 0.5])
        result = et.subobject_classifier(X, Y)
        assert result.shape == X.shape


# ---------------------------------------------------------------------------
# topos_ai.adjunction — verify_hom_bijection line 356
# ---------------------------------------------------------------------------
from topos_ai.formal_category import FiniteCategory
from topos_ai.adjunction import FiniteAdjunction


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
    return FiniteAdjunction(
        C=_C(), D=_D(),
        F_obj={"0": "A", "1": "A"},
        F_mor={"id_0": "id_A", "id_1": "id_A", "f": "id_A"},
        G_obj={"A": "1", "B": "1"},
        G_mor={"id_A": "id_1", "id_B": "id_1", "g": "id_1"},
        unit={"0": "f", "1": "id_1"},
        counit={"A": "id_A", "B": "g"},
    )


class TestVerifyHomBijection:
    def test_phi_psi_not_inverse_returns_false_line356(self):
        """Line 356: phi(psi(psi_mor, c), d) != psi_mor → return False."""
        adj = _adjunction()
        real_phi = adj.phi

        _call_count = [0]

        def phi_patched(phi_mor, d):
            _call_count[0] += 1
            if _call_count[0] == 1:
                return real_phi(phi_mor, d)  # correct for check-1
            return "WRONG_VALUE"             # wrong for check-2

        with patch.object(adj, "phi", side_effect=phi_patched):
            result = adj.verify_hom_bijection()
        assert result is False


# ---------------------------------------------------------------------------
# topos_ai.hott — PathFamily invalid transports, FormalHomotopyEquivalence
# ---------------------------------------------------------------------------
from topos_ai.hott import FinitePathGroupoid, PathFamily, FormalHomotopyEquivalence


def _two_point_groupoid():
    """Groupoid: A ←p_inv— B, refl on both, p and p_inv compose to refl."""
    return FinitePathGroupoid(
        objects=["A", "B"],
        paths={
            "refl_A": ("A","A"), "refl_B": ("B","B"),
            "p": ("A","B"), "p_inv": ("B","A"),
        },
        identities={"A": "refl_A", "B": "refl_B"},
        inverses={
            "refl_A": "refl_A", "refl_B": "refl_B",
            "p": "p_inv", "p_inv": "p",
        },
        composition={
            ("refl_A","refl_A"): "refl_A",
            ("refl_B","refl_B"): "refl_B",
            ("p","refl_A"): "p",
            ("refl_B","p"): "p",
            ("p_inv","refl_B"): "p_inv",
            ("refl_A","p_inv"): "p_inv",
            ("p_inv","p"): "refl_A",
            ("p","p_inv"): "refl_B",
        },
    )


def _valid_path_family(base):
    """Valid PathFamily with bijective transports {v1↦u1, v2↦u2}."""
    return PathFamily(
        base=base,
        fibers={"A": ["v1","v2"], "B": ["u1","u2"]},
        transports={
            "refl_A": {"v1":"v1","v2":"v2"},
            "refl_B": {"u1":"u1","u2":"u2"},
            "p":     {"v1":"u1","v2":"u2"},
            "p_inv": {"u1":"v1","u2":"v2"},
        },
    )


class TestPathFamilyTransportEquivalences:
    @staticmethod
    def _first_transport_equivalence_guard(pf):
        """
        Return which guard fails first in validate_transport_equivalences.
        Mirrors guard order to make tests assert the exact failure point.
        """
        for path in pf.base.paths:
            forward, backward = pf.transport_equivalence(path)
            src = pf.base.source(path)
            dst = pf.base.target(path)

            if set(forward) != set(pf.fibers[src]) or set(backward) != set(pf.fibers[dst]):
                return "domain_mismatch"
            if set(forward.values()) != set(pf.fibers[dst]):
                return "forward_not_surjective"
            if set(backward.values()) != set(pf.fibers[src]):
                return "backward_not_surjective"
            for value in pf.fibers[src]:
                if backward[forward[value]] != value:
                    return "backward_after_forward_not_identity"
            for value in pf.fibers[dst]:
                if forward[backward[value]] != value:
                    return "forward_after_backward_not_identity"
        return "ok"

    def test_forward_not_surjective_returns_false_line178(self):
        """Line 178: forward range ≠ fibers[dst] → return False."""
        base = _two_point_groupoid()
        pf = _valid_path_family(base)
        # Mock validate_functorial_transport to do nothing, then corrupt transport
        with patch.object(pf, "validate_functorial_transport", return_value=True):
            pf.transports["p"] = {"v1":"u1","v2":"u1"}      # non-surjective
            pf.transports["p_inv"] = {"u1":"v1","u2":"v1"}  # non-surjective inverse
            result = pf.validate_transport_equivalences()
        assert result is False

    def test_backward_not_surjective_returns_false_line180(self):
        """Line 180: backward range ≠ fibers[src] → return False."""
        base = _two_point_groupoid()
        pf = _valid_path_family(base)
        with patch.object(pf, "validate_functorial_transport", return_value=True):
            # forward maps correctly but backward is non-surjective
            pf.transports["p"] = {"v1":"u1","v2":"u2"}      # surjective forward
            pf.transports["p_inv"] = {"u1":"v1","u2":"v1"}  # non-surjective backward
            result = pf.validate_transport_equivalences()
        assert result is False

    def test_backward_forward_not_identity_returns_false_line184(self):
        """Line 184: backward(forward(v)) ≠ v → return False."""
        base = _two_point_groupoid()
        pf = _valid_path_family(base)
        with patch.object(pf, "validate_functorial_transport", return_value=True):
            # forward: v1→u1, v2→u2 (ok); backward: u1→v2, u2→v1 (swapped)
            pf.transports["p"] = {"v1":"u1","v2":"u2"}
            pf.transports["p_inv"] = {"u1":"v2","u2":"v1"}
            result = pf.validate_transport_equivalences()
        # backward(forward(v1)) = backward(u1) = v2 ≠ v1 → False
        assert result is False

    def test_line187_unreachable_under_current_invariants(self):
        """
        Under current guard order/invariants, line 187 is unreachable.
        If line 184 passes, forward is injective; with line 178 this makes forward bijective.
        Then any backward satisfying line 184 must be forward inverse on dst too.
        So a crafted non-identity on dst is caught by an earlier guard.
        """
        base = _two_point_groupoid()
        pf = _valid_path_family(base)
        with patch.object(pf, "validate_functorial_transport", return_value=True):
            # Candidate that *tries* to break only forward(backward(.)) on dst.
            pf.fibers["A"] = frozenset(["v1", "v2", "v3"])
            pf.fibers["B"] = frozenset(["u1", "u2"])
            pf.transports["p"] = {"v1": "u1", "v2": "u2", "v3": "u1"}
            pf.transports["p_inv"] = {"u1": "v1", "u2": "v2"}
            pf.transports["refl_A"] = {"v1": "v1", "v2": "v2", "v3": "v3"}

            guard = self._first_transport_equivalence_guard(pf)
            result = pf.validate_transport_equivalences()

        assert guard == "backward_not_surjective"
        assert result is False


class TestFormalHomotopyEquivalence:
    def _arrow_cat(self, name_prefix=""):
        """A→B category for isomorphism tests."""
        p = name_prefix
        return FiniteCategory(
            objects=[f"{p}A", f"{p}B"],
            morphisms={
                f"{p}idA": (f"{p}A", f"{p}A"),
                f"{p}idB": (f"{p}B", f"{p}B"),
                f"{p}f":   (f"{p}A", f"{p}B"),
            },
            identities={f"{p}A": f"{p}idA", f"{p}B": f"{p}idB"},
            composition={
                (f"{p}idA",f"{p}idA"): f"{p}idA",
                (f"{p}idB",f"{p}idB"): f"{p}idB",
                (f"{p}f",f"{p}idA"):   f"{p}f",
                (f"{p}idB",f"{p}f"):   f"{p}f",
            },
        )

    def test_brute_force_search_triggers_functorial_fail_lines259_260_263(self):
        """Lines 259-260, 263: first morphism permutation fails functoriality → continue."""
        # cat_A: morphisms in standard order [idA, idB, f]
        cat_A = self._arrow_cat("A_")
        # cat_B: morphisms inserted as [f, idB, idA] — so first permutation
        # maps A_idA→B_f which fails functoriality (different src/dst endpoints)
        cat_B = FiniteCategory(
            objects=["B_A", "B_B"],
            morphisms={
                "B_f":   ("B_A","B_B"),  # inserted first so it appears first in keys()
                "B_idB": ("B_B","B_B"),
                "B_idA": ("B_A","B_A"),
            },
            identities={"B_A": "B_idA", "B_B": "B_idB"},
            composition={
                ("B_idA","B_idA"): "B_idA", ("B_idB","B_idB"): "B_idB",
                ("B_f","B_idA"): "B_f", ("B_idB","B_f"): "B_f",
            },
        )
        equiv = FormalHomotopyEquivalence(cat_A, cat_B)
        result = equiv.find_strict_isomorphism()
        # Eventually finds the isomorphism; lines 259-260,263 hit on bad permutations first
        assert result is not None

    def test_no_compose_method_uses_composition_dict_line242(self):
        """Line 242: cat without .compose → uses .composition.get((f,g))."""
        # Use FinitePathGroupoid which has .compose → goes to line 240-241
        # For line 242, need cat WITHOUT .compose attribute
        # Create a mock cat
        cat_mock = MagicMock(spec=[])  # no attributes by default
        cat_mock.paths = {"idX": ("X","X")}
        cat_mock.composition = {("idX","idX"): "idX"}
        cat_mock.objects = ["X"]

        # cat_mock has no .compose attribute since spec=[]
        # get_comp(cat_mock, "idX", "idX") → uses cat_mock.composition.get
        # We need two cats for FormalHomotopyEquivalence
        class SimpleCat:
            objects = ["X"]
            paths = {"idX": ("X","X")}
            morphisms = {"idX": ("X","X")}
            composition = {("idX","idX"): "idX"}
            # NO compose method

        cat_simple = SimpleCat()
        equiv = FormalHomotopyEquivalence(cat_simple, cat_simple)
        result = equiv.find_strict_isomorphism()
        # Both cats are identical 1-object 1-morphism: isomorphic
        assert result is not None
