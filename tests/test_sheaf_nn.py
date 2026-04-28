"""Tests for topos_ai.sheaf_nn — Sheaf Neural Networks."""

import pytest
import torch
import torch.nn as nn

from topos_ai.sheaf_nn import (
    build_coboundary,
    sheaf_laplacian_from_coboundary,
    SheafLaplacian,
    SheafConv,
    SheafNet,
    sheaf_laplacian_from_presheaf,
)
from topos_ai.formal_category import FiniteCategory, Presheaf


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _line_graph():
    """
    Path graph: 0 — 1 — 2  (two edges).
    vertices = [0, 1, 2], edges = [(0,1), (1,2)]
    """
    return 3, [(0, 1), (1, 2)]


def _triangle_graph():
    """
    Triangle: 0—1—2—0  (three edges).
    """
    return 3, [(0, 1), (1, 2), (2, 0)]


def _arrow_presheaf():
    """
    Presheaf over the walking-arrow category 0 → 1.
    F(0) = {a, b},  F(1) = {x},  F(f) maps everything to x.
    """
    C = FiniteCategory(
        objects=("0", "1"),
        morphisms={"id0": ("0", "0"), "id1": ("1", "1"), "f": ("0", "1")},
        identities={"0": "id0", "1": "id1"},
        composition={
            ("id0", "id0"): "id0",
            ("id1", "id1"): "id1",
            ("f", "id0"): "f",
            ("id1", "f"): "f",
        },
    )
    sets = {"0": frozenset({"a", "b"}), "1": frozenset({"x"})}
    restrictions = {
        "id0": {"a": "a", "b": "b"},
        "id1": {"x": "x"},
        "f": {"x": "a"},  # contravariant: F(f) : F(1) → F(0), x ↦ a
    }
    return Presheaf(category=C, sets=sets, restrictions=restrictions), C


# ------------------------------------------------------------------ #
# Tests: build_coboundary                                              #
# ------------------------------------------------------------------ #

class TestBuildCoboundary:
    def test_returns_tensor(self):
        verts = ["u", "v"]
        edges = [("u", "v")]
        rmaps = {("u", 0): torch.eye(1), ("v", 0): torch.eye(1)}
        B = build_coboundary(verts, edges, rmaps, 1, 1)
        assert isinstance(B, torch.Tensor)

    def test_shape(self):
        """B should have shape (n_e·d_e, n_v·d_v)."""
        verts = ["u", "v", "w"]
        edges = [("u", "v"), ("v", "w")]
        rmaps = {}
        B = build_coboundary(verts, edges, rmaps, 1, 1)
        assert B.shape == (2, 3)  # 2 edges, 3 vertices, stalk_dim=1

    def test_single_edge_values(self):
        """For identity restriction maps, B should have ±1 entries."""
        verts = ["u", "v"]
        edges = [("u", "v")]
        rmaps = {("u", 0): torch.eye(1), ("v", 0): torch.eye(1)}
        B = build_coboundary(verts, edges, rmaps, 1, 1)
        # Edge 0: row 0 gets +1 at column v (idx 1), -1 at column u (idx 0)
        assert B[0, 1].item() == pytest.approx(1.0)
        assert B[0, 0].item() == pytest.approx(-1.0)

    def test_larger_stalk_dim(self):
        verts = ["u", "v"]
        edges = [("u", "v")]
        rmaps = {("u", 0): torch.eye(2), ("v", 0): torch.eye(2)}
        B = build_coboundary(verts, edges, rmaps, stalk_dim_v=2, stalk_dim_e=2)
        assert B.shape == (2, 4)


# ------------------------------------------------------------------ #
# Tests: sheaf_laplacian_from_coboundary                               #
# ------------------------------------------------------------------ #

class TestSheafLaplacianFromCoboundary:
    def test_psd(self):
        """L = B^T B is positive semidefinite: all eigenvalues ≥ 0."""
        verts = ["u", "v", "w"]
        edges = [("u", "v"), ("v", "w")]
        rmaps = {}
        B = build_coboundary(verts, edges, rmaps, 1, 1)
        L = sheaf_laplacian_from_coboundary(B)
        eigvals = torch.linalg.eigvalsh(L)
        assert (eigvals >= -1e-6).all()

    def test_symmetric(self):
        verts = ["u", "v"]
        edges = [("u", "v")]
        rmaps = {}
        B = build_coboundary(verts, edges, rmaps, 1, 1)
        L = sheaf_laplacian_from_coboundary(B)
        assert torch.allclose(L, L.T, atol=1e-6)

    def test_shape(self):
        verts = ["u", "v", "w"]
        edges = [("u", "v"), ("v", "w")]
        rmaps = {}
        B = build_coboundary(verts, edges, rmaps, 1, 1)
        L = sheaf_laplacian_from_coboundary(B)
        assert L.shape == (3, 3)


# ------------------------------------------------------------------ #
# Tests: SheafLaplacian module                                         #
# ------------------------------------------------------------------ #

class TestSheafLaplacianModule:
    def test_construct(self):
        n_v, edges = _line_graph()
        layer = SheafLaplacian(n_v, edges, stalk_dim_v=1, stalk_dim_e=1)
        assert isinstance(layer, nn.Module)

    def test_laplacian_shape(self):
        n_v, edges = _line_graph()
        d_v = 2
        layer = SheafLaplacian(n_v, edges, stalk_dim_v=d_v, stalk_dim_e=d_v)
        L = layer.laplacian()
        assert L.shape == (n_v * d_v, n_v * d_v)

    def test_laplacian_psd(self):
        n_v, edges = _line_graph()
        layer = SheafLaplacian(n_v, edges, stalk_dim_v=1, stalk_dim_e=1)
        L = layer.laplacian()
        eigvals = torch.linalg.eigvalsh(L)
        assert (eigvals >= -1e-5).all()

    def test_forward_shape_2d(self):
        n_v, edges = _line_graph()
        d_v = 3
        layer = SheafLaplacian(n_v, edges, stalk_dim_v=d_v, stalk_dim_e=d_v)
        x = torch.randn(n_v, d_v)
        y = layer(x)
        assert y.shape == x.shape

    def test_forward_shape_3d(self):
        n_v, edges = _line_graph()
        d_v = 4
        layer = SheafLaplacian(n_v, edges, stalk_dim_v=d_v, stalk_dim_e=d_v)
        x = torch.randn(8, n_v, d_v)
        y = layer(x)
        assert y.shape == x.shape

    def test_forward_differentiable(self):
        n_v, edges = _line_graph()
        layer = SheafLaplacian(n_v, edges, stalk_dim_v=2, stalk_dim_e=2)
        x = torch.randn(n_v, 2, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None

    def test_has_learnable_params(self):
        n_v, edges = _line_graph()
        layer = SheafLaplacian(n_v, edges, stalk_dim_v=2, stalk_dim_e=2)
        params = list(layer.parameters())
        assert len(params) > 0

    def test_alpha_zero_is_identity(self):
        """With alpha=0, diffusion = identity: output should equal input."""
        n_v, edges = _line_graph()
        layer = SheafLaplacian(n_v, edges, stalk_dim_v=1, stalk_dim_e=1, alpha=0.0)
        x = torch.randn(n_v, 1)
        y = layer(x)
        assert torch.allclose(y, x, atol=1e-6)

    def test_triangle_graph(self):
        n_v, edges = _triangle_graph()
        layer = SheafLaplacian(n_v, edges, stalk_dim_v=2, stalk_dim_e=2)
        x = torch.randn(4, n_v, 2)
        y = layer(x)
        assert y.shape == x.shape


# ------------------------------------------------------------------ #
# Tests: SheafConv module                                              #
# ------------------------------------------------------------------ #

class TestSheafConv:
    def test_construct(self):
        n_v, edges = _line_graph()
        layer = SheafConv(n_v, edges, in_channels=4, out_channels=8)
        assert isinstance(layer, nn.Module)

    def test_forward_2d_shape(self):
        n_v, edges = _line_graph()
        layer = SheafConv(n_v, edges, in_channels=4, out_channels=8)
        x = torch.randn(n_v, 4)
        y = layer(x)
        assert y.shape == (n_v, 8)

    def test_forward_3d_shape(self):
        n_v, edges = _line_graph()
        layer = SheafConv(n_v, edges, in_channels=4, out_channels=8)
        x = torch.randn(16, n_v, 4)
        y = layer(x)
        assert y.shape == (16, n_v, 8)

    def test_backward_pass(self):
        n_v, edges = _line_graph()
        layer = SheafConv(n_v, edges, in_channels=3, out_channels=5)
        x = torch.randn(n_v, 3)
        y = layer(x)
        y.sum().backward()
        for p in layer.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_in_out_channels_same(self):
        n_v, edges = _triangle_graph()
        layer = SheafConv(n_v, edges, in_channels=6, out_channels=6)
        x = torch.randn(n_v, 6)
        y = layer(x)
        assert y.shape == (n_v, 6)

    def test_custom_stalk_dim_e(self):
        n_v, edges = _line_graph()
        layer = SheafConv(n_v, edges, in_channels=4, out_channels=4, stalk_dim_e=2)
        x = torch.randn(n_v, 4)
        y = layer(x)
        assert y.shape == (n_v, 4)


# ------------------------------------------------------------------ #
# Tests: SheafNet                                                      #
# ------------------------------------------------------------------ #

class TestSheafNet:
    def test_construct(self):
        n_v, edges = _line_graph()
        net = SheafNet(n_v, edges, channels=[4, 8, 2])
        assert isinstance(net, nn.Module)

    def test_too_few_channels_raises(self):
        n_v, edges = _line_graph()
        with pytest.raises(ValueError):
            SheafNet(n_v, edges, channels=[4])

    def test_forward_shape(self):
        n_v, edges = _line_graph()
        net = SheafNet(n_v, edges, channels=[4, 8, 2])
        x = torch.randn(n_v, 4)
        y = net(x)
        assert y.shape == (n_v, 2)

    def test_batched_forward(self):
        n_v, edges = _line_graph()
        net = SheafNet(n_v, edges, channels=[3, 6, 1])
        x = torch.randn(10, n_v, 3)
        y = net(x)
        assert y.shape == (10, n_v, 1)

    def test_gradient_flows(self):
        n_v, edges = _line_graph()
        net = SheafNet(n_v, edges, channels=[2, 4, 2])
        x = torch.randn(n_v, 2)
        y = net(x)
        y.sum().backward()
        for p in net.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_single_layer(self):
        n_v, edges = _line_graph()
        net = SheafNet(n_v, edges, channels=[3, 5])
        x = torch.randn(n_v, 3)
        y = net(x)
        assert y.shape == (n_v, 5)

    def test_deep_net(self):
        n_v, edges = _triangle_graph()
        net = SheafNet(n_v, edges, channels=[4, 8, 8, 16, 4])
        x = torch.randn(n_v, 4)
        y = net(x)
        assert y.shape == (n_v, 4)


# ------------------------------------------------------------------ #
# Tests: sheaf_laplacian_from_presheaf                                 #
# ------------------------------------------------------------------ #

class TestSheafLaplacianFromPresheaf:
    def test_returns_sheaf_laplacian(self):
        P, C = _arrow_presheaf()
        layer = sheaf_laplacian_from_presheaf(
            presheaf=P,
            vertex_objects=["0", "1"],
            edge_morphisms=["f"],
            stalk_dim=2,
        )
        assert isinstance(layer, SheafLaplacian)

    def test_parameters_frozen(self):
        P, C = _arrow_presheaf()
        layer = sheaf_laplacian_from_presheaf(
            presheaf=P,
            vertex_objects=["0", "1"],
            edge_morphisms=["f"],
            stalk_dim=2,
        )
        for param in layer.parameters():
            assert not param.requires_grad

    def test_forward_shape(self):
        P, C = _arrow_presheaf()
        layer = sheaf_laplacian_from_presheaf(
            presheaf=P,
            vertex_objects=["0", "1"],
            edge_morphisms=["f"],
            stalk_dim=2,
        )
        x = torch.randn(2, 2)  # 2 vertices, stalk_dim=2
        y = layer(x)
        assert y.shape == x.shape

    def test_laplacian_psd(self):
        P, C = _arrow_presheaf()
        layer = sheaf_laplacian_from_presheaf(
            presheaf=P,
            vertex_objects=["0", "1"],
            edge_morphisms=["f"],
            stalk_dim=2,
        )
        L = layer.laplacian()
        eigvals = torch.linalg.eigvalsh(L)
        assert (eigvals >= -1e-5).all()
