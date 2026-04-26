"""
Sheaf Neural Networks — differentiable message-passing grounded in sheaf theory.

A *cellular sheaf* on a graph G = (V, E) assigns:
  - a stalk  F(v) ∈ ℝ^{d_v}   to every vertex  v ∈ V
  - a stalk  F(e) ∈ ℝ^{d_e}   to every edge    e ∈ E  (undirected: one stalk)
  - restriction maps  F_{v◁e} : F(v) → F(e)  for each (vertex, incident edge) pair.

The **sheaf Laplacian** L_F = B^T B where B is the coboundary operator:
  B ∈ ℝ^{(Σ_e d_e) × (Σ_v d_v)},   B_{e, v} = ±F_{v◁e}

**SheafConv** propagates features via the learnable sheaf diffusion:
  X' = (I - α L_F) X   or   X' = (D^{-1/2} L_F D^{-1/2}) X

The restriction maps F_{v◁e} are either
  (a) fixed from a ``Presheaf`` object (formal, for structured domains), or
  (b) learnable linear maps (standard GNN setting).

References
----------
- Hansen & Ghrist 2020 — "Sheaf Neural Networks"
- Bodnar et al. 2022  — "Neural Sheaf Diffusion"
- Barbero et al. 2022 — "Sheaf Attention Networks"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .formal_category import FiniteCategory, Presheaf


# ------------------------------------------------------------------ #
# Coboundary operator construction                                     #
# ------------------------------------------------------------------ #

def build_coboundary(
    vertices: List[str],
    edges: List[Tuple[str, str]],
    restriction_maps: Dict[Tuple[str, str], torch.Tensor],
    stalk_dim_v: int,
    stalk_dim_e: int,
) -> torch.Tensor:
    """
    Assemble the coboundary matrix  B ∈ ℝ^{(|E|·d_e) × (|V|·d_v)}.

    For each oriented edge  e = (u, v):
      - row-block e gets  +F_{v◁e}  in column-block v
      - row-block e gets  -F_{u◁e}  in column-block u

    Parameters
    ----------
    vertices : list of vertex names (order defines column-block layout)
    edges    : list of (tail, head) pairs (order defines row-block layout)
    restriction_maps : dict mapping (vertex_name, edge_idx) → ℝ^{d_e × d_v} tensor
        Key convention: (v, i) where i is the index into ``edges``.
    stalk_dim_v : feature dimension at each vertex
    stalk_dim_e : feature dimension at each edge
    """
    n_v = len(vertices)
    n_e = len(edges)
    B = torch.zeros(n_e * stalk_dim_e, n_v * stalk_dim_v)
    v_idx = {v: i for i, v in enumerate(vertices)}

    for e_i, (u, v) in enumerate(edges):
        row_s = e_i * stalk_dim_e
        row_t = row_s + stalk_dim_e

        # +F_{v ◁ e} at column v
        col_v = v_idx[v] * stalk_dim_v
        R_v = restriction_maps.get((v, e_i), torch.eye(stalk_dim_e, stalk_dim_v))
        B[row_s:row_t, col_v : col_v + stalk_dim_v] += R_v

        # −F_{u ◁ e} at column u
        col_u = v_idx[u] * stalk_dim_v
        R_u = restriction_maps.get((u, e_i), torch.eye(stalk_dim_e, stalk_dim_v))
        B[row_s:row_t, col_u : col_u + stalk_dim_v] -= R_u

    return B  # (n_e·d_e, n_v·d_v)


def sheaf_laplacian_from_coboundary(B: torch.Tensor) -> torch.Tensor:
    """
    Compute the sheaf Laplacian  L_F = B^T B.

    Shape: (n_v·d_v, n_v·d_v) — positive semidefinite by construction.
    """
    return B.T @ B


# ------------------------------------------------------------------ #
# SheafLaplacian: differentiable module                               #
# ------------------------------------------------------------------ #

class SheafLaplacian(nn.Module):
    """
    Learnable sheaf Laplacian for a fixed graph topology.

    The restriction maps F_{v◁e} ∈ ℝ^{d_e × d_v} are learnable parameters.
    During forward, the module assembles B, computes L = B^T B, and returns
    the diffused signal  (I − α L) X  (heat-equation truncation).

    Parameters
    ----------
    n_vertices   : int   — number of graph vertices
    edges        : list of (int, int) pairs (vertex indices, 0-based)
    stalk_dim_v  : int   — stalk dimension at each vertex
    stalk_dim_e  : int   — stalk dimension at each edge
    alpha        : float — diffusion step size  (default 0.1)
    """

    def __init__(
        self,
        n_vertices: int,
        edges: List[Tuple[int, int]],
        stalk_dim_v: int = 1,
        stalk_dim_e: int = 1,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.n_v = n_vertices
        self.edges = edges
        self.d_v = stalk_dim_v
        self.d_e = stalk_dim_e
        self.alpha = alpha

        n_e = len(edges)
        # Two restriction maps per edge: R_tail and R_head
        # Shape: (n_e, 2, d_e, d_v)
        self.restriction_maps = nn.Parameter(
            torch.randn(n_e, 2, stalk_dim_e, stalk_dim_v) * 0.01
        )

    def _build_coboundary(self) -> torch.Tensor:
        """Assemble the coboundary operator B ∈ ℝ^{(n_e·d_e) × (n_v·d_v)}."""
        n_e = len(self.edges)
        n_v = self.n_v
        d_v, d_e = self.d_v, self.d_e
        B = torch.zeros(n_e * d_e, n_v * d_v, device=self.restriction_maps.device)

        for e_i, (u, v) in enumerate(self.edges):
            row_s = e_i * d_e
            row_t = row_s + d_e

            # +R_head (v ◁ e)
            col_v = v * d_v
            B[row_s:row_t, col_v : col_v + d_v] += self.restriction_maps[e_i, 1]

            # −R_tail (u ◁ e)
            col_u = u * d_v
            B[row_s:row_t, col_u : col_u + d_v] -= self.restriction_maps[e_i, 0]

        return B

    def laplacian(self) -> torch.Tensor:
        """Return the sheaf Laplacian L_F = B^T B  (n_v·d_v × n_v·d_v)."""
        B = self._build_coboundary()
        return B.T @ B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Diffuse node features through the sheaf Laplacian.

        Parameters
        ----------
        x : Tensor of shape (batch, n_vertices, d_v) or (n_vertices, d_v)

        Returns
        -------
        Tensor of same shape:  (I − α L_F) x   (vectorised over the batch).
        """
        L = self.laplacian()  # (n_v·d_v, n_v·d_v)
        n_v, d_v = self.n_v, self.d_v

        batched = x.dim() == 3
        if not batched:
            x = x.unsqueeze(0)  # (1, n_v, d_v)

        batch = x.shape[0]
        # Flatten vertex stalks: (batch, n_v·d_v)
        x_flat = x.reshape(batch, n_v * d_v)
        # Diffuse: x' = x - α L x
        x_diff = x_flat - self.alpha * (x_flat @ L.T)
        out = x_diff.reshape(batch, n_v, d_v)

        return out if batched else out.squeeze(0)


# ------------------------------------------------------------------ #
# SheafConv: full convolutional layer                                 #
# ------------------------------------------------------------------ #

class SheafConv(nn.Module):
    """
    Sheaf convolutional layer: one step of sheaf-diffusion followed by
    a per-vertex linear transform and optional nonlinearity.

    Architecture:
        1. Diffuse:   H = (I − α L_F) X    via SheafLaplacian
        2. Transform: Y_v = W H_v + b       (shared or per-vertex linear)
        3. Activate:  Z_v = σ(Y_v)

    Parameters
    ----------
    n_vertices    : int
    edges         : list[(int, int)]
    in_channels   : int  — input feature channels  (= d_v input stalk dim)
    out_channels  : int  — output feature channels
    stalk_dim_e   : int  — edge stalk dimension for sheaf Laplacian
    alpha         : float — diffusion step size
    bias          : bool
    """

    def __init__(
        self,
        n_vertices: int,
        edges: List[Tuple[int, int]],
        in_channels: int,
        out_channels: int,
        stalk_dim_e: int = 1,
        alpha: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.n_v = n_vertices
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.laplacian_layer = SheafLaplacian(
            n_vertices=n_vertices,
            edges=edges,
            stalk_dim_v=in_channels,
            stalk_dim_e=stalk_dim_e,
            alpha=alpha,
        )
        # Shared linear transform across all vertices
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, n_vertices, in_channels)
              or            (n_vertices, in_channels)

        Returns
        -------
        Tensor of shape (batch, n_vertices, out_channels)
              or        (n_vertices, out_channels)
        """
        batched = x.dim() == 3
        if not batched:
            x = x.unsqueeze(0)

        # Step 1: sheaf diffusion
        h = self.laplacian_layer(x)     # (batch, n_v, in_channels)

        # Step 2: linear transform (shared across vertices and batch)
        out = self.linear(h)            # (batch, n_v, out_channels)

        return out if batched else out.squeeze(0)


# ------------------------------------------------------------------ #
# SheafNet: multi-layer sheaf network                                 #
# ------------------------------------------------------------------ #

class SheafNet(nn.Module):
    """
    Multi-layer Sheaf Neural Network.

    Each layer is a ``SheafConv`` followed by ReLU (except the last layer).

    Parameters
    ----------
    n_vertices   : int
    edges        : list[(int, int)]
    channels     : list[int]   — channel widths; len(channels)-1 = number of layers
    stalk_dim_e  : int
    alpha        : float
    """

    def __init__(
        self,
        n_vertices: int,
        edges: List[Tuple[int, int]],
        channels: List[int],
        stalk_dim_e: int = 1,
        alpha: float = 0.1,
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("channels must have at least 2 entries (input and output).")
        self.layers = nn.ModuleList([
            SheafConv(
                n_vertices=n_vertices,
                edges=edges,
                in_channels=channels[i],
                out_channels=channels[i + 1],
                stalk_dim_e=stalk_dim_e,
                alpha=alpha,
                bias=True,
            )
            for i in range(len(channels) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


# ------------------------------------------------------------------ #
# Formal presheaf → SheafLaplacian factory                            #
# ------------------------------------------------------------------ #

def sheaf_laplacian_from_presheaf(
    presheaf: Presheaf,
    vertex_objects: List[str],
    edge_morphisms: List[str],
    stalk_dim: int,
    alpha: float = 0.1,
) -> SheafLaplacian:
    """
    Construct a **fixed** (non-learnable) ``SheafLaplacian`` from a formal
    ``Presheaf`` over a ``FiniteCategory``.

    The ``Presheaf`` defines restriction maps  F(f) : F(B) → F(A)  for each
    morphism  f : A → B.  We interpret:
      - ``vertex_objects`` = objects of the base category (nodes of the graph)
      - ``edge_morphisms`` = non-identity morphisms (undirected edges)

    The restriction maps  F_{v◁e}  are taken from the presheaf's restriction
    functions, encoded as identity matrices scaled by the presheaf value.

    Since a formal presheaf maps finite sets to finite sets (not vector spaces),
    this factory treats each element of the presheaf's stalk as a basis vector
    and maps restriction functions to the corresponding permutation matrices.

    Parameters
    ----------
    presheaf       : the formal ``Presheaf`` object
    vertex_objects : list of object names to use as vertices
    edge_morphisms : list of morphism names to use as edges
                     (each morphism f: A→B gives the edge (src, dst))
    stalk_dim      : embedding dimension for each stalk element
    alpha          : diffusion step size

    Returns
    -------
    SheafLaplacian  with fixed (frozen) restriction maps
    """
    C = presheaf.category
    v_idx = {v: i for i, v in enumerate(vertex_objects)}
    edges = []
    restriction_maps: Dict[Tuple[str, str], torch.Tensor] = {}

    stalk_sizes = {v: len(presheaf.sets[v]) for v in vertex_objects}
    elem_idx = {v: {e: i for i, e in enumerate(sorted(presheaf.sets[v]))}
                for v in vertex_objects}

    for e_i, mor_name in enumerate(edge_morphisms):
        src_obj = C.source(mor_name)   # A : f : A → B
        dst_obj = C.target(mor_name)   # B
        if src_obj not in v_idx or dst_obj not in v_idx:
            continue
        edges.append((v_idx[src_obj], v_idx[dst_obj]))

        # Presheaf restriction F(f) : F(B) → F(A)  (contravariant)
        restr = presheaf.restrictions[mor_name]  # dict {b_elem: a_elem}

        # Build permutation matrix of shape (stalk_dim, stalk_dim)
        # treating stalk elements as one-hot encoded in ℝ^{stalk_dim}
        n_src = stalk_sizes[src_obj]
        n_dst = stalk_sizes[dst_obj]
        dim = max(n_src, n_dst, stalk_dim)

        R_dst = torch.zeros(dim, dim)
        for b_elem, a_elem in restr.items():
            b_i = elem_idx[dst_obj].get(b_elem)
            a_i = elem_idx[src_obj].get(a_elem)
            if b_i is not None and a_i is not None and b_i < dim and a_i < dim:
                R_dst[a_i, b_i] = 1.0

        R_src = torch.eye(dim)
        restriction_maps[(src_obj, e_i)] = R_src[:stalk_dim, :stalk_dim]
        restriction_maps[(dst_obj, e_i)] = R_dst[:stalk_dim, :stalk_dim]

    layer = SheafLaplacian(
        n_vertices=len(vertex_objects),
        edges=edges,
        stalk_dim_v=stalk_dim,
        stalk_dim_e=stalk_dim,
        alpha=alpha,
    )

    # Freeze and initialise with formal restriction maps
    with torch.no_grad():
        for e_i, (u_idx, v_idx_val) in enumerate(edges):
            u_name = vertex_objects[u_idx]
            v_name = vertex_objects[v_idx_val]
            mor_name = edge_morphisms[e_i]
            R_u = restriction_maps.get((u_name, e_i), torch.eye(stalk_dim))
            R_v = restriction_maps.get((v_name, e_i), torch.eye(stalk_dim))
            layer.restriction_maps[e_i, 0].copy_(R_u[:stalk_dim, :stalk_dim])
            layer.restriction_maps[e_i, 1].copy_(R_v[:stalk_dim, :stalk_dim])

    for param in layer.parameters():
        param.requires_grad_(False)

    return layer
