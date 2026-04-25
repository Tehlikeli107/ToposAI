import torch


class CechCohomology:
    """
    Small Cech-style cohomology helper for overlap consistency demos.

    The implementation builds the first coboundary matrix for a graph cover.
    It can measure whether local node sections agree on overlaps (H0-style
    consensus) and whether edge flows fail to come from a global potential
    (H1-style obstruction).
    """

    def __init__(self, num_nodes, edges):
        """
        Args:
            num_nodes: Number of local sections/nodes.
            edges: Oriented overlaps as `(i, j)` pairs.
        """
        self.num_nodes = num_nodes
        self.edges = edges
        self.num_edges = len(edges)

        self.d0 = torch.zeros(self.num_edges, self.num_nodes, dtype=torch.float32)
        self._build_d0_matrix()

    def _build_d0_matrix(self):
        for e_idx, (i, j) in enumerate(self.edges):
            self.d0[e_idx, i] = -1.0
            self.d0[e_idx, j] = 1.0

    def _strict_topological_rank(self, matrix, tol=1e-4):
        """
        Rank with an explicit numerical tolerance.

        Boundary matrices in these demos are exact small tensors, but using an
        explicit tolerance makes the Betti diagnostics stable under small
        floating-point perturbations.
        """
        _, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
        return torch.sum(singular_values > tol).item()

    def compute_H0_consensus(self, local_sections):
        """
        Return overlap disagreement and the dimension of the H0-like kernel.

        `total_disagreement == 0` means every connected overlap agrees for the
        provided local sections.
        """
        local_sections = local_sections.view(self.num_nodes, -1)
        disagreements = torch.matmul(self.d0, local_sections)
        total_disagreement = torch.norm(disagreements).item()

        rank_d0 = self._strict_topological_rank(self.d0)
        betti_0 = self.num_nodes - rank_d0

        return total_disagreement, betti_0

    def compute_H1_obstruction(self, edge_flows):
        """
        Project edge flows onto `im(d0)` and return the residual obstruction.

        For a graph without 2-cells, the coarse Betti-1 count is
        `num_edges - rank(d0)`. The residual vector is zero when the supplied
        flow is exactly induced by node potentials.
        """
        edge_flows = edge_flows.view(self.num_edges, -1)

        d0_pinv = torch.linalg.pinv(self.d0, rcond=1e-4)
        best_fit_potentials = torch.matmul(d0_pinv, edge_flows)
        projected_flows = torch.matmul(self.d0, best_fit_potentials)

        obstruction_vector = edge_flows - projected_flows
        obstruction_magnitude = torch.norm(obstruction_vector).item()

        rank_d0 = self._strict_topological_rank(self.d0)
        betti_1 = self.num_edges - rank_d0

        return obstruction_magnitude, betti_1, obstruction_vector
