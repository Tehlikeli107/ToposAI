from .sheaf_computer import ToposSheafComputer


class PersistentHomology:
    """
    Minimal Vietoris-Rips persistent-homology helper.

    This preserves the historical `topos_ai.topology.PersistentHomology`
    import path now that `topos_ai.topology` is also a package for topology
    utilities.
    """

    def __init__(self, num_nodes):
        self.N = num_nodes

    def _boundary_matrix_rank(self, graph, triangles):
        import numpy as np

        edges = list(graph.edges())
        if not triangles or not edges:
            return 0

        edge_idx = {(min(u, v), max(u, v)): i for i, (u, v) in enumerate(edges)}
        boundary = np.zeros((len(edges), len(triangles)))

        for col, triangle in enumerate(triangles):
            u, v, w = sorted(triangle)
            boundary[edge_idx[(v, w)], col] = 1.0
            boundary[edge_idx[(u, w)], col] = -1.0
            boundary[edge_idx[(u, v)], col] = 1.0

        return int(np.linalg.matrix_rank(boundary))

    def calculate_betti(self, distance_matrix, threshold):
        """Build a thresholded simplicial complex and return `(beta0, beta1)`."""
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(self.N))

        for i in range(self.N):
            for j in range(i + 1, self.N):
                if distance_matrix[i, j] <= threshold:
                    graph.add_edge(i, j)

        beta_0 = nx.number_connected_components(graph)
        beta_1_raw = graph.number_of_edges() - graph.number_of_nodes() + beta_0

        triangles = [tuple(clique) for clique in nx.enumerate_all_cliques(graph) if len(clique) == 3]
        dim_im_d2 = self._boundary_matrix_rank(graph, triangles)
        beta_1 = max(0, beta_1_raw - dim_im_d2)

        return beta_0, beta_1


__all__ = ["PersistentHomology", "ToposSheafComputer"]
