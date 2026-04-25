import networkx as nx
import torch


class PersistentHomology:
    """
    Minimal Vietoris-Rips persistent-homology helper.

    At a fixed threshold, this builds an undirected graph and fills all
    triangles as 2-simplices. It returns Betti-0 and Betti-1 for that finite
    complex.
    """

    def __init__(self, num_nodes):
        self.N = num_nodes

    def _boundary_matrix_rank(self, G, triangles):
        import numpy as np

        edges = list(G.edges())
        if not triangles or not edges:
            return 0

        edge_idx = {(min(u, v), max(u, v)): i for i, (u, v) in enumerate(edges)}
        B2 = np.zeros((len(edges), len(triangles)))

        for col, tri in enumerate(triangles):
            u, v, w = sorted(tri)
            B2[edge_idx[(v, w)], col] = 1.0
            B2[edge_idx[(u, w)], col] = -1.0
            B2[edge_idx[(u, v)], col] = 1.0

        return int(np.linalg.matrix_rank(B2))

    def calculate_betti(self, distance_matrix, threshold):
        """
        Build a thresholded simplicial complex and return `(beta0, beta1)`.
        """
        G = nx.Graph()
        G.add_nodes_from(range(self.N))

        for i in range(self.N):
            for j in range(i + 1, self.N):
                if distance_matrix[i, j] <= threshold:
                    G.add_edge(i, j)

        beta_0 = nx.number_connected_components(G)
        beta_1_raw = G.number_of_edges() - G.number_of_nodes() + beta_0

        triangles = [tuple(c) for c in nx.enumerate_all_cliques(G) if len(c) == 3]
        dim_im_d2 = self._boundary_matrix_rank(G, triangles)
        beta_1 = max(0, beta_1_raw - dim_im_d2)

        return beta_0, beta_1
