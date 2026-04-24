import networkx as nx
import torch


class PersistentHomology:
    """
    Topological Data Analysis (TDA) modülü.
    Bir olasılık/kategori matrisi üzerindeki bilgi deliklerini (Holes/Paradoxes)
    ve bilgi adalarını (Connected Components) Betti Sayılarını (β) ile hesaplar.
    """
    def __init__(self, num_nodes):
        self.N = num_nodes

    def _boundary_matrix_rank(self, G, triangles):
        import numpy as np
        edges = list(G.edges())
        if not triangles or not edges:
            return 0
        edge_idx = {(min(u,v), max(u,v)): i for i, (u,v) in enumerate(edges)}
        B2 = np.zeros((len(edges), len(triangles)))
        for col, tri in enumerate(triangles):
            u, v, w = sorted(tri)
            # Üçgenin sınırları: [v, w] - [u, w] + [u, v] (Alternating sum)
            B2[edge_idx[(v,w)], col] = 1.0
            B2[edge_idx[(u,w)], col] = -1.0
            B2[edge_idx[(u,v)], col] = 1.0
        return int(np.linalg.matrix_rank(B2))

    def calculate_betti(self, distance_matrix, threshold):
        """
        Belirli bir eşik (Threshold) değerinde Simplicial Complex kurar
        ve Betti-0 (β0) ile Betti-1 (β1) sayılarını hesaplar.
        """
        G = nx.Graph()
        G.add_nodes_from(range(self.N))

        for i in range(self.N):
            for j in range(i + 1, self.N):
                # distance_matrix genellikle 1.0 - R_matrix şeklindedir
                if distance_matrix[i, j] <= threshold:
                    G.add_edge(i, j)

        # β0: Bağlantılı alt ağların sayısı
        betti_0 = nx.number_connected_components(G)

        # β1: 1D Deliklerin (Bağımsız Döngüler) Sayısı
        edges = G.number_of_edges()
        nodes = G.number_of_nodes()
        betti_1_raw = edges - nodes + betti_0

        # Üçgenleri (2-Simplex) delik saymamak için doğru sınır (boundary) matrisiyle çıkarıyoruz
        triangles = [tuple(c) for c in nx.enumerate_all_cliques(G) if len(c) == 3]
        dim_im_d2 = self._boundary_matrix_rank(G, triangles)
        betti_1 = max(0, betti_1_raw - dim_im_d2)

        return betti_0, betti_1
