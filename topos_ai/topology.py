import torch
import networkx as nx

class PersistentHomology:
    """
    Topological Data Analysis (TDA) modülü.
    Bir olasılık/kategori matrisi üzerindeki bilgi deliklerini (Holes/Paradoxes)
    ve bilgi adalarını (Connected Components) Betti Sayıları (β) ile hesaplar.
    """
    def __init__(self, num_nodes):
        self.N = num_nodes

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
        
        # Üçgenleri (2-Simplex) delik saymamak için çıkarıyoruz
        triangles = sum(nx.triangles(G).values()) // 3
        betti_1 = max(0, betti_1_raw - triangles)
        
        return betti_0, betti_1
