import torch
import torch.nn as nn

# =====================================================================
# INFINITY-CATEGORIES (HIGHER TOPOS THEORY) & SIMPLICIAL NETWORKS
# Amacı: Klasik YZ (1-Category) sadece düğümler ve kenarlar (Nodes & Edges) 
# üzerinden çalışır. Jacob Lurie'nin Higher Topos Theory'si, evrenin 
# sonsuz boyutlu morfizmalardan (Oklar, Yüzeyler, Hacimler) oluştuğunu 
# söyler.
# Bu modül, veri uzayındaki noktaları bir "Simplicial Set (Kan Complex)"
# olarak inşa eder ve Hodge Laplacian operatörleri (L_0, L_1, L_2) ile
# Yüksek Kategori Teorisini (Higher-Order Message Passing) 
# PyTorch üzerinde Donanımsal olarak uygulanabilir hale getirir.
# =====================================================================

class SimplicialComplexBuilder:
    """Veri bulutundan (Point Cloud) sonsuz-kategori (Simplex) çıkarıcı."""
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon

    def build_complex(self, X):
        """
        X: [N, D] veri noktaları
        Returns: Düğümler, Kenarlar (1-Morphisms) ve Üçgenler (2-Morphisms)
        """
        N = X.size(0)
        dist_matrix = torch.cdist(X, X, p=2)
        
        # 1. 1-Morfizmalar (Edges)
        adj = (dist_matrix < self.epsilon).float()
        adj.fill_diagonal_(0.0)
        
        edges = torch.nonzero(torch.triu(adj)).tolist()
        
        # Kenar index haritası
        edge_to_idx = {(i, j): idx for idx, (i, j) in enumerate(edges)}
        
        # 2. 2-Morfizmalar (Triangles / 2-Simplices)
        triangles = []
        for i, j in edges:
            # i ve j'nin ortak komşuları (k)
            common_neighbors = torch.nonzero(adj[i] * adj[j]).squeeze(-1).tolist()
            for k in common_neighbors:
                if k > j: # Sıralı tutmak için (i < j < k)
                    triangles.append((i, j, k))
                    
        return edges, triangles, edge_to_idx

class HodgeLaplacianEngine:
    """Kategori Teorisindeki Sınır (Boundary) ve Hodge Laplacian Matrisleri."""
    def __init__(self, num_nodes, edges, triangles, edge_to_idx):
        self.V = num_nodes
        self.E = len(edges)
        self.T = len(triangles)
        
        self.B1 = torch.zeros((self.V, self.E), dtype=torch.float32)
        self.B2 = torch.zeros((self.E, self.T), dtype=torch.float32)
        
        self._build_boundaries(edges, triangles, edge_to_idx)
        
    def _build_boundaries(self, edges, triangles, edge_to_idx):
        # B1: Kenarlardan Düğümlere (1-Morphism -> 0-Morphism)
        for e_idx, (i, j) in enumerate(edges):
            self.B1[i, e_idx] = -1.0
            self.B1[j, e_idx] = 1.0
            
        # B2: Üçgenlerden Kenarlara (2-Morphism -> 1-Morphism)
        for t_idx, (i, j, k) in enumerate(triangles):
            e_ij = edge_to_idx[(i, j)]
            e_jk = edge_to_idx[(j, k)]
            e_ik = edge_to_idx[(i, k)]
            
            # Alternating sum (Kategori teorisindeki yönelim)
            self.B2[e_ij, t_idx] = 1.0
            self.B2[e_jk, t_idx] = 1.0
            self.B2[e_ik, t_idx] = -1.0
            
    def get_laplacians(self):
        # L0 = B1 * B1^T (Düğüm uzayındaki topoloji / Klasik Graph Laplacian)
        L0 = torch.matmul(self.B1, self.B1.t())
        
        # L1 = B1^T * B1 + B2 * B2^T (Kenar uzayındaki Üst-Topoloji)
        # Bu, bilginin sadece kenarlardan değil, YÜZEYLERDEN (2-Morphisms) de akmasını sağlar!
        L1_lower = torch.matmul(self.B1.t(), self.B1)
        L1_upper = torch.matmul(self.B2, self.B2.t())
        L1 = L1_lower + L1_upper
        
        return L0, L1

class InfinityCategoryLayer(nn.Module):
    """
    Sonsuz-Kategori (Higher Topos) Sinir Ağı Katmanı.
    Aynı anda hem 0-Hücreleri (Nodes) hem de 1-Hücreleri (Edges)
    kendi Hodge Laplacian'ları üzerinden eğitir.
    """
    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__()
        self.W0 = nn.Linear(node_dim, out_dim)
        self.W1 = nn.Linear(edge_dim, out_dim)
        
    def forward(self, H0, H1, L0, L1):
        # Düğüm bilgisi L0 topolojisi üzerinden akar
        H0_new = torch.relu(self.W0(torch.matmul(L0, H0)))
        
        # Kenar bilgisi L1 (Yüzeyler/Üçgenler) topolojisi üzerinden akar!
        # KLASİK YZ BURAYI GÖREMEZ!
        H1_new = torch.relu(self.W1(torch.matmul(L1, H1)))
        
        return H0_new, H1_new
