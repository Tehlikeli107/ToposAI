import torch

# =====================================================================
# ČECH COHOMOLOGY ENGINE (GROTHENDIECK TOPOLOGIES)
# Amacı: Merkeziyetsiz veya Parçalı (Local) veri yapılarından,
# Global bir "Hakikat (Consensus)" çıkarılıp çıkarılamayacağını
# topolojik olarak ölçmek.
# Eğer H^1 (Birinci Kohomoloji Grubu) > 0 ise, sistemde "Lokal olarak
# tutarlı ama Global olarak çelişkili" bir Paradoks/Engel (Obstruction)
# vardır (Örn: Penrose Merdiveni, Finansal Hortumlama Döngüleri).
# =====================================================================

class CechCohomology:
    def __init__(self, num_nodes, edges):
        """
        num_nodes: Ajan veya Veri Düğümü sayısı (V)
        edges: Düğümlerin birbirleriyle kesiştiği yerler (E). Format: [(i, j), ...]
        """
        self.num_nodes = num_nodes
        self.edges = edges
        self.num_edges = len(edges)

        # d0: C^0 -> C^1 Sınır Operatörü (Boundary/Coboundary Matrix)
        # Boyut: [Kenar Sayısı, Düğüm Sayısı]
        # Her kenar (i, j) için: Düğüm j'de +1, Düğüm i'de -1.
        self.d0 = torch.zeros(self.num_edges, self.num_nodes, dtype=torch.float32)
        self._build_d0_matrix()

    def _build_d0_matrix(self):
        for e_idx, (i, j) in enumerate(self.edges):
            # Yönlü kesişim: i'den j'ye
            self.d0[e_idx, i] = -1.0
            self.d0[e_idx, j] = 1.0

    def compute_H0_consensus(self, local_sections):
        """
        Lokal kesitler (Ajanların elindeki veriler).
        Eğer d0 * local_sections == 0 ise, veriler kesişim noktalarında
        %100 uyumludur. Bu, H^0'ın (Global Section) bir elemanıdır.
        """
        local_sections = local_sections.view(self.num_nodes, -1) # [V, D]

        # Kesişimlerdeki fikir ayrılıkları (Disagreements on overlaps)
        # [E, V] * [V, D] = [E, D]
        disagreements = torch.matmul(self.d0, local_sections)

        # Hata payı (L2 Norm)
        total_disagreement = torch.norm(disagreements).item()

        # Betti-0 (H^0 boyutu): d0 matrisinin Null Space (Çekirdek) boyutudur.
        # Rank-Nullity teoremi: Nullity = V - Rank(d0)
        rank_d0 = torch.linalg.matrix_rank(self.d0).item()
        betti_0 = self.num_nodes - rank_d0

        return total_disagreement, betti_0

    def compute_H1_obstruction(self, edge_flows):
        """
        Eğer ajanlar arası veri transferi (Edge Flows / C^1 zincirleri) varsa,
        bu akışın global bir potansiyelden (C^0) mi türediğini, yoksa kendi
        içinde dönen kapalı bir "Hortum/Döngü" (H^1 Obstruction) mü olduğunu ölçer.

        edge_flows: [Kenar Sayısı, D]
        H^1 = ker(d1) / im(d0). Basit ağlarda d1 yoktur (veya ker(d1)=C^1'dir).
        Bu durumda H^1 boyutu Betti-1 = E - Rank(d0).
        Ancak biz "Verili" bir akışın H^1 elemanı olup olmadığına (Yani im(d0)'a
        dik olup olmadığına) bakıyoruz.
        """
        edge_flows = edge_flows.view(self.num_edges, -1)

        # Edge flows'un d0'ın imajında (Image) olup olmadığını kontrol et.
        # Bunun için edge_flows'u d0'ın sütun uzayına (Column Space) yansıtıyoruz.
        # Eğer d0'ın sütun uzayında değilse (Yani d0 * x = edge_flows çözümü yoksa),
        # bu akış bir H^1 OBSTRUCTION (Paradoks/Dolandırıcılık Döngüsü) demektir!

        # Pseudo-inverse kullanarak en yakın çözümü (x) bulalım
        d0_pinv = torch.linalg.pinv(self.d0) # [V, E]
        best_fit_potentials = torch.matmul(d0_pinv, edge_flows) # [V, D]

        # Bu potansiyellerin yarattığı akış
        projected_flows = torch.matmul(self.d0, best_fit_potentials) # [E, D]

        # Gerçek akış ile Olası/Yasal akış arasındaki fark (H^1 Class / Obstruction Vector)
        obstruction_vector = edge_flows - projected_flows
        obstruction_magnitude = torch.norm(obstruction_vector).item()

        # Betti-1 (H^1 boyutu)
        rank_d0 = torch.linalg.matrix_rank(self.d0).item()
        betti_1 = self.num_edges - rank_d0

        return obstruction_magnitude, betti_1, obstruction_vector
