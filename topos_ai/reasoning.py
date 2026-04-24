import torch

from .math import lukasiewicz_composition, soft_godel_composition


class DefeasibleReasoning:
    """
    [AI YARGIÇ] Pozitif (Destek) ve Negatif (İptal / Defeater) okları
    Kategori Teorisinde birleştirerek çelişki çözen Nöro-Sembolik Mantık.
    """
    def __init__(self, num_nodes):
        self.N = num_nodes
        self.R_pos = torch.zeros(self.N, self.N)
        self.R_neg = torch.zeros(self.N, self.N)

    def add_rule(self, u, v, weight=1.0, is_defeater=False):
        if is_defeater:
            self.R_neg[u, v] = weight
        else:
            self.R_pos[u, v] = weight

    def deliberate(self, iterations=3, start_node=0):
        """Topolojik Mahkeme Kararı: İptalleri Uygula ve Geçişlilik Yap."""
        filtered_pos = self.R_pos.clone()

        # 1. Hangi düğümlerin olaya müdahil olduğu (Reachability)
        R_reach = self.R_pos.clone()
        for _ in range(iterations):
            R_reach = torch.max(R_reach, lukasiewicz_composition(R_reach, R_reach))

        active_facts = R_reach[start_node, :]
        active_facts[start_node] = 1.0

        # 2. Topolojik Makas (İptalleri uygula - Vektörize)
        # Sadece aktif olan düğümlerden gelen negatif okları (Defeaters) al
        active_mask = (active_facts > 0).float().unsqueeze(1) # [N, 1]
        active_R_neg = self.R_neg * active_mask # [N, N]

        # Her 'j' hedefi için gelen tüm aktif negatif okların (1 - güç) çarpanlarını bul ve çarp
        defeater_multipliers = torch.prod(1.0 - active_R_neg, dim=0) # [N]

        # Filtrelenmiş pozitif matrise tüm çarpanları aynı anda (Broadcasting) uygula
        filtered_pos = filtered_pos * defeater_multipliers.unsqueeze(0)

        # 3. Sağlam Kanunlarla Geçişlilik (Closure)
        R_current_pos = filtered_pos.clone()
        for _ in range(iterations):
            R_next_pos = lukasiewicz_composition(R_current_pos, R_current_pos)
            R_current_pos = torch.max(R_current_pos, R_next_pos)

        return R_current_pos

class AutonomousTheoremProver:
    """
    [OTONOM İCAT] Verilen temel aksiyomları (A->B, B->C) MCTS mantığıyla
    ve Soft-Gödel Pürüzsüz (Smooth) Geçişliliğiyle çarparak, sözlükte
    olmayan yepyeni Teoremler (A->C) sentezleyen makine.
    """
    def __init__(self, R_initial):
        self.N = R_initial.size(0)
        self.R = R_initial.clone()

    def discover_theorems(self, iterations=3, threshold=0.75):
        """Yeni yollar (Teoremler) bulduğunda bunların listesini (ve yeni evreni) döner."""
        new_theorems = []
        R_current = self.R.clone()

        for step in range(iterations):
            R_next = soft_godel_composition(R_current, R_current, tau=5.0)

            # Vektörize edilmiş Teorem Keşfi (O(N^2) Python döngüsünü kaldırır)
            new_mask = (R_current < 0.1) & (R_next > threshold)
            new_mask.fill_diagonal_(False)

            indices = new_mask.nonzero(as_tuple=False)
            for idx in indices:
                i, j = idx.tolist()
                new_theorems.append((i, j, step + 1, R_next[i, j].item()))

            R_current = torch.max(R_current, R_next)

        return R_current, new_theorems
