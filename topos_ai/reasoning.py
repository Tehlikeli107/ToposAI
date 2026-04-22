import torch
from .math import soft_godel_composition, lukasiewicz_composition

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

    def deliberate(self, iterations=3):
        """Topolojik Mahkeme Kararı: İptalleri Uygula ve Geçişlilik Yap."""
        filtered_pos = self.R_pos.clone()
        
        # 1. Hangi düğümlerin olaya müdahil olduğu (Reachability)
        R_reach = self.R_pos.clone()
        for _ in range(iterations):
            R_reach = torch.max(R_reach, lukasiewicz_composition(R_reach, R_reach))
            
        active_facts = R_reach[0, :]
        active_facts[0] = 1.0 
        
        # 2. Topolojik Makas (İptalleri uygula)
        for i in range(self.N):
            if active_facts[i] > 0:
                for j in range(self.N):
                    if self.R_neg[i, j] > 0:
                        defeater_strength = self.R_neg[i, j].item()
                        filtered_pos[:, j] *= (1.0 - defeater_strength)

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
            
            for i in range(self.N):
                for j in range(self.N):
                    # Daha önce bağ zayıf (<0.1) iken sentezle güçlendiyse (>Threshold)
                    if i != j and R_current[i, j] < 0.1 and R_next[i, j] > threshold:
                        new_theorems.append((i, j, step + 1, R_next[i, j].item()))
                        
            R_current = torch.max(R_current, R_next)
            
        return R_current, new_theorems
