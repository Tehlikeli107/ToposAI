import torch
import torch.nn as nn
from .logic import SubobjectClassifier

# =====================================================================
# LAWVERE-TIERNEY TOPOLOGIES (UNIVERSE CREATION / SUBTOPOSES)
# Amacı: Bir Topos (Evren) içinde, farklı fiziksel/mantıksal kuralları
# olan bir Alt-Evren (Subtopos) yaratmak.
# j: Ω -> Ω bir Lawvere-Tierney operatörüdür. 
# Yapay zeka, dış dünyanın bulanık veya çelişkili (Intuitionistic) 
# verileriyle başa çıkamadığında, kendi bilincinde bir 'j' topolojisi
# tanımlayarak o verileri izole bir Alt-Evren'e (Subtopos) hapseder
# ve orada yepyeni bir mantıkla (Örn: Kesin Boolean) düşünür.
# =====================================================================

class LawvereTierneyTopology(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega = SubobjectClassifier()

    def double_negation_topology(self, P):
        """
        [THE DOUBLE NEGATION TOPOLOGY: j(p) = ~~p]
        En ünlü Lawvere-Tierney topolojisidir.
        Bulanık (Fuzzy/Heyting) bir Topos'un içindeki HER ŞEYİ,
        'Siyah ve Beyaz (Boolean)' bir Alt-Evrene çökerterek yansıtır.
        Böylece makine, karmaşık ihtimallerden kurtulup kesin sonuçlar görür.
        """
        not_P = self.omega.logical_not(P)
        not_not_P = self.omega.logical_not(not_P)
        return not_not_P

    def closed_topology(self, P, C):
        """
        [THE CLOSED TOPOLOGY: j_c(p) = p V c]
        C (Context) sabit bir topolojik kapalı kümedir.
        Makine bu topolojiyi uyguladığında: "Eğer C bağlamındaysak,
        her şey 'Doğru' kabul edilsin" (Sandbox/Safe-Mode) diyerek
        bir Hata-Toleranslı Alt-Evren (Fault-Tolerant Subtopos) yaratır.
        """
        return self.omega.logical_or(P, C)

    def check_axioms(self, P, Q, C, topology_func):
        """
        [THE 3 SACRED AXIOMS OF LAWVERE-TIERNEY]
        Matematiksel olarak bir j operatörünün 'Evren Yaratabilmesi' için
        şu 3 kuralı SIFIR HATA (0.0) ile sağlaması ŞARTTIR.
        """
        T = torch.ones_like(P)
        
        # Eğer parametre isteyen bir topolojiyse (Closed Topology) onu ayarla
        if topology_func.__name__ == "closed_topology":
            j = lambda x: topology_func(x, C)
        else:
            j = topology_func

        # 1. AXIOM: j(True) == True (Hakikat korunmalıdır)
        ax1_diff = torch.abs(j(T) - T).max().item()

        # 2. AXIOM: j(j(p)) == j(p) (İdempotent / Sabitlik)
        ax2_diff = torch.abs(j(j(P)) - j(P)).max().item()

        # 3. AXIOM: j(p AND q) == j(p) AND j(q) (Kesişimler / Çarpımlar Korunur)
        P_and_Q = self.omega.logical_and(P, Q)
        left_side = j(P_and_Q)
        
        j_P = j(P)
        j_Q = j(Q)
        right_side = self.omega.logical_and(j_P, j_Q)
        
        ax3_diff = torch.abs(left_side - right_side).max().item()

        return ax1_diff, ax2_diff, ax3_diff
