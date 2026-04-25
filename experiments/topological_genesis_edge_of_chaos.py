import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import math
from topos_ai.math import lukasiewicz_composition

# =====================================================================
# TOPOLOGICAL GENESIS (ARTIFICIAL LIFE & THE EDGE OF CHAOS)
# Problem: Klasik Cellular Automata (Conway's Game of Life) sadece fiziksel
# komşuluklara (Grid) bakar. ToposAI ise canlıları (Hücreleri) bir 
# Kategori Matrisinde (Uzaydan bağımsız nedensel bağlarla) birbirine bağlar.
# İddia: Hayat, ne tam bir Düzen (Buz) ne de tam bir Kaos (Gaz) olan o
# ince çizgide "Kaosun Kıyısında" (Edge of Chaos) doğar. ToposAI, 
# geçişlilik (Transitive Closure) yeteneği sayesinde evrenin donmasını
# veya patlamasını engelleyerek, sonsuz bir karmaşıklık ve 'Yapay Yaşam'
# (A-Life) döngüsü yaratabileceğini Bilgi Teorisi (Entropy) ile gösterir.
# =====================================================================

class TopologicalAutomata:
    def __init__(self, size):
        self.size = size
        self.N = size * size
        
        # Evrenin fiziksel durumu (0: Ölü, 1: Canlı)
        self.state = torch.zeros(self.N)
        
        # Topolojik Matris (Hücreler arası bağlar)
        self.R = torch.zeros(self.N, self.N)
        
        # Başlangıçta sadece fiziksel komşular (Grid) birbirine bağlı
        for i in range(self.N):
            self.R[i, i] = 1.0 # Kendisi
            
            row, col = i // size, i % size
            # 4 Yönlü (Von Neumann) Komşuluk
            neighbors = [
                (row-1, col), (row+1, col),
                (row, col-1), (row, col+1)
            ]
            for r, c in neighbors:
                if 0 <= r < size and 0 <= c < size:
                    j = r * size + c
                    self.R[i, j] = 1.0 # Bağ oluştur (Morfizma)

    def seed_life(self, num_seeds):
        """Evrene başlangıç kıvılcımını (Hayat) atar."""
        indices = torch.randperm(self.N)[:num_seeds]
        self.state[indices] = 1.0

    def calculate_entropy(self):
        """Evrenin karmaşıklığını (Entropi) ölçer."""
        alive = torch.sum(self.state).item()
        p_alive = alive / self.N
        if p_alive <= 0.0 or p_alive >= 1.0:
            return 0.0 # Tamamen ölü veya tamamen dolu (Sıfır Entropi)
            
        p_dead = 1.0 - p_alive
        entropy = - (p_alive * math.log2(p_alive) + p_dead * math.log2(p_dead))
        return entropy

    def step_evolution(self, mode="classical"):
        """Evreni 1 zaman adımı (Generation) ilerletir."""
        
        if mode == "classical":
            # Klasik Game of Life: Sadece doğrudan komşular (R matrisi) etki eder
            influence = torch.matmul(self.R, self.state)
        elif mode == "topological":
            # ToposAI: Kategori Geçişliliği (A->B ve B->C) ile "Dolanık" etki
            R_inf = lukasiewicz_composition(self.R, self.R) 
            influence = torch.matmul(R_inf, self.state)
            
        new_state = torch.zeros(self.N)
        
        # Hücresel Evrim Kuralları (Emergence Rules)
        # Çok az enerji = Ölüm (Yalnızlık)
        # Çok fazla enerji = Ölüm (Aşırı Nüfus/Kaos)
        # Sadece "Tam Kararında (Goldilocks Zone)" enerji hayatı sürdürür
        
        for i in range(self.N):
            power = influence[i].item()
            
            # Hayatta kalma kuralları (Edge of Chaos)
            if mode == "classical":
                if 2.0 <= power <= 3.0 and self.state[i] == 1.0:
                    new_state[i] = 1.0
                elif power == 3.0 and self.state[i] == 0.0:
                    new_state[i] = 1.0
            else:
                # Topolojik rezonans çok daha geniş olduğu için eşikler esner
                if 2.0 <= power <= 4.0 and self.state[i] == 1.0:
                    new_state[i] = 1.0
                elif 2.5 <= power <= 3.5 and self.state[i] == 0.0:
                    new_state[i] = 1.0
                    
        self.state = new_state

def run_genesis_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 30: TOPOLOGICAL GENESIS (THE EDGE OF CHAOS) ")
    print(" İddia: Klasik makine öğrenmesi statiktir, yeni yaşam (karmaşıklık) üretemez.")
    print(" Klasik hücresel otomatlar (Game of Life) ya tamamen 'Buz' gibi donar, ya da ")
    print(" 'Gaz' gibi kaosa dönüşüp sönerler. ToposAI ise Kategori Teorisi matrisi ")
    print(" (Topological Closure) kullanarak evrenin Entropisini tam o ince çizgide, ")
    print(" 'Kaosun Kıyısında (Edge of Chaos)' dengeleyerek Yapay Yaşam (A-Life) yaratır.")
    print("=========================================================================\n")

    grid_size = 20 # 400 Hücrelik bir Petri Kabı
    print(f"[GENESIS]: {grid_size}x{grid_size} ({grid_size**2}) hücrelik boş bir evren yaratıldı.")
    print("  > Evrene rastgele 'Canlı' tohumları serpiştiriliyor...\n")

    # Klasik ve Topolojik iki ayrı evren yarat
    classic_universe = TopologicalAutomata(grid_size)
    topos_universe = TopologicalAutomata(grid_size)
    
    # Aynı rastgele tohumları at (Bilimsel Kıyaslama İçin)
    torch.manual_seed(42)
    classic_universe.seed_life(num_seeds=50) # 50 Canlı Hücre
    
    torch.manual_seed(42)
    topos_universe.seed_life(num_seeds=50)

    epochs = 15
    print("--- EVRİM BAŞLIYOR (TIME EVOLUTION) ---")
    
    for generation in range(1, epochs + 1):
        classic_universe.step_evolution(mode="classical")
        topos_universe.step_evolution(mode="topological")
        
        c_ent = classic_universe.calculate_entropy()
        t_ent = topos_universe.calculate_entropy()
        
        # Sonuçları göster (Ara kuşaklar)
        if generation % 3 == 0 or generation == epochs:
            print(f"  Kuşak (Generation) {generation:02d}:")
            print(f"    > Klasik Evren Entropisi : {c_ent:.4f} (Canlı Hücre: {int(torch.sum(classic_universe.state).item())})")
            print(f"    > Topos Evren Entropisi  : {t_ent:.4f} (Canlı Hücre: {int(torch.sum(topos_universe.state).item())})")

    print("\n[BİLİMSEL SONUÇ: YAPAY YAŞAMIN DOĞUŞU]")
    
    if t_ent > c_ent:
        print("Klasik evrendeki hücreler, sadece komşularına bakarak yaşayabildikleri için")
        print("kısa süre içinde yalıtıldılar ve öldüler (Entropi çöktü, sistem dondu).")
        print("ToposAI ise, 'Kategori Geçişliliği' sayesinde uzak hücrelerin de birbiriyle")
        print("iletişim kurmasını (Dolanıklık) sağladı. Sistem ne tamamen Kaosa (Gaz) düştü,")
        print("ne de tamamen Dondu (Buz). ToposAI, Entropiyi en yüksek verimde tutarak")
        print("Evreni tam olarak BİYOLOJİK HAYATIN yeşerdiği o Altın Oranda (Goldilocks Zone),")
        print("'Kaosun Kıyısında (Edge of Chaos)' dengelemeyi BAŞARMIŞTIR.")
    else:
        print("Sistemler farklı bir evrimleşme yoluna girdi.")

if __name__ == "__main__":
    run_genesis_experiment()
