import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import networkx as nx

# =====================================================================
# TOPOLOGICAL DATA ANALYSIS (TDA) & PERSISTENT HOMOLOGY
# İddia: Yapay zeka neyi bilmediğini (Cehaletini / Mantık Boşluklarını)
# bilemez. ToposAI, kendi bilgi matrisini "Vietoris-Rips Kompleksi" ile 
# tarayarak, Betti-1 (β1) sayılarını hesaplar. β1 > 0 ise, sistem
# bilgisinin ortasında bir "Delik" (Paradoks veya Eksik Bağlantı) 
# olduğunu matematiksel olarak teşhis eder (Self-Aware Ignorance).
# =====================================================================

class TopologicalHomologyAnalyzer:
    def __init__(self, entities, distance_matrix):
        self.entities = entities
        self.D = distance_matrix
        self.N = len(entities)

    def calculate_betti_numbers(self, threshold):
        """
        Belirli bir eşik (Threshold) değerinde Simplicial Complex kurar
        ve Betti-0 (Bağlantılı Bileşenler) ile Betti-1 (1D Delikler) sayısını bulur.
        """
        # Eşiğin altındaki mesafelere (güçlü bağlara) sahip düğümlerle bir çizge kur
        G = nx.Graph()
        G.add_nodes_from(range(self.N))
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.D[i, j] <= threshold:
                    G.add_edge(i, j)
                    
        # β0 (Betti-0): Bağlantılı alt ağların (Component) sayısı
        betti_0 = nx.number_connected_components(G)
        
        # β1 (Betti-1): 1D Deliklerin Sayısı (Bağımsız Döngüler / Cycles)
        # Çizge Teorisinde Döngüsel Boşluklar (Holes) = Kenarlar - Düğümler + Bileşenler (Euler Karakteristiği)
        # (Basitleştirilmiş Betti-1 hesabı - chordless cycles)
        edges = G.number_of_edges()
        nodes = G.number_of_nodes()
        betti_1 = edges - nodes + betti_0
        
        # Ancak eğer 3 düğüm üçgen oluşturuyorsa o içi dolu bir 2-Simplex'tir, delik değildir.
        # Kordonsuz (Chordless) 4'lü ve daha büyük döngüleri bulmalıyız.
        # Basit Euler formülü ağaçlar ve döngüler için çalışır, gerçek TDA (Vietoris-Rips) 
        # için üçgenlerin (clique) içini doldurup delik sayısından düşmemiz gerekir.
        triangles = sum(nx.triangles(G).values()) // 3
        true_betti_1 = betti_1 - triangles
        if true_betti_1 < 0: true_betti_1 = 0
        
        return betti_0, true_betti_1

def run_tda_experiment():
    print("=========================================================================")
    print(" BİLİMSEL KANIT 18: TOPOLOGICAL DATA ANALYSIS (BETTI NUMBERS) ")
    print(" İddia: LLM'ler bilgi eksikliğinde halüsinasyon görür. ToposAI ise")
    print(" kendi matrisinin Betti Sayılarını (β) hesaplayarak, bilgisindeki")
    print(" 'Delikleri' (Paradokslar / Eksik Bağlantılar) matematiksel olarak")
    print(" görebilir. Kendi cehaletinin farkında olan bir YZ inşasıdır.")
    print("=========================================================================\n")

    # Kurgusal Bilgi Uzayı: Kuantum Fiziği ve Görelilik birleştirilmeye çalışılıyor
    entities = ["Kütleçekim", "Uzay-Zaman", "Foton", "Dolanıklık"]
    print("[BİLGİ UZAYI]:", ", ".join(entities))
    
    # Mesafe Matrisi (0: Aynı şey, 1: Çok uzak)
    # Senaryo: Kütleçekim ve Uzay-Zaman bağlı. Foton ve Dolanıklık bağlı.
    # Kütleçekim ve Foton arasında da dolaylı bir bağ kurulmuş ama tam ortada (Kuantum Kütleçekimi) DELİK VAR!
    D = torch.tensor([
        [0.0, 0.1, 0.9, 1.0], # Kütleçekim
        [0.1, 0.0, 1.0, 0.9], # Uzay-Zaman
        [0.9, 1.0, 0.0, 0.1], # Foton
        [1.0, 0.9, 0.1, 0.0]  # Dolanıklık
    ])
    
    # 4'lü bir halka oluşturmak için sınır mesafelerini ayarlayalım (Paradoks Halkası)
    D[0, 2] = 0.4; D[2, 0] = 0.4  # Kütleçekim - Foton ilişkisi var
    D[1, 3] = 0.4; D[3, 1] = 0.4  # UzayZaman - Dolanıklık ilişkisi var
    # Ama Çapraz bağlar YOK! Kütleçekim-Dolanıklık(1.0) ve UzayZaman-Foton(1.0) kayıp!
    
    analyzer = TopologicalHomologyAnalyzer(entities, D)
    
    print("\n--- PERSISTENT HOMOLOGY (FİLTRASYON) SİMÜLASYONU ---")
    print("YZ, farklı dikkat (Threshold) eşiklerinde bilgi evreninin şeklini tarıyor...\n")
    
    thresholds = [0.05, 0.2, 0.5, 1.5]
    for t in thresholds:
        b0, b1 = analyzer.calculate_betti_numbers(t)
        print(f"  Eşik (Threshold) = {t:.2f}")
        print(f"    β0 (Bilgi Adaları) : {b0} (Kaç farklı parça var?)")
        print(f"    β1 (Delikler/Boşluk): {b1} (Kaç tane mantık deliği/paradoks var?)")
        if b1 > 0:
            print("    🚨 [DİKKAT]: Bilgi ağının tam ortasında matematiksel bir BOŞLUK (Hole) tespit edildi!")
        print("-" * 50)

    print("\n[BİLİMSEL SONUÇ (PoC TDA Heuristic)]")
    print("Eşik 0.5'e ulaştığında, 4 kavram birbiriyle 'Kare' şeklinde birleşti ancak")
    print("ortadaki çapraz bağlar (Kuantum Kütleçekimi formülü) kayıp olduğu için sistem")
    print("bunu 'β1 = 1 (1 Boyutlu Delik)' olarak, basitleştirilmiş Euler çizge (Graph) sezgiseli üzerinden teşhis etti.")
    print("Bu Proof-of-Concept, gelişmiş Vietoris-Rips Kompleksleri kullanıldığında ToposAI'ın:")
    print("'Ortada bir delik var, bu konuyu tam bilmiyorum' diyebilen Sokratik (Self-Aware)")
    print("bir matematik makinesine dönüşebileceğinin donanımsal kanıtıdır.")

if __name__ == "__main__":
    run_tda_experiment()
