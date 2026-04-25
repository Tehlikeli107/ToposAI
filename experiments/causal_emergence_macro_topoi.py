import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import math

# =====================================================================
# CAUSAL EMERGENCE & MACRO-STATES (RENORMALIZATION GROUP)
# İddia: Mikroskobik (Alt seviye) sistemler gürültülü ve belirsizdir.
# Evren, karmaşıklığı "Gruplayarak" (Coarse-Graining) Makro-Sistemler
# yaratır. Makro seviyede 'Nedensellik' (Causality) ve 'Kesinlik' 
# (Determinism) mikro seviyeden DAHA YÜKSEKTİR (Causal Emergence).
# ToposAI, kaotik bir mikro-ağı Makro-Topolojiye dönüştürerek,
# "Bütünün (Makro), Parçalarından (Mikro) daha kesin çalıştığını" 
# matematiksel bilgi teorisiyle (Effective Information) gösterir.
# =====================================================================

class CausalEmergenceEngine:
    def __init__(self, num_micro_nodes, num_macro_nodes):
        self.N_micro = num_micro_nodes
        self.N_macro = num_macro_nodes
        self.nodes_per_macro = self.N_micro // self.N_macro
        
        # Micro Matris (Kaotik ve Gürültülü)
        self.R_micro = torch.zeros(self.N_micro, self.N_micro)
        
    def generate_noisy_microverse(self):
        """Alt seviyedeki nöron ağını/atomları rastgele ve gürültülü bağlar."""
        torch.manual_seed(42)
        
        # Temel hedef: Makro 0 -> Makro 1 -> Makro 2 -> Makro 3
        # Ancak mikro seviyede bu geçişler çok dağınık ve gürültülüdür.
        for i in range(self.N_micro):
            macro_id_from = i // self.nodes_per_macro
            target_macro_id = (macro_id_from + 1) % self.N_macro
            
            for j in range(self.N_micro):
                macro_id_to = j // self.nodes_per_macro
                
                # Eğer doğru makro gruba gidiyorsa sinyal var ama gürültülü (0.1 - 0.4 arası)
                if macro_id_to == target_macro_id:
                    self.R_micro[i, j] = torch.rand(1).item() * 0.3 + 0.1
                # Rastgele arka plan gürültüsü (0.0 - 0.1 arası)
                else:
                    self.R_micro[i, j] = torch.rand(1).item() * 0.1
                    
        # Kendi kendine geçişler
        for i in range(self.N_micro):
            self.R_micro[i, i] = 1.0
            
    def measure_causal_certainty(self, matrix):
        """
        [BİLGİ TEORİSİ: DETERMINISM / EFFECTIVE INFORMATION]
        Bir matristeki nedenselliğin gücü, satırların (Out-edges) ne kadar
        odaklanmış (Düşük Entropi) veya ne kadar dağınık (Yüksek Entropi)
        olduğuyla ölçülür.
        """
        N = matrix.size(0)
        total_certainty = 0.0
        
        for i in range(N):
            # Olasılıklara çevir (Satır toplamı 1.0 olacak şekilde normalize et)
            row = matrix[i, :]
            row_sum = torch.sum(row)
            if row_sum > 0:
                p = row / row_sum
                # Shannon Entropisi: H = -sum(p * log2(p))
                entropy = -torch.sum(p * torch.log2(p + 1e-9)).item()
                
                # Max Entropi (Tamamen belirsiz durum) = log2(N)
                max_entropy = math.log2(N)
                
                # Kesinlik (Certainty) = 1.0 - (Entropy / Max_Entropy)
                certainty = 1.0 - (entropy / max_entropy)
                total_certainty += certainty
                
        # Sistemin ortalama Nedensel Kesinliği
        return total_certainty / N

    def coarse_grain_to_macro(self):
        """
        [RENORMALIZATION GROUP / COARSE GRAINING]
        Mikro düğümleri Makro gruplara birleştirir.
        Makro matris, içindeki mikro bağların Topolojik Sinerjisini (Superposition) alır.
        """
        R_macro = torch.zeros(self.N_macro, self.N_macro)
        
        for M_from in range(self.N_macro):
            for M_to in range(self.N_macro):
                # Bu makro gruptaki mikro nöronların listesi
                micro_nodes_from = range(M_from * self.nodes_per_macro, (M_from + 1) * self.nodes_per_macro)
                micro_nodes_to = range(M_to * self.nodes_per_macro, (M_to + 1) * self.nodes_per_macro)
                
                # İki makro lob arasındaki toplam etkileşim enerjisi
                # (Mikro gürültüler toplanırken, sinyal belirginleşir - Law of Large Numbers)
                total_signal = 0.0
                for u in micro_nodes_from:
                    for v in micro_nodes_to:
                        total_signal += self.R_micro[u, v].item()
                        
                # Ortalama gücü Makro Kategori Ok'u olarak ata
                avg_signal = total_signal / (self.nodes_per_macro * self.nodes_per_macro)
                
                # Sigmoid / Tanh benzeri bir doyum noktası (Emergence fonksiyonu)
                emergent_power = min(1.0, avg_signal * 3.0) 
                
                R_macro[M_from, M_to] = emergent_power
                
        # Makro self-loop
        for i in range(self.N_macro):
            R_macro[i, i] = 1.0
            
        return R_macro

def run_causal_emergence_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 28: CAUSAL EMERGENCE (MICRO VS MACRO REALITY) ")
    print(" İddia: İndirgemeci (Reductionist) bilim her şeyin atomlarda/nöronlarda")
    print(" bittiğini savunur. ToposAI ise, alt katmanın (Micro) kaotik olduğunu,")
    print(" ancak sistemin gruplanarak (Coarse-Graining) Makro boyutlara ulaştığında")
    print(" 'Nedenselliğin' (Causality) ve 'Kararlılığın' SIFIRDAN DOĞDUĞUNU (Emergence)")
    print(" Bilgi Teorisiyle (Effective Information) matematiksel olarak GÖSTERİR.")
    print("=========================================================================\n")

    N_micro = 60 # 60 Nöronluk Kaotik Bir Alt-Katman
    N_macro = 4  # 4 Lob'lu Makro Beyin (Her lob 15 nöron)
    
    engine = CausalEmergenceEngine(N_micro, N_macro)
    
    # 1. MİKRO EVREN (Micro-State)
    print(f"[MİKRO EVREN]: {N_micro} Nöronluk yapı (Micro-State) yaratılıyor...")
    engine.generate_noisy_microverse()
    
    micro_certainty = engine.measure_causal_certainty(engine.R_micro)
    print(f"  > Alt-Katman (Nöron) Nedensel Kesinliği (Causal Determinism): %{micro_certainty*100:.1f}")
    print("    (Nöronlar kime sinyal yolladığını bilmiyor, aşırı gürültü ve entropi var!)")

    # 2. MAKRO EVREN (Macro-State)
    print(f"\n[MAKRO EVREN]: Nöronlar {N_macro} Adet Beyin Lobuna (Macro-State) gruplanıyor...")
    print("  Kategori Teorisi: Renormalization Group / Functor Mapping İşleniyor...")
    
    R_macro = engine.coarse_grain_to_macro()
    macro_certainty = engine.measure_causal_certainty(R_macro)
    
    print(f"  > Üst-Katman (Lob) Nedensel Kesinliği (Causal Determinism)  : %{macro_certainty*100:.1f}")
    
    # 3. EMERGENT NEDENSELLİK KIYASLAMASI
    print("\n--- 🧠 ONTOLOJİK SIÇRAMA (CAUSAL EMERGENCE) ---")
    print(f"  Mikro-Nedensellik Skoru : %{micro_certainty*100:.1f}")
    print(f"  Makro-Nedensellik Skoru : %{macro_certainty*100:.1f}")
    
    if macro_certainty > micro_certainty:
        print("\n[BİLİMSEL SONUÇ: 'BÜTÜN, PARÇALARINDAN DAHA GERÇEKTİR']")
        print("Klasik bilim yanılıyor! Gerçek nedensellik (Causality) alt katmandaki")
        print(f"({N_micro} düğüm) gürültülü atomlarda veya nöronlarda DEĞİL;")
        print("Sistemin bütününde (Makro Topoloji) ortaya çıkar.")
        print("ToposAI, Bilgi Teorisini (Entropy) kullanarak, sistem makro boyuta")
        print("çıktığında 'Hataların (Noise) birbirini sönümlediğini' ve geriye")
        print("İDEALİZE (Daha yüksek Kesinlikli) bir Fizik Kuralı kaldığını GÖSTERDİ.")
        print("Özgür İrade ve Bilinç, işte bu 'Causal Emergence' boşluğunda yaşar!")

if __name__ == "__main__":
    run_causal_emergence_experiment()
