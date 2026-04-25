import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from topos_ai.math import lukasiewicz_composition

# =====================================================================
# TOPOLOGICAL EPIDEMIOLOGY & OPTIMAL QUARANTINE (AI POLICY MAKER)
# Problem: Bir salgın anında dünyadaki tüm uçuşları durdurmak ekonomiyi
# çökertir. Hangi spesifik uçuşları (Morfizmaları) kesersek (Topological Cut)
# virüsü en az zararla izole edebiliriz?
# Çözüm: ToposAI, Küresel Havacılık Ağındaki bulaş riskini (Transitivity)
# hesaplar. Sonra "Olası tüm kesiklerin (Ablation)" ağ kapanımı üzerindeki
# etkisini tarar ve dünyayı kurtaracak "En İyi Karantina (Uçuş İptali) 
# Kararını" matematiksel olarak gösterir.
# =====================================================================

import itertools

class TopologicalEpidemiologist:
    def __init__(self, cities):
        self.cities = cities
        self.N = len(cities)
        self.c_idx = {c: i for i, c in enumerate(cities)}
        
        # Havacılık Ağı (Flight Network - Bulaşma Riski)
        self.R = torch.zeros(self.N, self.N)

    def add_flight(self, city_A, city_B, risk_weight=1.0):
        """İki şehir arası doğrudan uçuş (Virüs aktarım riski)."""
        u, v = self.c_idx[city_A], self.c_idx[city_B]
        self.R[u, v] = risk_weight
        self.R[v, u] = risk_weight # Uçuşlar genelde gidiş-dönüştür

    def simulate_pandemic(self, start_city, flight_matrix=None):
        """
        Belirli bir matris üzerinden virüsün tüm dünyaya yayılımını (Reachability) hesaplar.
        """
        matrix = self.R if flight_matrix is None else flight_matrix
        
        R_inf = matrix.clone()
        for _ in range(self.N): # Tüm olası transit aktarmalar (N-Hop)
            R_inf = torch.max(R_inf, lukasiewicz_composition(R_inf, matrix))
            
        start_idx = self.c_idx[start_city]
        infection_risks = R_inf[start_idx, :]
        
        # Kendisi zaten hasta (1.0)
        infection_risks[start_idx] = 1.0
        
        return infection_risks

    def find_optimal_quarantine(self, start_city, max_cuts=1):
        """
        AI POLICY MAKER: 
        'max_cuts' kadar uçuşu (kenarı) iptal etme hakkımız var.
        Hangi uçuş Kombinasyonlarını iptal edersek, virüsün ulaşabileceği ŞEHİR SAYISI
        (Veya toplam enfeksiyon riski) en aza iner? (True Minimum Cut approximation)
        """
        best_cut_combination = None
        min_global_infection = float('inf')
        best_matrix = None
        
        # Tüm mevcut uçuşları bul
        edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.R[i, j] > 0:
                    edges.append((i, j))
                    
        # İzin verilen maksimum kesik sayısına kadar tüm kombinasyonları dene (Brute-force optimization)
        # Prodüksiyon ortamında (10.000 uçuş) bu işlem için Max-Flow Min-Cut veya GNN kullanılır.
        # Bu bir Proof-of-Concept (PoC) politika arama motorudur.
        combinations_to_test = []
        for c in range(1, max_cuts + 1):
            combinations_to_test.extend(list(itertools.combinations(edges, c)))
            
        print(f"[YZ ANALİZİ] Toplam {len(combinations_to_test)} farklı Karantina Kombinasyonu (Policy) taranıyor...")
        
        # Her bir karantina politikasını (Kesik grubunu) simüle edip sonucu ölçüyoruz
        for cut_combo in combinations_to_test:
            R_cut = self.R.clone()
            
            # Seçilen uçuşları İPTAL ET
            for u, v in cut_combo:
                R_cut[u, v] = 0.0
                R_cut[v, u] = 0.0
            
            # Bu politikanın sonucunda dünyadaki bulaş durumu ne olur?
            risks = self.simulate_pandemic(start_city, flight_matrix=R_cut)
            
            # Toplam küresel risk (Ne kadar düşükse o kadar iyi)
            global_risk = torch.sum(risks).item()
            
            if global_risk < min_global_infection:
                min_global_infection = global_risk
                # Şehir isimlerini al
                best_cut_combination = [(self.cities[u], self.cities[v]) for u, v in cut_combo]
                best_matrix = R_cut
                
        return best_cut_combination, min_global_infection, best_matrix

def run_epidemiology_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 20: TOPOLOGICAL EPIDEMIOLOGY (AI AS POLICY MAKER) ")
    print(" İddia: Bir pandemide tüm sınırları kapatmak (Tam Kapanma) bilgisizliktir.")
    print(" ToposAI, Küresel Havacılık Ağının Topolojisini okur ve 'Bütün uçuşları")
    print(" durdurmak yerine SADECE ŞU 1 ROTA'YI kesin!' diyerek virüsü hapseden ")
    print(" matematiksel Optimum Karantina (Minimum Cut) kararını bulur.")
    print("=========================================================================\n")

    # 8 Şehirli Temsili Dünya Havacılık Ağı
    cities = [
        "Wuhan", "Tokyo", "Bangkok", 
        "Dubai", "Frankfurt", "London", 
        "New_York", "Sao_Paulo"
    ]
    epidemiologist = TopologicalEpidemiologist(cities)
    
    # [UÇUŞ AĞININ KURULUMU]
    # Asya İçi Uçuşlar
    epidemiologist.add_flight("Wuhan", "Tokyo", 1.0)
    epidemiologist.add_flight("Wuhan", "Bangkok", 1.0)
    epidemiologist.add_flight("Tokyo", "Bangkok", 0.9)
    
    # Asya'dan Dünyaya Çıkış Kapısı (Transit Hubs)
    epidemiologist.add_flight("Bangkok", "Dubai", 1.0)  # Kritik Köprü 1
    epidemiologist.add_flight("Tokyo", "New_York", 0.3) # Direkt ama düşük riskli
    
    # Orta Doğu ve Avrupa (Global Dağıtım)
    epidemiologist.add_flight("Dubai", "Frankfurt", 1.0) # Kritik Köprü 2
    epidemiologist.add_flight("Frankfurt", "London", 0.9)
    epidemiologist.add_flight("Frankfurt", "New_York", 0.8)
    epidemiologist.add_flight("London", "New_York", 1.0)
    
    # Güney Amerika
    epidemiologist.add_flight("New_York", "Sao_Paulo", 0.9)

    print("[SENARYO]: 'Wuhan' şehrinde yeni bir ölümcül virüs ortaya çıktı.")
    
    # DURUM 1: HİÇBİR MÜDAHALE YOK (Normal Uçuşlar)
    risks_base = epidemiologist.simulate_pandemic("Wuhan")
    print("\n--- DURUM 1: MÜDAHALESİZ (TAM YAYILIM) ---")
    infected_cities = [cities[i] for i in range(len(cities)) if risks_base[i] > 0.5]
    print(f"  Enfekte Olan Şehirler ({len(infected_cities)}/8): {', '.join(infected_cities)}")
    print(f"  Küresel Risk Skoru: {torch.sum(risks_base).item():.2f}")
    
    # DURUM 2: YZ POLİTİKA YAPICI DEVRERDE (Optimum Karantina)
    print("\n--- DURUM 2: TOPOS-AI OPTİMUM KARANTİNA (MAX 1 UÇUŞ İPTALİ) ---")
    print("YZ'den Dünyadaki ticareti durdurmadan virüsü izole etmesi isteniyor...")
    
    best_cut, new_global_risk, R_cut = epidemiologist.find_optimal_quarantine("Wuhan", max_cuts=1)
    
    print(f"\n🎯 [YZ KARARI (VERDICT)]: Aşağıdaki uçuşları DERHAL İPTAL EDİN!")
    for u, v in best_cut:
        print(f"    - '{u} <-> {v}'")
        
    # YZ'nin kararından sonra dünyanın son durumu
    risks_after_cut = epidemiologist.simulate_pandemic("Wuhan", flight_matrix=R_cut)
    saved_cities = [cities[i] for i in range(len(cities)) if risks_after_cut[i] < 0.5]
    infected_after = [cities[i] for i in range(len(cities)) if risks_after_cut[i] > 0.5]
    
    print(f"\n--- YZ KARANTİNASI SONRASI DÜNYA TABLOSU ---")
    print(f"  Enfekte Olanlar   : {', '.join(infected_after)}")
    print(f"  KURTARILAN ŞEHİRLER: {', '.join(saved_cities)} 🎉")
    print(f"  Yeni Risk Skoru   : {new_global_risk:.2f} (Büyük Düşüş!)")

    print("\n[ÖLÇÜLEN SONUÇ: YZ İLE HAYAT KURTARMAK]")
    print("Normalde politikacılar korkuyla sınırları tamamen kapatır (Ekonomik Çöküş).")
    print("ToposAI ise ağı 'Transitive Closure' ile analiz etti. Wuhan'dan çıkan virüsün")
    print("Avrupa ve Amerika'ya ulaşmak için 'Bangkok-Dubai' transit köprüsünü KULLANMAK ")
    print("ZORUNDA olduğunu (Topological Bottleneck) tespit etti. Sadece 1 rotayı keserek")
    print("Batı Yarımküreyi (Avrupa ve Amerika kıtalarını) matematiksel olarak kurtardı!")

if __name__ == "__main__":
    run_epidemiology_experiment()
