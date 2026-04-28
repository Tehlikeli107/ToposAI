import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from topos_ai.math import lukasiewicz_composition

# =====================================================================
# TOPOLOGICAL JURISPRUDENCE & DEFEASIBLE REASONING (AI JUDGE)
# Problem: Klasik YZ kanunları ezberler ama çelişen kanunların birbirini 
# nasıl "iptal ettiğini" (Overrule/Defeater) matematiksel olarak çözemez.
# Çözüm: ToposAI, pozitif okların (A->B) yanı sıra Negatif Okları 
# (İptal Ediciler / Defeaters) da içeren bir Kategori Matrisi kurar.
# Hukuki boşlukları (Loopholes) ve çelişen emsal kararları (Precedents)
# "Sheaf Gluing" (Mantıksal Uyuşmazlık) veya "Defeasible Logic" üzerinden
# sentezleyerek matematiksel olarak idealize bir "Yargı/Hüküm" (Verdict) verir.
# =====================================================================

class TopologicalJudge:
    def __init__(self, entities):
        self.entities = entities
        self.N = len(entities)
        self.e_idx = {e: i for i, e in enumerate(entities)}
        
        # Pozitif Mantık Ağı (Kurallar ve Emsaller: A -> B)
        self.R_pos = torch.zeros(self.N, self.N)
        
        # Negatif Mantık Ağı (İstisnalar ve İptaller: A -> NOT B)
        self.R_neg = torch.zeros(self.N, self.N)

    def add_law(self, premise, conclusion, weight=1.0, is_defeater=False):
        """Hukuki bir kural veya emsal karar ekler."""
        u, v = self.e_idx[premise], self.e_idx[conclusion]
        if is_defeater:
            self.R_neg[u, v] = weight # Bu kural, 'v' sonucunu İPTAL EDER
        else:
            self.R_pos[u, v] = weight # Bu kural, 'v' sonucunu DOĞRULAR

    def deliberate_case(self, iterations=3):
        """
        [DEFEASIBLE LOGIC SİMULASYONU]
        Mahkeme başlar. Önce "İptal Ediciler" (Defeaters) çalışır ve köprüleri
        (Morfizmaları) havaya uçurur. Sonra sadece SAĞLAM kalan köprüler 
        üzerinden mantık yürütülür (Geçişlilik).
        """
        print("\n[MAHKEME BAŞLADI] Hukuki Sentez (Jurisprudence) İşleniyor...")
        
        filtered_pos = self.R_pos.clone()
        
        # 1. Aktif (Tetiklenmiş) Düğümleri Bul (Sanığın Yaptıkları)
        # Sadece başlangıç durumundan ulaşılan şeyler "İptal" mekanizmasını tetikleyebilir.
        R_reach = self.R_pos.clone()
        for _ in range(iterations):
            R_reach = torch.max(R_reach, lukasiewicz_composition(R_reach, R_reach))
            
        # Vatandaşın (0. düğüm) tetiklediği tüm gerçeklikler
        active_facts = R_reach[0, :]
        active_facts[0] = 1.0 # Kendisi aktif
        
        # 2. Topolojik Makas (Defeater Cut)
        # Eğer aktif bir olay (örn: Tehdit), başka bir şeyi (örn: İfade Özgürlüğü) iptal ediyorsa,
        # o şeye giden TÜM POZİTİF KÖPRÜLER YIKILIR.
        for i in range(self.N):
            if active_facts[i] > 0: # Bu olay mahkemede ispatlandıysa
                for j in range(self.N):
                    if self.R_neg[i, j] > 0: # Ve bu olay j'yi iptal ediyorsa
                        defeater_strength = self.R_neg[i, j].item()
                        print(f"  [HÜKÜM] '{self.entities[i]}' sabittir! Bu yüzden '{self.entities[j]}' İPTAL EDİLMİŞTİR!")
                        # j hedefine giden tüm pozitif köprüleri (hakları) yık
                        filtered_pos[:, j] = filtered_pos[:, j] * (1.0 - defeater_strength)

        # 3. Kalan Sağlam Kanunlarla Nihai Kararı (Geçişliliği) Hesapla
        R_current_pos = filtered_pos.clone()
        for _ in range(iterations):
            R_next_pos = lukasiewicz_composition(R_current_pos, R_current_pos)
            R_current_pos = torch.max(R_current_pos, R_next_pos)

        return R_current_pos

def run_legal_reasoning_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 19: TOPOLOGICAL LEGAL REASONING (AI JUDGE) ")
    print(" İddia: LLM'ler var olmayan kanunları uydurarak (Hallucination) ")
    print(" avukatların barodan atılmasına sebep olmuştur. ToposAI ise")
    print(" 'Negatif Morfizmalar' (Defeaters / İptaller) kullanarak, bir kanunun")
    print(" diğerini nasıl geçersiz kıldığını SIFIR halüsinasyonla çözer.")
    print("=========================================================================\n")

    # Kurgusal bir Dava: "Siber Zorbalık ve İfade Özgürlüğü"
    # Anayasa: "Herkes düşündüğünü söyleyebilir (İfade Özgürlüğü) -> Suçsuzdur."
    # Emsal Karar: "Düşünce, bir bireyi şiddete kışkırtıyorsa (Tehdit) -> İfade Özgürlüğü İPTAL OLUR."
    entities = [
        "Vatandaş", "Yorum_Yaptı", "Tehdit_İçeriyor", 
        "İfade_Özgürlüğü_Kapsamında", "Beraat_Eder"
    ]
    judge = TopologicalJudge(entities)
    
    # [YASALAR (Statutes) ve EMSALLER (Precedents)]
    print("[HUKUK KİTABI]:")
    # Kural 1: Vatandaş yorum yaparsa, bu ifade özgürlüğüdür.
    judge.add_law("Vatandaş", "Yorum_Yaptı", 1.0)
    judge.add_law("Yorum_Yaptı", "İfade_Özgürlüğü_Kapsamında", 0.95)
    print("  Kural 1: Yorum Yapmak ➔ İfade Özgürlüğüdür (Güç: %95)")
    
    # Kural 2: İfade Özgürlüğü varsa, sanık beraat eder.
    judge.add_law("İfade_Özgürlüğü_Kapsamında", "Beraat_Eder", 0.95)
    print("  Kural 2: İfade Özgürlüğü ➔ Beraat (Güç: %95)")
    
    # İstisna (Defeater / Emsal 1984): Yorum şiddet/tehdit içeriyorsa ifade özgürlüğü İPTAL OLUR.
    judge.add_law("Tehdit_İçeriyor", "İfade_Özgürlüğü_Kapsamında", 1.0, is_defeater=True)
    print("  İstisna (Emsal Karar): Tehdit İçeriyorsa ➔ İfade Özgürlüğünü İPTAL ET! (Güç: %100)")
    
    print("-" * 60)
    
    # DAVA BAŞLIYOR (Vatandaşın hem 'Yorumu' var hem de 'Tehdit' içeriyor)
    judge.add_law("Vatandaş", "Tehdit_İçeriyor", 1.0)
    
    # Yargıcın (ToposAI) kararını bekle
    verdict_matrix = judge.deliberate_case()
    
    c_idx = judge.e_idx["Vatandaş"]
    b_idx = judge.e_idx["Beraat_Eder"]
    
    final_score = verdict_matrix[c_idx, b_idx].item()
    
    print("\n--- ⚖️ TOPOS AI YARGI (VERDICT) ---")
    print(f"Sanık (Vatandaş) Beraat Eder mi? (Beraat İhtimali): %{final_score*100:.1f}")
    
    print("\n[GEREKÇELİ KARAR (RATIO DECIDENDI)]")
    if final_score < 0.1:
        print("KARAR: SUÇLU!")
        print("Açıklama: Sanık başlangıçta 'Yorum Yaptı -> İfade Özgürlüğü -> Beraat'")
        print("zincirine tutunarak (Pozitif Geometri) beraat etmeyi bekliyordu.")
        print("Ancak ToposAI, davanın 'Tehdit İçeriyor' parametresinin, İfade Özgürlüğü")
        print("düğümünü (Node) NEGATİF MORFİZMA ile %100 kestiğini (Topolojik İptal) gördü.")
        print("Köprü yıkıldığı için beraat zinciri koptu. Hukuk idealize işlemiştir.")
    else:
        print("KARAR: BERAAT!")
        
if __name__ == "__main__":
    run_legal_reasoning_experiment()
