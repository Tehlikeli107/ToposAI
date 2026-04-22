import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn as nn
import numpy as np

# =====================================================================
# TOPOS MULTI-AGENT SWARM (ÇOKLU-AJAN DEMET UZLAŞMASI)
# 3 Farklı Uzman (Agent), bir vakayı kendi evrenlerinden (Local Topoi) 
# değerlendirir. Sistemi "Sheaf Gluing" (Demet Yapıştırıcısı) yönetir.
# Çelişen Ajan sistemden atılır, Uyuşan Ajanlar "Global Gerçekliği" kurar.
# =====================================================================

def sheaf_gluing_condition(truth_A, truth_B, threshold=0.2):
    certainty_A = torch.abs(truth_A - 0.5) * 2.0
    certainty_B = torch.abs(truth_B - 0.5) * 2.0
    overlap = certainty_A * certainty_B
    disagreement = torch.abs(truth_A - truth_B)
    conflict_score = torch.sum(overlap * disagreement).item()
    
    if conflict_score > threshold:
        return False, conflict_score, None
    return True, conflict_score, torch.max(truth_A, truth_B)

def test_topos_multi_agent_swarm():
    print("--- TOPOS MULTI-AGENT SWARM (UZLAŞMA KONSEYİ) ---")
    print("3 Farklı Yapay Zeka Uzmanı (Ajan) bir hastayı tartışıyor...\n")

    # Kavramlar (Entities)
    entities = ["GÖĞÜS_AĞRISI", "KALP_KRİZİ", "REFLÜ", "YÜKSEK_TROPONİN"]
    idx = {e: i for i, e in enumerate(entities)}
    N = len(entities)
    
    # 3 AJANIN KENDİ "LOCAL TRUTH" (Yerel Evren) BİLGİLERİ
    # Başlangıçta hepsi her şeye 0.5 (Bilmiyorum) der.
    Agent1 = torch.full((N, N), 0.5) # Kardiyolog
    Agent2 = torch.full((N, N), 0.5) # Gastroenterolog (Mideci)
    Agent3 = torch.full((N, N), 0.5) # Laboratuvar Yapay Zekası

    # AJAN 1: Kardiyolog (Göğüs ağrısı Kalp Krizidir der)
    Agent1[idx["GÖĞÜS_AĞRISI"], idx["KALP_KRİZİ"]] = 0.95
    
    # AJAN 2: Mideci (Göğüs ağrısı Reflüdür der. Kalp krizini inkar eder - Halüsinasyon!)
    Agent2[idx["GÖĞÜS_AĞRISI"], idx["REFLÜ"]] = 0.90
    Agent2[idx["GÖĞÜS_AĞRISI"], idx["KALP_KRİZİ"]] = 0.10 # ÇELİŞKİ BURADA!
    
    # AJAN 3: Laboratuvar (Göğüs ağrısını bilmez, ama Troponin yüksekse Kalp krizidir der)
    Agent3[idx["YÜKSEK_TROPONİN"], idx["KALP_KRİZİ"]] = 0.99
    # Lab AI ayrıca hastanın Göğüs Ağrısı ile geldiğini de raporlardan onaylar.
    Agent3[idx["GÖĞÜS_AĞRISI"], idx["KALP_KRİZİ"]] = 0.85
    
    agents = {"Ajan 1 (Kardiyolog)": Agent1, "Ajan 2 (Mideci)": Agent2, "Ajan 3 (Laboratuvar AI)": Agent3}
    
    print("Topos Demet Kuramı (Sheaf Theory) Ajanları Çapraz Test Ediyor...")
    
    # Çapraz Test (Pairwise Sheaf Gluing)
    agent_names = list(agents.keys())
    valid_alliances = []
    
    for i in range(len(agent_names)):
        for j in range(i+1, len(agent_names)):
            nameA, nameB = agent_names[i], agent_names[j]
            R_A, R_B = agents[nameA], agents[nameB]
            
            can_glue, conflict, glued_R = sheaf_gluing_condition(R_A, R_B)
            print(f"[{nameA}] vs [{nameB}] -> Çelişki Skoru: {conflict:.2f}")
            
            if can_glue:
                print("  => UZLAŞMA SAĞLANDI! (Sheaf Yapıştırıldı)")
                valid_alliances.append((nameA, nameB, glued_R))
            else:
                print("  => ÇELİŞKİ! (Mideci Ajanı, Kardiyoloji bulgularını reddediyor. İZOLE EDİLDİ.)")
                
    print("\n--- NİHAİ KONSENSÜS (GLOBAL SECTION) ---")
    if valid_alliances:
        # Uzlaşan ajanların bilgilerini birleştir (Sentezle)
        best_alliance = valid_alliances[0] # İlk başarılı uzlaşmayı alalım
        global_truth = best_alliance[2]
        
        print(f"Konsensüs Konseyi ({best_alliance[0]} ve {best_alliance[1]}) Teşhisi Doğruladı:")
        for e1 in entities:
            for e2 in entities:
                val = global_truth[idx[e1], idx[e2]].item()
                if val > 0.8:
                    print(f"  KANITLANMIŞ GERÇEK: {e1} ===> {e2} (Doğruluk: {val:.2f})")
    else:
        print("Ajanlar hiçbir konuda uzlaşamadı. Sistem teşhisi reddediyor (Sıfır Halüsinasyon).")

if __name__ == "__main__":
    test_topos_multi_agent_swarm()
