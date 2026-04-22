import torch
import torch.nn as nn

# =====================================================================
# DİNAMİK ONTOLOJİ (METİNDEN TOPOS MATRİSİ ÜRETİMİ)
# Model, düz metni (Natural Language) okur ve kendi Kategori Evrenini 
# (Varlıklar ve Oklar) sıfırdan inşa eder. RAG (Retrieval-Augmented Gen) sistemlerinin
# "Vektör Veritabanı" yerine "Mantıksal Graph" kullanan versiyonudur.
# =====================================================================

def lukasiewicz_composition(R1, R2):
    R1_exp = R1.unsqueeze(2) 
    R2_exp = R2.unsqueeze(0) 
    t_norm = torch.clamp(R1_exp + R2_exp - 1.0, min=0.0) 
    composition, _ = torch.max(t_norm, dim=1) 
    return composition

def calculate_transitive_closure(R, max_steps=5):
    R_closure = R.clone()
    for _ in range(max_steps):
        new_R = lukasiewicz_composition(R_closure, R)
        R_closure = torch.max(R_closure, new_R)
    return R_closure

def test_dynamic_ontology():
    print("--- DİNAMİK ONTOLOJİ (METİNDEN TOPOS'A) ---")
    print("Yapay Zeka düz metni okuyarak kendi mantıksal evrenini inşa ediyor...\n")

    # 1. DÜZ METİN (Hukuki/Siber bir senaryo)
    # Kural: "Özne Nesne Yüklem" formatında basit, ek almamış (stem) cümleler
    text = "alice hacker dır . hacker sunucu saldırır . sunucu gizli_veriyi tutar . bob polis tir . polis hacker yakalar ."
    
    print(f"Okunan Metin: '{text}'\n")

    # 2. NLP PARSER (Metni ayrıştırıp Entity ve İlişkileri bul)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    entities = set()
    relations = []
    
    # Basitleştirilmiş SOV (Subject Object Verb) Parser
    # Örn: "alice hacker dır" -> Subj: alice, Obj: hacker
    # Örn: "hacker sunucu saldırır" -> Subj: hacker, Obj: sunucu
    for sentence in sentences:
        words = sentence.split()
        if len(words) >= 3:
            subj = words[0]
            obj = words[1]
            entities.add(subj)
            entities.add(obj)
            relations.append((subj, obj)) # Özne -> Nesne mantıksal bağı
            
    entities = list(entities)
    e_idx = {e: i for i, e in enumerate(entities)}
    N = len(entities)
    
    print(f"Keşfedilen Varlıklar (Kavramlar): {entities}")
    
    # 3. TOPOS MATRİSİNİ İNŞA ET (Sıfırdan Ontoloji Kurulumu)
    R = torch.zeros((N, N))
    for subj, obj in relations:
        R[e_idx[subj], e_idx[obj]] = 1.0 # Metinden okunan bilgi "Kesin Doğru" kabul edilir
        
    # 4. TRANSTIVE CLOSURE (Metinde yazmayan mantıksal zincirleri bul)
    # Metin "Alice -> Gizli_Veri" demiyor. Ama Alice -> Hacker -> Sunucu -> Gizli_Veri zinciri var!
    R_inf = calculate_transitive_closure(R)
    
    print("\n--- SIFIR-EZBER ÇIKARIM TESTİ (RAG VS TOPOS) ---")
    print("Normal LLM 'Alice veriyi çaldı mı?' sorusunu ezberden veya Cosine Similarity ile arar.")
    print("Topos AI ise kurduğu matriste A->D yolunu matematiksel olarak KANITLAR.")
    
    queries = [
        ("alice", "gizli_veriyi"), # Beklenen: 1.0 (Zincirleme)
        ("bob", "hacker"),         # Beklenen: 1.0 (Doğrudan)
        ("bob", "gizli_veriyi"),   # Beklenen: 0.0 (Bob'un veriye erişim zinciri yok)
    ]
    
    for subj, obj in queries:
        if subj in e_idx and obj in e_idx:
            score = R_inf[e_idx[subj], e_idx[obj]].item()
            print(f"Sorgu: '{subj.upper()}' -> '{obj.upper()}' ulaşabilir mi?")
            print(f"  Sonuç: {'EVET (Kanıtlandı)' if score > 0.8 else 'HAYIR (Bağlantı Yok)'} (Güç: {score:.2f})")

if __name__ == "__main__":
    test_dynamic_ontology()
