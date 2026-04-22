import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import sys

# Windows konsolunda emoji çökmesini engelle (P3 Fix)
if sys.stdout.encoding.lower() != 'utf-8':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

# Framework'ten import et (Kopya kodları sildik - DRY Principle)
from topos_ai.math import transitive_closure

# =====================================================================
# DİNAMİK ONTOLOJİ (METİNDEN TOPOS MATRİSİ ÜRETİMİ)
# Model, düz metni (Natural Language) okur ve kendi Kategori Evrenini 
# (Varlıklar ve Oklar) sıfırdan inşa eder. RAG (Retrieval-Augmented Gen) sistemlerinin
# "Vektör Veritabanı" yerine "Mantıksal Graph" kullanan versiyonudur.
# =====================================================================

def test_dynamic_ontology():
    print("--- DİNAMİK ONTOLOJİ (METİNDEN TOPOS'A) ---")
    print("Yapay Zeka düz metni okuyarak kendi mantıksal evrenini inşa ediyor...\n")

    # 1. DÜZ METİN (Hukuki/Siber bir senaryo)
    # Kural: "Özne Nesne Yüklem" formatında basit, ek almamış (stem) cümleler
    text = "alice hacker dır . hacker sunucu saldırır . sunucu gizli_veriyi tutar . bob polis tir . polis hacker yakalar . charlie masum dur . charlie kedi sever ."
    
    print(f"Okunan Metin: '{text}'\n")

    # 2. NLP PARSER (Metni ayrıştırıp Entity ve İlişkileri bul)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    entities = set()
    relations = []
    
    # Basitleştirilmiş SOV (Subject Object Verb) Parser
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
    R_inf = transitive_closure(R)
    
    print("\n--- SIFIR-EZBER ÇIKARIM TESTİ (RAG VS TOPOS) ---")
    print("Normal LLM 'Alice veriyi çaldı mı?' sorusunu ezberden veya Cosine Similarity ile arar.")
    print("Topos AI ise kurduğu matriste A->D yolunu matematiksel olarak KANITLAR.")
    
    queries = [
        ("alice", "gizli_veriyi"), # Beklenen: 1.0 (Zincirleme bağlantı)
        ("bob", "hacker"),         # Beklenen: 1.0 (Doğrudan bağlantı)
        ("bob", "gizli_veriyi"),   # Beklenen: 1.0 (Bob -> Polis -> Hacker -> Sunucu -> Veri zinciri mevcut!)
        ("charlie", "gizli_veriyi")# Beklenen: 0.0 (Charlie'nin hiçbir şekilde sisteme bağı yok)
    ]
    
    for subj, obj in queries:
        if subj in e_idx and obj in e_idx:
            score = R_inf[e_idx[subj], e_idx[obj]].item()
            print(f"Sorgu: '{subj.upper()}' -> '{obj.upper()}' ulaşabilir mi?")
            print(f"  Sonuç: {'EVET (Kanıtlandı)' if score > 0.8 else 'HAYIR (Bağlantı Yok)'} (Güç: {score:.2f})")

if __name__ == "__main__":
    test_dynamic_ontology()
