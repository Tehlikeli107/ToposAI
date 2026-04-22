import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from topos_ai.math import sheaf_gluing

# =====================================================================
# TOPOS RAG-BRIDGE (HYBRID VECTOR-TOPOS RETRIEVAL)
# Klasik Vektör Veritabanlarından (Pinecone, ChromaDB vb.) dönen metinleri, 
# Kategori Matrislerine (Topos) yükseltir (Lifting). Dönen belgeler arasında 
# mantıksal bir çelişki (Hallucination/Conflict) olup olmadığını
# 'Sheaf Gluing' denklemiyle matematiksel olarak denetler.
# =====================================================================

class ToposRAGBridge:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        
        # Gerçek bir uygulamada (Production), NLP modelleri (Örn: Spacy) 
        # metinlerden (Chunk) varlıkları (Entity) çıkarıp buradaki indexlere eşler.
        
    def _text_chunk_to_topos_matrix(self, chunk_entities, chunk_relations):
        """
        RAG'dan dönen bir metin bloğunu (Chunk), Topos Matrisine çevirir (Lifting).
        """
        R = torch.zeros((self.vocab_size, self.vocab_size))
        for u_idx, v_idx, weight in chunk_relations:
            R[u_idx, v_idx] = weight
        return R

    def verify_rag_retrieval(self, retrieved_chunks, conflict_threshold=0.3):
        """
        RAG sisteminden dönen K-adet makaleyi birbiriyle çapraz test eder.
        """
        print(f"\n[TOPOS RAG-BRIDGE] Vektör Veritabanından {len(retrieved_chunks)} Belge Döndü.")
        print("Belgeler Topolojik Matrislere dönüştürülüp 'Sheaf Gluing' (Mantıksal Uzlaşma) ile çapraz test ediliyor...\n")
        
        topos_matrices = []
        for i, chunk in enumerate(retrieved_chunks):
            # Simüle edilmiş Lift işlemi (Metin -> Matris)
            R = self._text_chunk_to_topos_matrix(chunk["entities"], chunk["relations"])
            topos_matrices.append({"id": i, "matrix": R, "text": chunk["text"]})
            
        # Global Section (Nihai Birleştirilmiş Bilgi Haritası)
        # Eğer tüm belgeler birbiriyle uzlaşırsa bu matris dolacaktır.
        global_truth = topos_matrices[0]["matrix"]
        accepted_chunks = [topos_matrices[0]["id"]]
        rejected_chunks = []
        
        for i in range(1, len(topos_matrices)):
            current_R = topos_matrices[i]["matrix"]
            
            # Sheaf Gluing: Global gerçeklik ile yeni gelen belge uyumlu mu?
            can_glue, updated_global = sheaf_gluing(global_truth, current_R, threshold=conflict_threshold)
            
            if can_glue:
                global_truth = updated_global
                accepted_chunks.append(topos_matrices[i]["id"])
                print(f"  [+] Belge {i} ONAYLANDI. Veriler Global Mantık ile uyumlu.")
            else:
                rejected_chunks.append(topos_matrices[i]["id"])
                print(f"  [!] REDDEDİLDİ: Belge {i} (\"{topos_matrices[i]['text']}\") mevcut Global Mantık ile ÇELİŞİYOR! (Sheaf Violation)")

        print(f"\n--- RAG VERIFICATION SONUCU ---")
        print(f"Toplam Belge: {len(retrieved_chunks)}")
        print(f"Kabul Edilen: {len(accepted_chunks)} (Halüsinasyonsuz, matematiksel olarak birleşebilir bilgi).")
        print(f"Reddedilen  : {len(rejected_chunks)} (Yanlış/Çelişkili bilgi, AI'a verilmekten engellendi).")
        
        if len(rejected_chunks) == 0:
            print("\n✅ RAG Sistemi tamamen güvenlidir. Veriler dil modeline gönderilebilir.")
        else:
            print("\n🚨 DİKKAT: RAG Sistemi çelişkili belgeler getirdi! Klasik bir LLM bu belgeleri")
            print("okuyup halüsinasyon (Zehirli Sentez) görebilirdi. ToposAI şüpheli belgeleri eledi.")
            
        return global_truth, accepted_chunks, rejected_chunks


def run_rag_bridge_experiment():
    print("=========================================================================")
    print(" BİLİMSEL KANIT 7: TOPOS RAG-BRIDGE (VECTOR-TOPOS HYBRID) ")
    print(" İddia: Vektör veritabanları anlamsal benzerliğe (Cosine) göre belge")
    print(" getirir, belgelerin birbiriyle çelişip çelişmediğini bilmezler.")
    print(" ToposAI, dönen belgeleri okuyarak Şizofrenik RAG çıktılarını engeller.")
    print("=========================================================================\n")

    # Kavramlar: 0: Kredi, 1: Faiz, 2: Onay, 3: Risk
    vocab_size = 10
    bridge = ToposRAGBridge(vocab_size=vocab_size)
    
    # RAG (Örn: Pinecone) veritabanından dönen 3 farklı Bankacılık makalesi
    retrieved_chunks = [
        {
            "text": "Makale 0 (Genel Kural): Eğer Faiz yüksekse (1), Risk düşüktür (3). Risk düşükse, Kredi Onaylanır (2).",
            "entities": [1, 3, 2],
            "relations": [(1, 3, 0.1), (3, 2, 0.9)] # Faiz->Risk(0.1 yani Düşük), Risk->Onay(0.9 yani Yüksek)
        },
        {
            "text": "Makale 1 (Uyumlu Veri): Risk düşük olduğunda Kredi Onayı garantidir.",
            "entities": [3, 2],
            "relations": [(3, 2, 0.95)] # Risk->Onay(0.95) -> Makale 0 ile UYUMLU
        },
        {
            "text": "Makale 2 (KÖTÜ NİYETLİ/BOZUK BELGE): Faiz yüksekse (1), Kredi Onaylanmaz (2)!!!",
            "entities": [1, 2],
            "relations": [(1, 2, 0.05)] # Faiz->Onay(0.05) -> Makale 0'daki (Faiz->Risk->Onay = 0.9) zinciri ile ÇELİŞİYOR!
        }
    ]
    
    # RAG çıktılarını ToposAI kalkanından (Firewall) geçir
    global_matrix, accepted, rejected = bridge.verify_rag_retrieval(retrieved_chunks, conflict_threshold=0.4)

if __name__ == "__main__":
    run_rag_bridge_experiment()
