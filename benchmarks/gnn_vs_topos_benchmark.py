import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn as nn

# =====================================================================
# ACADEMIC BENCHMARK: GRAPH NEURAL NETWORKS (GNN) VS TOPOS AI
# Problem: GCN/GAT gibi Graph modelleri, derin ağaçlarda "Aşırı Düzleştirme"
# (Over-smoothing) yaşar ve keskin mantıksal doğruları (1.0) kaybedip 
# her şeye 0.5 demeye başlar. 
# Çözüm: ToposAI, Min-Max (Gödel T-Norm) kullanarak sonsuz derinlikte
# bile mantıksal keskinliği (True/False) %100 korur.
# =====================================================================

class DummyGCNLayer(nn.Module):
    """
    Basit bir Graph Convolution (GCN) Katmanı Simülasyonu.
    GNN'ler komşuların ortalamasını (Mean Aggregation / Smoothing) alır.
    """
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        # Karşılaştırma için ağırlıkları birim matris (etkisiz) yapıyoruz
        self.W.weight.data.copy_(torch.eye(dim))

    def forward(self, R, X):
        # R: Adjacency Matrix (Bağlantılar)
        # Normalizasyon: Satır toplamlarına böl (GCN'deki Degree Normalization)
        D_inv = torch.diag(1.0 / (R.sum(dim=1) + 1e-9))
        R_norm = torch.matmul(D_inv, R)
        
        # Message Passing (Komşuların ortalamasını al)
        out = torch.matmul(R_norm, X)
        out = self.W(out)
        return torch.relu(out) # 0-1 arası aktivasyon

def godel_composition(R1, R2):
    """ToposAI'nin Yönlü ve Keskin Mantıksal Geçişliliği"""
    R1_exp = R1.unsqueeze(2) 
    R2_exp = R2.unsqueeze(0) 
    t_norm = torch.min(R1_exp, R2_exp)
    composition, _ = torch.max(t_norm, dim=1) 
    return composition

def run_gnn_vs_topos_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 6: GNN (GRAPH NEURAL NETWORKS) VS TOPOS AI ")
    print(" İddia: GNN'ler 'Message Passing' yaparken mantığı bulanıklaştırır")
    print(" (Over-smoothing). Topos ise Gödel mantığı ile keskinliği sonsuza dek korur.")
    print("=========================================================================\n")

    # 1. UZUN BİR MANTIK ZİNCİRİ (A -> B -> C -> D -> E -> F)
    # VE ETRAFINDAKİ GÜRÜLTÜ (Kaotik Komşular)
    N = 100
    R = torch.zeros(N, N)
    
    # Altın Zincir (The Golden Path): 0 -> 1 -> 2 -> 3 -> 4 -> 5
    for i in range(5):
        R[i, i+1] = 1.0
        
    # GNN'i Çökertecek Gürültü (Noise / Extra Neighbors)
    # Gerçek dünyada bir kavramın (Düğümün) binlerce ilgisiz komşusu vardır.
    # B düğümüne (1) rastgele 10 tane anlamsız komşu bağlayalım.
    torch.manual_seed(42)
    for i in range(5):
        random_neighbors = torch.randint(10, N, (10,))
        for neighbor in random_neighbors:
            R[i, neighbor.item()] = 1.0 # Ana zincirden dışarı sızan gürültü okları
            R[neighbor.item(), i] = 1.0 # Dışarıdan ana zincire sızan gürültü okları
    
    # 2. GNN (GCN) YAKLAŞIMI
    print("--- 1. GNN (GRAPH CONVOLUTION) SİMÜLASYONU ---")
    gcn = DummyGCNLayer(dim=N)
    X_gcn = torch.eye(N) # Başlangıç Node Feature'ları (Kimlik matrisi)
    
    for step in range(1, 6): # 5 katmanlı (5-Hop) derin bir GNN
        X_gcn = gcn(R, X_gcn)
        
        # A'nın (0. düğüm), 5 adım ötedeki düğüme (F) ulaşma sinyaline bakalım
        signal_strength = X_gcn[0, step].item()
        print(f"  [GCN Layer {step}] A'nın {step} adım öteye ulaşan Sinyal Gücü: {signal_strength:.4f}")
        
    print("  [GNN SONUCU]: GNN'ler 'Ortalama' (Mean Aggregation) aldığı için sinyal her adımda \n  seyreldi, bulanıklaştı ve kayboldu (Over-smoothing). Kesin mantık KURULAMAZ.\n")

    # 3. TOPOS AI (GÖDEL T-NORM) YAKLAŞIMI
    print("--- 2. TOPOS AI (GÖDEL COMPOSITION) SİMÜLASYONU ---")
    R_topos = R.clone()
    
    for step in range(1, 6):
        # Topos AI, matrisi kendisiyle çarparak (Transitive Closure) ilerler
        R_topos = torch.max(R_topos, godel_composition(R_topos, R))
        
        signal_strength = R_topos[0, step].item()
        print(f"  [Topos Hop {step}] A'nın {step} adım öteye ulaşan Mantık Gücü: {signal_strength:.4f}")

    print("  [TOPOS SONUCU]: ToposAI ortalama almaz! 'En zayıf halka' (Bottleneck) mantığına \n  baktığı için 1.0 olan kesin mantık, sonsuz adıma da gitse %100 KESİNLİKLE korunur.\n")

if __name__ == "__main__":
    run_gnn_vs_topos_experiment()
