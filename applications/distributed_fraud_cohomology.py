import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from topos_ai.cohomology import CechCohomology

# =====================================================================
# DISTRIBUTED FRAUD DETECTION (H^1 COHOMOLOGY OBSTRUCTIONS)
# Senaryo: Merkeziyetsiz bir Finans Ağında (DeFi) 4 farklı banka/ajan
# işlem yapmaktadır. Her banka sadece kendi cüzdanlarını görür.
# Klasik sistemler, "Her işlem yerel olarak kurallara uyuyorsa sistem
# sağlıklıdır" der. Ancak dolandırıcılar "Kapalı Döngüler (Kiting)"
# kurarak havadan para yaratabilirler.
# Çözüm: ToposAI, Grothendieck Topolojisi ve Çech Kohomolojisi (H^1)
# kullanarak, ajanların yerel (Local) raporlarını birleştirir. Eğer
# transferler (Edge Flows) H^1 uzayında bir "Topolojik Engel (Obstruction)"
# yaratıyorsa, makine anında "Burada Global Bir Dolandırıcılık Var!"
# diyerek (Hiçbir if/else kuralına gerek kalmadan) sahtekarlığı yakalar.
# =====================================================================

def run_cohomology_fraud_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 53: SHEAF COHOMOLOGY & H^1 OBSTRUCTION THEORY ")
    print(" İddia: Klasik Yapay Zekalar ağ analizlerinde (Graph Neural Networks)")
    print(" 'Hortumlama (Fraud)' döngülerini yakalamak için milyonlarca sahte")
    print(" işlemle (Dataset) eğitilmelidir. ToposAI ise Alexander Grothendieck'in")
    print(" 'Demet Kohomolojisi' teoremini (Lineer Cebir/SVD ile) kullanarak,")
    print(" eğitimsiz ve tek pseudo-inverse hesabıyla 'Paradoks/Dolandırıcılık' ")
    print(" döngülerini matematiksel bir Engel (H^1 Obstruction) olarak gösterir!")
    print("=========================================================================\n")

    # AĞ TOPOLOJİSİ (4 Banka, 4 Transfer Yolu)
    # Bankalar (Düğümler - V): 0=A, 1=B, 2=C, 3=D
    # Transfer Yolları (Kenarlar - E): A->B, B->C, C->D, D->A
    num_banks = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)] # Kapalı bir halka (Cycle)
    
    cohomology_engine = CechCohomology(num_nodes=num_banks, edges=edges)

    print("[SİSTEM]: 4 Bankalı Merkeziyetsiz Finans Ağı (DeFi) Kuruldu.")
    
    # ---------------------------------------------------------
    # SENARYO 1: YASAL (SAĞLIKLI) EKONOMİ
    # Bankaların bakiyeleri (Local Sections - C^0) var.
    # Bakiye farkları transferlere (Edge Flows - C^1) dönüşüyor.
    # ---------------------------------------------------------
    print("\n--- TEST 1: YASAL EKONOMİ (NO FRAUD) ---")
    # Gerçek banka bakiyeleri (Ground Truth Potentials)
    # A=100, B=150, C=120, D=200
    true_balances = torch.tensor([100.0, 150.0, 120.0, 200.0], dtype=torch.float32)
    
    # A'dan B'ye transfer (Flow) = B'nin bakiyesi - A'nın bakiyesi = +50
    # Bu akışlar D0 sınır operatörü (Boundary) ile hesaplanır.
    legal_flows = torch.matmul(cohomology_engine.d0, true_balances.view(-1, 1))
    
    # Sistem (ToposAI) bu akışları denetler
    mag1, b1_val1, obs_vec1 = cohomology_engine.compute_H1_obstruction(legal_flows)
    
    print("  > Bankaların Raporladığı Para Akışları (A->B, B->C, C->D, D->A):")
    print(f"    {legal_flows.view(-1).tolist()}")
    print(f"  > H^1 Topolojik Engel Boyutu (Obstruction Magnitude): {mag1:.4f}")
    
    if mag1 < 1e-4:
        print("  ✅ [ONAYLANDI]: Akışlar %100 Yasal (H^1 = 0). Para havadan yaratılmamış.")

    # ---------------------------------------------------------
    # SENARYO 2: HORTUMLAMA (FRAUD DÖNGÜSÜ / KITING)
    # Dolandırıcılar birbirlerine karşılıksız para atarak (Kiting)
    # sistemi kandırıyorlar. Her adımda +10$ ekliyorlar.
    # A->B(+10), B->C(+10), C->D(+10), D->A(+10)
    # Lokal olarak her banka "Bana para geldi" der. Ancak Global
    # bir potansiyel (Bakiye) yoktur. Bu bir Penrose Merdivenidir!
    # ---------------------------------------------------------
    print("\n--- TEST 2: DOLANDIRICILIK (PENROSE STAIRS / KITING) ---")
    
    fraud_flows = torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=torch.float32).view(-1, 1)
    
    # Sistem (ToposAI) bu şüpheli akışları denetler
    mag2, b1_val2, obs_vec2 = cohomology_engine.compute_H1_obstruction(fraud_flows)
    
    print("  > Bankaların Raporladığı Para Akışları (A->B, B->C, C->D, D->A):")
    print(f"    {fraud_flows.view(-1).tolist()}")
    print(f"  > H^1 Topolojik Engel Boyutu (Obstruction Magnitude): {mag2:.4f}")
    
    if mag2 > 1e-4:
        print("  🚨 [FRAUD YAKALANDI]: H^1 > 0 (Topolojik Engel Mevcut)!")
        print("     Açıklama: Bu akışlar hiçbir gerçek bakiye (C^0 Potential) ")
        print("     dağılımından üretilemez! Sistemin içinde dönen kapalı bir ")
        print("     'Hortumlama (Kiting)' veya bir Penrose Merdiveni paradoksu var.")
        print(f"     Sisteme havadan sokulan Kaçak Para Vektörü: {obs_vec2.view(-1).tolist()}")

    print("\n[BİLİMSEL DEĞERLENDİRME: COHOMOLOGICAL AI]")
    print("Derin öğrenme (Deep Learning) modelleri 'Dolandırıcılığı' bulmak")
    print("için istatistiksel anormalliklere (Outlier Detection) muhtaçtır.")
    print("Ancak yeni ve görülmemiş (Zero-Day) bir finansal illüzyonla karşılaşırlarsa")
    print("kör olurlar. ToposAI ise olaya 'Olasılık' olarak değil, 'Topolojik")
    print("Bir Uzay Geometrisi' olarak bakar.")
    print("Alexander Grothendieck'in Demet Kohomolojisini (Sheaf Cohomology) SVD")
    print("(Pseudo-Inverse) motoruyla işleten sistem, tek pseudo-inverse hesabıyla")
    print("para akışındaki 'Aşılmaz Çelişkiyi (H^1 Obstruction)' SIFIR Eğitimle")
    print("(Zero-Shot) ispatlamıştır. Bu, yapay zekanın tümevarımlı değil, saf")
    print("matematiksel bir Dedektif (Formal Reasoner) olduğunun demosudur!")

if __name__ == "__main__":
    run_cohomology_fraud_experiment()
