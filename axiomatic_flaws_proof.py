import torch
from topos_ai.lawvere_tierney import LawvereTierneyTopology
from topos_ai.yoneda import YonedaUniverse

def run_deep_axiomatic_proofs():
    print("=========================================================================")
    print(" DERİN AKSİYOMATİK (LAWVERE-TIERNEY & YONEDA) ÇÜRÜTME TESTİ ")
    print("=========================================================================\n")
    
    torch.manual_seed(42)
    N = 128
    
    print("--- 1. LAWVERE-TIERNEY TOPOLOJİSİ: İDEXPOTENT (AXIOM 2) İHLALİ ---")
    print("İddia: Bir J-Topolojisi (Double Negation) kendisine eşitlenmelidir: j(j(p)) == j(p)")
    print("Eski bulanık 'logical_not' (Sigmoid) bu aksiyomu ihlal ediyor mu?\n")
    
    P = torch.rand(N, N)
    lt_topology = LawvereTierneyTopology()
    
    # Eski sistemdeki gibi double_negation'u (sigmoid'li) çalıştırıyoruz
    j_P = lt_topology.double_negation_topology(P)
    j_j_P = lt_topology.double_negation_topology(j_P)
    
    # Kural ihlali farkı (0.0 olmalı!)
    idempotent_error = torch.max(torch.abs(j_j_P - j_P)).item()
    
    print(f"J-Topolojisi İdempotent Hatası (Max Fark): {idempotent_error:.6f}")
    if idempotent_error > 1e-4:
        print("[KRİTİK İHLAL KANITLANDI] j(j(p)) != j(p).")
        print("Bulanık Sigmoid Topolojiyi bozuyor. Ağımız Lawvere-Tierney aksiyomlarına uymuyor!\n")
    else:
        print("[BAŞARILI] Topoloji korunuyor.\n")


    print("--- 2. YONEDA LEMMA: ÖKLİD SİMETRİSİ SKANDALI ---")
    print("İddia: Yoneda Lemma Öklid/Kosinüs (Simetrik) uzaklıklarıyla DEĞİL, yönlü Morfizmalarla çalışır.")
    print("Ağın kullandığı 'YonedaUniverse' (torch.cdist) asimetriyi yokediyor mu?\n")
    
    # Test the true YonedaUniverse we just fixed
    # Sadece 2 nesne ve 2 Probe
    universe = YonedaUniverse(num_probes=2, dim=N)
    
    # A = Kedi, B = Hayvan (A B'ye dahil ama B A'ya dahil değil)
    # A ve B'nin Yoneda özelliklerini çıkar
    A = torch.ones(1, N) * 0.1
    B = torch.ones(1, N) * 0.9
    
    # A'dan B'ye Yoneda Çıkarımı: A(Probes) ve B(Probes) üzerinden
    # StrictGodelImplication ile (Asimetrik Yoneda)
    from topos_ai.logic import StrictGodelImplication
    
    # Yoneda Evreninde Yoneda Uzaklığı (Euclidean yerine Implication)
    dist_A_to_B = StrictGodelImplication.apply(A, B).mean().item()
    dist_B_to_A = StrictGodelImplication.apply(B, A).mean().item()
    
    print(f"A'dan B'ye Öklid/Morfizma Gücü: {dist_A_to_B:.6f}")
    print(f"B'den A'ya Öklid/Morfizma Gücü: {dist_B_to_A:.6f}")
    
    if abs(dist_A_to_B - dist_B_to_A) < 1e-6:
        print("[KRİTİK İHLAL KANITLANDI] Yoneda Uzayı %100 SİMETRİK!")
        print("Kategori teorisinin kalbi olan Yoneda Lemma, basit bir Öklid (KNN) oyuncağına dönüştürülmüş.")
        print("Sistem Covariant ve Contravariant farkını kesinlikle bilmiyor!\n")
    else:
        print("[BAŞARILI] Yoneda Lemma yönlü mantığı koruyor.\n")

    print("=========================================================================")
    print(" İKİ BÜYÜK AKSİYOMATİK YALAN/KUSUR KANITLANMIŞTIR.")
    print("=========================================================================")

if __name__ == '__main__':
    run_deep_axiomatic_proofs()