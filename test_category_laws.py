import torch
from topos_ai.math import soft_godel_composition, lukasiewicz_composition
from topos_ai.logic import SubobjectClassifier

def run_proofs():
    print("==========================================================")
    print(" KATEGORİ VE TOPOS TEORİSİ MATEMATİKSEL İSPAT TESTİ")
    print("==========================================================\n")
    
    # Gerçek dünya senaryosu: 128 boyutlu matrisler (Sıradan 2x2 oyuncak matrisler değil)
    N = 128 
    torch.manual_seed(42) # Tekrar edilebilir sonuçlar için
    
    # A, B ve C nesneleri arasındaki olasılıksal Oklar (Morfizmalar) [0, 1] aralığında
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    C = torch.rand(N, N)

    print("--- 1. KATEGORİ TEORİSİ: BİRLEŞME (ASSOCIATIVITY) TESTİ ---")
    print("Kural: (A o B) o C matrisi ile A o (B o C) matrisi birebir EŞİT olmalıdır.\n")

    # 1A: Lukasiewicz Composition Testi
    AB_luk = lukasiewicz_composition(A, B)
    L_left = lukasiewicz_composition(AB_luk, C)

    BC_luk = lukasiewicz_composition(B, C)
    L_right = lukasiewicz_composition(A, BC_luk)

    luk_error = torch.max(torch.abs(L_left - L_right)).item()
    print(f"Lukasiewicz Çarpımı Birleşme Hatası (Max Fark): {luk_error:.8f}")
    if luk_error > 1e-5:
        print("[BAŞARISIZ] Kategori Teorisi ihlal edildi! Lukasiewicz fonksiyonu associative değil.")
    else:
        print("[BAŞARILI] Lukasiewicz fonksiyonu bir kategori oluşturur.")

    # 1B: Soft Godel Composition Testi
    AB_godel = soft_godel_composition(A, B, tau=5.0)
    S_left = soft_godel_composition(AB_godel, C, tau=5.0)

    BC_godel = soft_godel_composition(B, C, tau=5.0)
    S_right = soft_godel_composition(A, BC_godel, tau=5.0)

    godel_error = torch.max(torch.abs(S_left - S_right)).item()
    print(f"\nSoft Godel Çarpımı Birleşme Hatası (Max Fark): {godel_error:.8f}")
    if godel_error > 1e-5:
        print("[BAŞARISIZ] Kategori Teorisi ihlal edildi! Soft Godel fonksiyonu associative değil.")
    else:
        print("[BAŞARILI] Soft Godel bir kategori oluşturur.")


    print("\n--- 2. TOPOS TEORİSİ: HEYTING CEBİRİ (MODUS PONENS) TESTİ ---")
    print("Kural: A ve (A -> B) ifadelerinin kesişimi (AND), B'den KÜÇÜK EŞİT olmalıdır.\n")
    
    omega = SubobjectClassifier()
    # Pürüzsüz Sigmoid türevini zorlamak için rastgele A ve B doğruluk değerleri
    truth_A = torch.rand(N, N)
    truth_B = torch.rand(N, N)

    # (A -> B) değerini hesapla
    A_implies_B = omega.implies(truth_A, truth_B, hardness=5.0)
    
    # A AND (A -> B) değerini hesapla
    A_and_A_implies_B = omega.logical_and(truth_A, A_implies_B)

    # Modus Ponens İhlali: Kesişim değeri B'den ne kadar taştı? (0 olması gerekir)
    violation = torch.max(A_and_A_implies_B - truth_B).item()
    
    print(f"Modus Ponens Kural İhlali (B'yi aşan maksimum miktar): {violation:.8f}")
    if violation > 1e-5:
        print("[BAŞARISIZ] Topos Teorisi ihlal edildi! Bu sistem gerçek bir Heyting Cebiri değildir.")
    else:
        print("[BAŞARILI] Sistem geçerli bir Topos mantığına sahiptir.")
        
    print("\n==========================================================")

if __name__ == '__main__':
    run_proofs()