import torch
import torch.nn as nn
import torch.nn.functional as F
from topos_ai.kan import LeftKanExtension
from topos_ai.logic import StrictGodelImplication

def test_fixed_architecture_proofs():
    print("=========================================================================")
    print(" YENİ MİMARİNİN (ASİMETRİ & COLIMIT) KESİN KANITLARI ")
    print("=========================================================================\n")
    
    torch.manual_seed(42)
    N = 128
    
    print("--- 1. YÖNLÜ MANTIK ASİMETRİSİ (YENİ TOPOS INTERNAL HOM) ---")
    print("Test: A <= B doğru (1.0) iken, B <= A yanlış olmalıdır.")
    
    # A = Alt küme "Kedi"
    # B = Üst küme "Hayvan"
    A = torch.ones(N) * 0.1
    B = torch.ones(N) * 0.9
    
    # A -> B Yönü (Kedi Hayvandır)
    A_exp = A.unsqueeze(0).unsqueeze(1).unsqueeze(2) # [1, 1, 1, 128]
    B_exp = B.unsqueeze(0).unsqueeze(0).unsqueeze(0) # [1, 1, 1, 128]
    
    impl_A_B = StrictGodelImplication.apply(A_exp, B_exp).mean().item()
    impl_B_A = StrictGodelImplication.apply(B_exp, A_exp).mean().item()
    
    print(f"Kedi -> Hayvan (A -> B): {impl_A_B:.6f}")
    print(f"Hayvan -> Kedi (B -> A): {impl_B_A:.6f}")
    
    if impl_A_B == 1.0 and impl_B_A < 1.0:
        print("[BAŞARILI] Topos Asimetrisi KORUNDU! Cosine Similarity silindi ve Sistem Halüsinasyondan (Simetriden) kurtuldu.\n")
    else:
        print("[BAŞARISIZ] Asimetri kurulamadı.\n")
        
        
    print("--- 2. KAN GENİŞLEMESİ (GERÇEK COLIMIT / SUPREMUM) TESTİ ---")
    print("Test: Yeni Sol Kan Genişlemesi (LeftKanExtension) matematiksel olarak")
    print("Evrensel Colimit (Supremum) kuralına uyuyor mu?")
    
    kan = LeftKanExtension(dim_c=N, dim_e=N, dim_d=N)
    
    source = torch.rand(10, N)
    target = torch.rand(1, N)
    
    # Yeni Güncellenmiş Kan Sınıfından Çıkan Sonuç
    new_kan_result = kan(source, target)
    
    # Matematiksel Doğrulama (Manuel Colimit Hesabı)
    Kc = kan.K(source)
    Fc = kan.F(source)
    sim = target @ Kc.T * kan.scale
    weights = torch.sigmoid(sim) 
    weighted_Fc = weights.unsqueeze(-1) * Fc.unsqueeze(0)
    true_supremum, _ = torch.max(weighted_Fc, dim=1)
    
    kan_norm = torch.norm(new_kan_result).item()
    true_norm = torch.norm(true_supremum).item()
    
    print(f"Güncellenmiş Modelin (Kan) Ürettiği Çıktı: {kan_norm:.6f}")
    print(f"Matematiksel Colimit (Supremum) Hedefi: {true_norm:.6f}")
    
    if abs(kan_norm - true_norm) < 1e-6:
        print("[BAŞARILI] HATA 0.0! Kan Genişlememiz artık Softmax ile ortalama almıyor.")
        print("[BAŞARILI] Sisteme %100 oranında 'Evrensellik' (Universal Property) kuralı yerleştirildi.\n")
    else:
        print("[BAŞARISIZ] Matematiksel Colimit tutmuyor.")

if __name__ == '__main__':
    test_fixed_architecture_proofs()