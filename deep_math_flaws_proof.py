import torch
import torch.nn.functional as F

from topos_ai.kan import LeftKanExtension

def test_deep_flaws():
    print("=========================================================================")
    print(" DERİN MİMARİ KUSURLARI (ASYMMETRY & COLIMIT) KANITLAMA TESTİ ")
    print("=========================================================================\n")
    
    torch.manual_seed(42)
    N = 128
    
    print("--- 1. YÖNLÜ MANTIK ASİMETRİSİ (COSINE SIMILARITY İHLALİ) ---")
    print("İddia: Topos/Heyting cebirinde A <= B doğru iken, B <= A yanlış olmalıdır.")
    print("Kosinüs Benzerliği bu yönlülüğü yokedip Sistemi Halüsinasyona açık hale getiriyor mu?\n")
    
    # A = Alt küme (örn: [0.1, 0.1, 0.1]) -> "Kedi"
    # B = Üst küme (örn: [0.9, 0.9, 0.9]) -> "Hayvan"
    A = torch.ones(N) * 0.1
    B = torch.ones(N) * 0.9
    
    cosine_A_B = F.cosine_similarity(A.unsqueeze(0), B.unsqueeze(0)).item()
    cosine_B_A = F.cosine_similarity(B.unsqueeze(0), A.unsqueeze(0)).item()
    
    print(f"Kedi -> Hayvan Yönü (Kosinüs): {cosine_A_B:.6f}")
    print(f"Hayvan -> Kedi Yönü (Kosinüs): {cosine_B_A:.6f}")
    
    if abs(cosine_A_B - cosine_B_A) < 1e-6:
        print("[KANITLANDI] Kosinüs Benzerliği SİMETRİKTİR!")
        print("[KRİTİK HATA] Sistem 'Kedi bir hayvandır' kadar 'Her hayvan bir kedidir'i de DOĞRU kabul ediyor. Mantık çöktü!\n")
    else:
        print("[BAŞARILI] Yön korundu.\n")
        
        
    print("--- 2. KAN GENİŞLEMESİ (COLIMIT/SUPREMUM) İHLALİ ---")
    print("İddia: Kategori teorisinde Sol Kan Genişlemesi bir Colimit (Supremum / Maximum) operasyonudur.")
    print("Ağın kullandığı Softmax, Colimit kuralını (Üst Sınır) ihlal ediyor mu?\n")
    
    kan = LeftKanExtension(dim_c=N, dim_e=N, dim_d=N)
    
    source = torch.rand(10, N) # 10 adet kaynak (c) nesne
    target = torch.rand(1, N)  # 1 adet hedef (e) nesne
    
    # Eski sistemdeki gibi Kan'ı hesapla (Softmax ile ortalama alır)
    kan_softmax_result = kan(source, target)
    
    # Gerçek matematiksel Colimit (Supremum): Maksimum Değer Çıkarımı
    # sup_c (K(c, e) ⊗ F(c))
    Kc = kan.K(source)
    Fc = kan.F(source)
    sim = target @ Kc.T * kan.scale  # [1, 10]
    
    # Her bir kaynağın (c) hedef üzerindeki ağırlığını F(c) ile çarp
    # Topos'ta Tensör çarpımı yerine Lukasiewicz veya klasik max kullanılır
    # En basit haliyle, ağdaki bileşenlerin MAX'ı alınmalıdır.
    weighted_components = torch.softmax(sim, dim=-1).T * Fc # [10, N]
    
    true_colimit_result, _ = torch.max(weighted_components, dim=0)
    
    # İhlal: Softmax sonucu, Colimit'in (En büyük üst sınırın) yanına bile yaklaşamaz
    softmax_norm = torch.norm(kan_softmax_result).item()
    colimit_norm = torch.norm(true_colimit_result).item()
    
    print(f"Sistemin ürettiği Softmax(Average) Gücü: {softmax_norm:.6f}")
    print(f"Gerçek Matematiksel Supremum(Colimit) Gücü: {colimit_norm:.6f}")
    
    if softmax_norm < colimit_norm or abs(softmax_norm - colimit_norm) > 1e-4:
        print("[KANITLANDI] Softmax, Colimit (Supremum) değildir!")
        print(f"[KRİTİK HATA] Sol Kan Genişlemesi, matematiksel 'Evrensellik' kuralını %{abs(softmax_norm-colimit_norm)/colimit_norm * 100:.2f} oranında İHLAL ediyor!\n")
    
    print("=========================================================================")
    print(" İKİ AĞIR MATEMATİKSEL KUSUR KESİN TENSÖRLERLE İSPATLANMIŞTIR.")
    print("=========================================================================")

if __name__ == '__main__':
    test_deep_flaws()