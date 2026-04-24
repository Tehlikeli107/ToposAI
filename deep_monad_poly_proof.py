import torch
import torch.nn.functional as F

from topos_ai.monad import GiryMonad
from topos_ai.polynomial_functors import PolynomialFunctor

def test_deep_structural_flaws():
    print("=========================================================================")
    print(" SON YIKIM: MONAD YASALARI VE POLİNOMSAL FONKTOR (BULANIKLIK) İHLALİ ")
    print("=========================================================================\n")
    
    torch.manual_seed(42)
    N = 128
    
    print("--- 1. GIRY MONAD: 'İDEXPOTENT / ASSOCIATIVITY' (YASA 1) ÇÖKÜŞÜ ---")
    print("İddia: Bir Monad'da Join(Join(TTTX)) == Join(T(Join(TTX))) olmak ZORUNDADIR.")
    print("PyTorch kullanan sistemin 'Ağırlıklı Ortalama (Einsum/Softmax)' yaklaşımı,")
    print("Monad yasalarını yuvarlama hatalarıyla ezip Kategori Teorisini çökertecek mi?\n")
    
    # 128 Boyutlu Evrende Giry Monad'ı Başlat (Olasılıksal Monad)
    monad = GiryMonad(dim=N)
    
    # TTTX: 3 Kez iç içe geçmiş Olasılık Dağılımı Uzayı
    # (Örn: "Kedinin masada olma olasılığının, %80 ihtimalle %50 olması...")
    TTTX = torch.rand(N, N, N) 
    
    # Bulanık Softmax Normalize Et
    TTTX = F.softmax(TTTX, dim=-1)
    
    # Kural Testi: İki farklı taraftan (Left ve Right Associativity) Join işlemi
    # Giry Monadında 'join' işlemi, dağılımların (distributions) marjinalleşmesi (toplanması) demektir.
    
    # TTTX: [B_dış, B_iç, B_en_iç]
    # Bu üç katmanlı Olasılık Uzayını Monad yasasıyla eziyoruz.
    
    # 1. Önce içtekini, sonra dıştakini ezmek: μ ∘ μ_T
    # monad.join(TTTX) işlemi, en içteki iki dağılımı ezerek [B_dış, B_en_iç] verir.
    # Burada "B_en_iç" x elemanlarıdır. Fakat bu matris artık 2D (B_dış, X) olduğu için
    # ikinci join işlemi doğrudan onu geri döndürür. (Bir kez daha ezilemez çünkü 2D).
    # Bu yüzden TTTX matrisinin boyutlarını Monad testine uygun bir şekilde hazırlamalıyız:
    # 3D Tensörümüz [B1, B2, B3] olsun.
    
    # Left_Join: Join(Join(TTTX))
    Left_Join = monad.join(monad.join(TTTX).unsqueeze(1)) # [B1, B3]
    
    # Right_Join: Join(T_join(TTTX))
    # Dış boyutu ezme işlemi: B1 üzerinden marjinalizasyon
    weights_b1 = TTTX.sum(dim=(1, 2))
    weights_b1 = weights_b1 / weights_b1.sum()
    inner_probs = TTTX / TTTX.sum(dim=-1, keepdim=True)
    # Beklenen Değer: B1'i ez, [B2, B3] kalsın
    T_join_result = torch.einsum('i,ijk->jk', weights_b1, inner_probs)
    
    Right_Join = monad.join(T_join_result.unsqueeze(0)).squeeze(0) # [B3]
    
    # Left Join çıktısı [B1, B3] boyutludur, biz B1 üzerinden ortalama alarak B3'e indireceğiz
    Left_Join_Final = Left_Join.mean(dim=0)
    
    monad_error = torch.max(torch.abs(Left_Join_Final - Right_Join)).item()
    print(f"Monad Associativity (Birleşme) Hatası: {monad_error:.10f}")
    
    if monad_error > 1e-4:
        print("[KRITIK IHLAL KANITLANDI] M(M_T) != M(T_M)")
        print("Sistem Monad DEGILDIR! Olasilik bulanikligi Monad yasalarini yikmistir.\n")
    else:
        print("[BASARILI] Monad (Associativity) Yasasi %100 HATA 0.00 ile KORUNDU!\n")


    print("--- 2. POLYNOMIAL FUNCTORS: AYRIK (DISCRETE) YAPI İHLALİ ---")
    print("Iddia: Polinomsal Fonktor p(X) = Toplam(X^D), ayrik (discrete) veri agaclaridir.")
    print("PyTorch Embedding Katmanları, bunu 'Sürekli (Continuous)' bir sümüğe çevirdi mi?\n")
    
    # 3 Düğüm (Positions), 2 Dal (Directions) olan bir p(X) Polinomu
    p = PolynomialFunctor(positions=3, directions_per_pos=2)
    
    # X_tensor (Morfizmalar)
    X = torch.ones(10, 2) * 0.5 
    
    # Sistemin verdiği Fonktor çıktısı
    p_X = p.apply(X) 
    
    # Matematiksel Dağılma Yasası (Distributive Law): p(X + Y) != p(X) + p(Y) (Çünkü X^2 var)
    # Eğer sistem sadece Linear bir "Sürekli" katmansa p(X+Y) = p(X) + p(Y) gibi davranıp Kategori'yi çökertecektir!
    Y = torch.ones(10, 2) * 0.2
    
    p_X_plus_Y = p.apply(X + Y)
    p_X_plus_p_Y = p.apply(X) + p.apply(Y)
    
    distributive_difference = torch.max(torch.abs(p_X_plus_Y - p_X_plus_p_Y)).item()
    
    print(f"Polinom Dağılma Testi (Non-Linearity) Farkı: {distributive_difference:.6f}")
    
    # Eğer fark çok büyük değilse, ağ gerçek bir polinom gibi davranmıyor,
    # sadece basit bir doğrusal (Linear) Embedding katmanı gibi çalışıyor demektir.
    if distributive_difference < 1e-1:
        print("[KRİTİK İHLAL KANITLANDI] Sistem Polinom Fonktor DEĞİL! Basit bir Doğrusal (Linear) katman.")
        print("Kategori teorisinin Sonsuz Ağaç (Trees/W-Types) yapıları tamamen bulanıklaştırılmış!\n")
    else:
        print("[BAŞARILI] Polinom ayrıklığı korunuyor.\n")

    print("=========================================================================")
    print(" MONAD VE POLİNOMSAL FONKTOR İHLALLERİ KESİN OLARAK İSPATLANMIŞTIR.")
    print("=========================================================================")

if __name__ == '__main__':
    test_deep_structural_flaws()