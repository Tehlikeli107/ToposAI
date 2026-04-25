import torch
import torch.nn.functional as F
import sys
import os

# Standalone çalışma desteği
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =====================================================================
# GROMOV-WASSERSTEIN OPTIMAL TRANSPORT (CROSS-DOMAIN ALIGNMENT)
# Araştırma: İki farklı uzay (Örn: Görüntü ve Dil) arasında hiçbir 
# eşleşme verisi (label) olmadan, sadece içsel geometrilerini (uzaklıklar) 
# birbirine yapıştırarak idealize eşleşmeyi bulma.
# =====================================================================

def sinkhorn_algorithm(M, epsilon=0.01, n_iter=100):
    """
    Entropic Regularized Optimal Transport (Sinkhorn).
    Maliyet matrisi M'den en uygun taşıma planını (P) çıkarır.
    """
    K = torch.exp(-M / epsilon)
    a = torch.ones(M.size(0), device=M.device) / M.size(0)
    b = torch.ones(M.size(1), device=M.device) / M.size(1)
    
    v = torch.ones_like(b)
    for _ in range(n_iter):
        u = a / (torch.matmul(K, v) + 1e-9)
        v = b / (torch.matmul(K.t(), u) + 1e-9)
        
    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    return P

def run_gromov_wasserstein_experiment():
    print("--- GROMOV-WASSERSTEIN OPTIMAL TRANSPORT (ZERO-SHOT ALIGNMENT) ---")
    print("Araştırma: İki farklı dilin (Türkçe ve İngilizce) uzaklık matrislerini \nhiç sözlük olmadan birbirinin üzerine katlama.\n")

    # 1. UZAY A: İNGİLİZCE KELİMELER (Vektör Uzayı)
    # [Kral, Kraliçe, Elma, Armut] arasındaki mesafeler
    # (Kral-Kraliçe yakın, Elma-Armut yakın, diğerleri uzak)
    names_A = ["King", "Queen", "Apple", "Pear"]
    # Temsili vektörler
    X_A = torch.tensor([
        [1.0, 0.0], [0.9, 0.1],  # King, Queen
        [0.0, 1.0], [0.1, 0.9]   # Apple, Pear
    ])
    dist_A = torch.cdist(X_A, X_A) # Uzaklık Matrisi (İçsel Geometri)

    # 2. UZAY B: TÜRKÇE KELİMELER (Sırası karışık!)
    # [Armut, Kral, Elma, Kraliçe]
    names_B = ["Armut", "Kral", "Elma", "Kraliçe"]
    X_B = torch.tensor([
        [0.1, 0.9], [1.0, 0.0],  # Armut, Kral
        [0.0, 1.0], [0.9, 0.1]   # Elma, Kraliçe
    ])
    dist_B = torch.cdist(X_B, X_B)

    print("[VERİ]: İki farklı dilin uzaklık matrisleri hazır.")
    print("Sözlük: YOK. Etiket: YOK. Sadece geometrik şekiller var.")

    # 3. GROMOV-WASSERSTEIN İTERASYONU
    # Amaç: Öyle bir eşleşme (Transport Plan) P bul ki; 
    # dist_A ile dist_B arasındaki fark minimize edilsin.
    
    N = len(names_A)
    # Başlangıç Planı: Eşit olasılık
    P = torch.ones(N, N) / (N * N)
    
    print("\n[OPTIMAL TRANSPORT]: Uzaylar birbirinin üzerine katlanıyor...")
    
    for i in range(20):
        # 1. Mevcut plana göre maliyet matrisini (Cost) hesapla
        # L(P) = dist_A^2 @ P @ 1 + 1 @ P @ dist_B^2 - 2 * dist_A @ P @ dist_B
        # (Basitleştirilmiş GW cost formulation)
        f_A = torch.matmul(dist_A**2, torch.ones(N, N)) @ P
        f_B = torch.matmul(P, dist_B**2) @ torch.ones(N, N)
        cross_term = torch.matmul(torch.matmul(dist_A, P), dist_B)
        
        M = f_A + f_B - 2 * cross_term
        
        # 2. Sinkhorn ile planı güncelle
        P = sinkhorn_algorithm(M, epsilon=0.05)

    print("İşlem Tamamlandı.\n")

    # 4. SONUÇLARIN ANALİZİ
    print("--- EŞLEŞME SONUÇLARI (ZERO-SHOT) ---")
    for i in range(N):
        # P matrisindeki i. İngilizce kelimesine denk gelen en yüksek ihtimal
        match_idx = torch.argmax(P[i]).item()
        confidence = P[i, match_idx].item() * 100
        
        print(f"  {names_A[i]:<8} ===> {names_B[match_idx]:<8} (Güven: %{confidence:.1f})")

    # Bilimsel Kontrol
    matches = [names_B[torch.argmax(P[i]).item()] for i in range(N)]
    correct = (matches[0] == "Kral" and matches[1] == "Kraliçe" and 
               matches[2] == "Elma" and matches[3] == "Armut")

    print("\n[ÖLÇÜLEN SONUÇ]")
    if correct:
        print("[✓] KANITLANDI: Model, tek bir kelime anlamı bilmeden, sadece")
        print("    iki uzayın topolojik uzaklıklarını eşleştirerek sözlüğü çözdü!")
        print("    Bu, farklı modaliteler arasında topolojik/optimal-transport")
        print("    hizalama fikrinin küçük bir demosudur.")
    else:
        print("[-] Hizalama başarısız oldu. Parametre ayarı gerekebilir.")

if __name__ == "__main__":
    run_gromov_wasserstein_experiment()
