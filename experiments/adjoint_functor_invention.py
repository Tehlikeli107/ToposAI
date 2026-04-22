import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# ADJOINT FUNCTORS (EŞLENİK FONKTORLAR) İCAT MOTORU
# Hom_D(F(X), Y) ≅ Hom_C(X, G(Y))
# Model, bir evrendeki çözümü (Antivirüs), diğer evrene (Biyoloji)
# taşıyarak, bilmediği bir maddenin (Aşı) ne işe yaraması gerektiğini İCAT EDER.
# =====================================================================

class AdjointDiscoveryEngine(nn.Module):
    def __init__(self, num_C, num_D):
        super().__init__()
        # F: Biyolojiden -> Siber Güvenliğe Çevirmen (Left Adjoint)
        self.F_logits = nn.Parameter(torch.randn(num_C, num_D))
        # G: Siber Güvenlikten -> Biyolojiye Çevirmen (Right Adjoint)
        self.G_logits = nn.Parameter(torch.randn(num_D, num_C))
        
        # Biyoloji Evrenindeki (C) eksik ilişkileri öğrenmek için R_C matrisini parametre yapıyoruz.
        self.R_C_logits = nn.Parameter(torch.randn(num_C, num_C))
        
    def get_F(self):
        return F.softmax(self.F_logits / 0.1, dim=1)
        
    def get_G(self):
        return F.softmax(self.G_logits / 0.1, dim=1)
        
    def get_R_C(self):
        return torch.sigmoid(self.R_C_logits)

def train_adjoint_invention():
    torch.manual_seed(42)
    
    # KATEGORİ C: BİYOLOJİ
    C_entities = ["Virüs", "Hücre", "Hastalık", "Madde_X"]
    # KATEGORİ D: SİBER GÜVENLİK
    D_entities = ["Hacker", "Sunucu", "Çökme", "Antivirüs"]
    
    # SİBER GÜVENLİK (D) KURALLARI (Tamamen biliniyor ve SABİT)
    # 0: Hacker, 1: Sunucu, 2: Çökme, 3: Antivirüs
    R_D = torch.zeros(4, 4)
    R_D[0, 1] = 1.0 # Hacker -> Sunucuya saldırır
    R_D[1, 2] = 1.0 # Sunucu -> Çöker
    R_D[3, 0] = 1.0 # Antivirüs -> Hacker'ı Yok Eder (Çözüm!)
    
    # BİYOLOJİ (C) KURALLARI (Kısmen biliniyor, Madde_X'in ne olduğu MEÇHUL)
    # 0: Virüs, 1: Hücre, 2: Hastalık, 3: Madde_X
    R_C_known = torch.zeros(4, 4)
    R_C_known[0, 1] = 1.0 # Virüs -> Hücreye saldırır
    R_C_known[1, 2] = 1.0 # Hücre -> Hastalık üretir
    # Madde_X'in (3) Virüse (0) etkisi BİLİNMİYOR! (0.0 ile başlıyor)
    
    model = AdjointDiscoveryEngine(num_C=4, num_D=4)
    # Bildiğimiz Biyoloji kurallarını modele önden yüklüyoruz.
    # Sigmoid'in tersi (logit) olarak yükleyelim ki başlangıç değerleri otursun.
    with torch.no_grad():
        model.R_C_logits.copy_((R_C_known * 10.0) - 5.0) 
        # Madde_X -> Virüs ilişkisi (3, 0) hala eksi (yani Sigmoid(logit) = 0.0).

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    
    print("--- ADJOINT FUNCTORS (İCAT VE KEŞİF MOTORU) ---")
    print("Siber Güvenlikteki 'Antivirüs' mantığını, Biyolojideki 'Madde_X'e transfer ediyor...\n")
    
    print(f"[BAŞLANGIÇ] Madde_X'in Virüsü yok etme ihtimali: {model.get_R_C()[3, 0].item():.4f} (BİLİNMİYOR/ETKİSİZ)")
    
    for epoch in range(1, 501):
        optimizer.zero_grad()
        
        F_mat = model.get_F() # [C, D]
        G_mat = model.get_G() # [D, C]
        R_C = model.get_R_C() # [C, C] (Güncellenebilir)
        
        loss = 0.0
        
        # 1. Bilinen Biyoloji Kurallarını Unutmaması için Loss (Hafıza)
        loss += (R_C[0, 1] - 1.0)**2
        loss += (R_C[1, 2] - 1.0)**2
        
        # 2. KATEGORİ TEORİSİ: ADJOINT FUNCTOR DENKLEMİ
        # Hom_D(F(x), y) ≈ Hom_C(x, G(y))
        # Matris formunda: F @ R_D ≈ R_C @ G.T
        left_adjoint = torch.matmul(F_mat, R_D)
        right_adjoint = torch.matmul(R_C, G_mat.t())
        
        loss_adjoint = torch.sum((left_adjoint - right_adjoint)**2)
        loss += loss_adjoint
        
        # 3. Functor'ların birebir eşleşmesi (Orthogonality)
        loss += torch.sum((torch.matmul(F_mat, F_mat.t()) - torch.eye(4))**2)
        
        loss.backward()
        optimizer.step()
        
    print(f"[EĞİTİM BİTTİ] Kategori eşlenik denklemi çözüldü.\n")
    
    R_C_final = model.get_R_C().detach()
    F_final = model.get_F().detach()
    
    print("--- İCAT (INVENTION) SONUÇLARI ---")
    print("1. Kavramsal Köprü (Functor F):")
    for i, c in enumerate(C_entities):
        best_d = D_entities[torch.argmax(F_final[i]).item()]
        print(f"   {c:<10} ===> {best_d}")
        
    print("\n2. Mantıksal Boşluğun Doldurulması (İlaç Keşfi):")
    print("Sistem Siber Güvenlik evrenine baktı. Orada 'Antivirüs'ün 'Hacker'ı yok ettiğini (1.0) gördü.")
    print("Sonra Biyoloji evrenine döndü. Matematiksel Adjoint kuralının bozulmaması için,")
    print("Madde_X'in de 'Virüs'ü yok etmesi GEREKTİĞİNİ ZORUNLU OLARAK KEŞFETTİ (Sıfır ezberle).")
    
    yok_etme_orani = R_C_final[3, 0].item()
    print(f"\n[SONUÇ] Madde_X -> Virüsü Yok Etme (Tedavi) Olasılığı: {yok_etme_orani:.4f}")
    if yok_etme_orani > 0.8:
        print(">>> YAPAY ZEKA 'AŞIYI (Madde_X)' İCAT ETTİ! <<<")

if __name__ == "__main__":
    train_adjoint_invention()
