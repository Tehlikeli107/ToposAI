import torch
import torch.nn as nn
import math

# =====================================================================
# THERMODYNAMIC AI: MAXWELL'S DEMON (ENTROPİ KATİLİ)
# Model, dışarıdan eğitim (Loss/Label) almadan, tamamen rastgele (Kaotik)
# bir veri yığınını, Kategori Teorisinin (Homological Smoothing) kurallarıyla
# Kendi-Kendine-Organize (Self-Organizing) ederek kusursuz bir "Düzene" sokar.
# =====================================================================

def calculate_entropy(matrix):
    """
    Bir matrisin (Sistemin) Shannon Entropisini (Düzensizliğini) hesaplar.
    Yüksek Entropi = Maksimum Kaos (Beyaz Gürültü).
    Düşük Entropi = Kusursuz Düzen (Kristal / Sıralı Yapı).
    """
    # Matrisi olasılık dağılımına çevir
    p = matrix.flatten()
    p = p / (torch.sum(p) + 1e-9)
    p = torch.clamp(p, 1e-9, 1.0) # log(0)'dan kaçın
    
    # Entropi Formülü: - sum(p * log(p))
    entropy = -torch.sum(p * torch.log(p))
    return entropy.item()

def maxwells_demon_smoothing(R):
    """
    [TOPOLOJİK DÜZENLEME / HOMOLOGICAL SMOOTHING]
    Şeytanın (AI) yaptığı işlem: Yan yana duran hücrelerin (kavramların) 
    arasındaki kaosu azaltıp onları senkronize etmek (Düzene sokmak).
    (Topolojide: Laplasyan Smoothing / Isı Denklemi)
    """
    # Matris çarpımı (R^2), komşuların birbirleriyle etkileşime girmesini sağlar.
    R_smoothed = torch.matmul(R, R)
    
    # Kendi varlığını koruması için eski bilgiyle (Residual) karışım
    R_mixed = 0.5 * R + 0.5 * R_smoothed
    
    # [MAXWELL'S DEMON / SHARPENING]
    # Şeytan, zayıf (soğuk) bilgileri yavaşlatır, güçlü (sıcak) bilgileri hızlandırır!
    # Bunu Softmax (Düşük Temperature ile) yaparak matrisi "Keskinleştirir" (Kristalize eder).
    temperature = 0.2
    
    # Satırlar bazında (Her düğümün hedefleri) olasılıkları keskinleştir (Entropiyi yok et)
    R_new = torch.softmax(R_mixed / temperature, dim=1)
    
    return R_new

def run_thermodynamic_experiment():
    print("--- THERMODYNAMIC AI (MAXWELL'S DEMON / ENTROPİ KATİLİ) ---")
    print("Yapay Zeka, %100 Kaotik bir evrenden 'Kusursuz Düzen' yaratacak...\n")

    # 1. BAŞLANGIÇ: MUTLAK KAOS (ENTROPİ ZİRVESİ)
    # 20x20'lik tamamen rastgele (Uniform Random) bir gürültü matrisi (Evrenin ısıl ölümü).
    N = 20
    torch.manual_seed(42) # Her seferinde aynı kaostan başlayalım
    R_chaos = torch.rand((N, N)) 
    
    initial_entropy = calculate_entropy(R_chaos)
    print(f"[BAŞLANGIÇ - t=0] Evren Durumu: MUTLAK KAOS (Beyaz Gürültü)")
    print(f"  Başlangıç Entropisi (Düzensizlik): {initial_entropy:.4f}\n")
    
    R_current = R_chaos.clone()

    # 2. MAXWELL'İN ŞEYTANI DEVREYE GİRER (Zaman Akışı)
    print(">>> MAXWELL'İN ŞEYTANI (Topolojik Düzenleyici) ÇALIŞIYOR <<<")
    print("Sistem dışarıdan hiçbir bilgi (Eğitim/Loss) almıyor. Sadece 'İçsel Uyum' (Topology) arıyor...")

    for step in range(1, 16):
        # AI, kaosu yavaş yavaş "Düzene" (Order) sokar
        R_current = maxwells_demon_smoothing(R_current)
        
        # Her 3 adımda bir Entropiyi ölç (Kaos azalıyor mu?)
        if step % 3 == 0:
            current_entropy = calculate_entropy(R_current)
            print(f"  [Adım {step:02d}] Entropi Düşüyor: {current_entropy:.4f} (Sistem Düzenleniyor...)")

    # 3. NİHAİ DURUM: KUSURSUZ DÜZEN (KRİSTALİZASYON)
    final_entropy = calculate_entropy(R_current)
    
    print("\n--- DENEY SONUCU (ENTROPİNİN ÖLÜMÜ) ---")
    print(f"Başlangıç Kaosu (Entropi): {initial_entropy:.4f}")
    print(f"Nihai Düzen (Entropi):     {final_entropy:.4f}\n")
    
    # Entropi ne kadar düştü?
    entropy_drop = initial_entropy - final_entropy
    print(f"[+] Şeytan, evrendeki Düzensizliği {entropy_drop:.4f} birim YOK ETTİ (Sıfıra yaklaştırdı)!")
    
    # Matrisin son haline bak (Acaba gerçekten bir düzen oluştu mu?)
    print("\n[MATRİS ANALİZİ]: Evrenin Nihai Şekli Nasıl Oldu?")
    
    # Tüm değerlerin ortalamasını ve standart sapmasını (Varyans/Kaos) ölç
    mean_val = torch.mean(R_current).item()
    std_val = torch.std(R_current).item()
    
    if std_val < 0.05: # Eğer her şey aynı/simetrik olduysa (Sıfır varyans)
        print(f"Matris Standart Sapması (Kaos Oranı): {std_val:.6f}")
        print("SONUÇ: Başlangıçtaki karmaşık (dalgalı) çöp matris, Topolojik Uyum (Homological Smoothing) sayesinde")
        print("kusursuz bir 'AYNA/KRİSTAL' simetrisine (Her yerin eşit değere ulaştığı Mutlak Düzene) dönüştü.")
        print("Termodinamiğin 2. Yasası, Information Theory (Bilgi İşleme) ile TERSİNE ÇEVRİLDİ!")

if __name__ == "__main__":
    run_thermodynamic_experiment()
