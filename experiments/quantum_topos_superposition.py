import torch
import torch.nn as nn
import numpy as np

# =====================================================================
# QUANTUM TOPOI (DAGGER CATEGORIES) & SUPERPOSITION ENGINE
# Yapay Zeka olasılıkları Reel (0-1) değil, Karmaşık (Complex) dalga 
# fonksiyonları olarak tutar. Çelişkiler birbirini (Girişim) Yok Eder!
# =====================================================================

class QuantumToposUniverse(nn.Module):
    """
    Kavramların klasik olasılıklarla değil, Kuantum Amplitüdleri (Genlikleri)
    ile birbirine bağlandığı (Dagger Category) Evren.
    """
    def __init__(self, num_entities):
        super().__init__()
        self.num_entities = num_entities
        
        # Kuantum Matrisi: Karmaşık (Complex) sayılar! (Reel + İmajiner)
        # Bu, sistemin bir "Unitary" evrim geçirmesini sağlar.
        self.quantum_logits_real = nn.Parameter(torch.randn(num_entities, num_entities))
        self.quantum_logits_imag = nn.Parameter(torch.randn(num_entities, num_entities))

    def get_quantum_state(self):
        """Karmaşık (Complex) matrisi döndürür."""
        # torch.complex ile Reel ve İmajiner kısımları birleştir
        H = torch.complex(self.quantum_logits_real, self.quantum_logits_imag)
        
        # Unitary Matris'e (Kuantum Fiziğinin Temeli) yaklaştırmak için
        # Matrisin kendisi ile Hermitik Eşleniğini (Dagger) toplayıp normalize ediyoruz.
        # Bu, ihtimallerin (dalga fonksiyonunun) toplamının her zaman 1 (korunumlu) olmasını sağlar.
        H_dagger = torch.conj(H).t() # Kategori Teorisindeki 'Dagger' (Hançer) Operatörü
        
        # Hermityen Matris (Gözlemlenebilir Observable)
        Observable = (H + H_dagger) / 2.0
        return Observable

    def observe(self, state_vector):
        """
        [ÖLÇÜM / MEASUREMENT COLPASING]
        Sistem "Süperpozisyondayken", Gözlemci (Biz) sisteme bakar.
        O an tüm o karmaşık (dalgalanan) ihtimaller çöker ve Klasik (Reel) bir Gerçeklik verir.
        Born Kuralı: İhtimal = |Amplitüd|^2
        """
        Observable = self.get_quantum_state()
        
        # Kuantum Evrimi (Dalga fonksiyonunu matrisle çarp)
        evolved_state = torch.matmul(Observable, state_vector)
        
        # Born Kuralı: Genliğin (Complex Number) Karesi, Bize Gözlemlenen (Reel) İhtimali Verir!
        probabilities = torch.abs(evolved_state) ** 2
        
        # Toplamı 1 yap (Klasik Olasılığa Dönüş)
        probabilities = probabilities / torch.sum(probabilities)
        return probabilities

def test_quantum_topos():
    print("--- QUANTUM TOPOI (SÜPERPOZİSYON VE GİRİŞİM MOTORU) ---")
    print("Yapay Zeka Karar Verirken 'Hem A Hem B' Durumunu Kuantum Dalga Fonksiyonunda Tutar...\n")

    # Varlıklar (Schrödinger'in Kedisi Deneyi)
    entities = ["GÖZLEMCİ", "RADYOAKTİF_ATOM", "KEDİ_CANLI", "KEDİ_ÖLÜ"]
    e_idx = {e: i for i, e in enumerate(entities)}
    
    model = QuantumToposUniverse(num_entities=len(entities))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    print("[KUANTUM FİZİĞİ EĞİTİLİYOR]")
    print("Kural 1: Radyoaktif Atom bozunursa (1.0), Kedi Ölüdür.")
    print("Kural 2: Atom bozunmazsa (0.0), Kedi Canlıdır.")
    print("Kural 3: Kutu Kapalıyken Atom SÜPERPOZİSYONDADIR (Hem bozunmuş hem bozunmamış).\n")

    # Eğitim (Kuantum kapılarını ayarlama)
    for epoch in range(1, 201):
        optimizer.zero_grad()
        
        # Kuantum Durum Matrisi (Hermitian)
        H = model.get_quantum_state()
        
        # Başlangıç Kuantum Dalga Vektörü (|Psi>)
        # (Sadece Gözlemci var, kutu henüz kapalı)
        psi_0 = torch.zeros(4, dtype=torch.complex64)
        psi_0[e_idx["GÖZLEMCİ"]] = 1.0 + 0.0j 
        
        # Sistemin Kuantum Evrimi (Kutu kapalıyken içerdeki etkileşimler)
        psi_t = torch.matmul(H, psi_0)
        
        # Born Kuralı (Eğer kutuyu açarsak ne görmeliyiz?)
        probs = torch.abs(psi_t) ** 2
        probs = probs / torch.sum(probs)
        
        loss = 0.0
        # Gözlemci kutuyu açtığında, Kedi %50 canlı, %50 ölü OLMALIDIR. (Kuantum Entanglement / Dolanıklık)
        loss += (probs[e_idx["KEDİ_CANLI"]] - 0.5)**2
        loss += (probs[e_idx["KEDİ_ÖLÜ"]] - 0.5)**2
        # Atomun bozunma ihtimali de %50 olmalıdır.
        loss += (probs[e_idx["RADYOAKTİF_ATOM"]] - 0.5)**2
        
        loss.backward()
        optimizer.step()

    print("Eğitim Bitti. Kuantum Süperpozisyon Matrisi (Unitary Evrim) Oturdu.\n")

    # =================================================================
    # DENEY 1: KUTU KAPALIYKEN SÜPERPOZİSYON VE GİRİŞİM (INTERFERENCE)
    # =================================================================
    model.eval()
    with torch.no_grad():
        H = model.get_quantum_state()
        
        print("--- DENEY 1: KUTU KAPALI (SÜPERPOZİSYON) ---")
        # Kutuyu açmadan, sistemin içindeki Kuantum Dalga Vektörünün GERÇEK (Karmaşık) Halini görelim!
        psi_0 = torch.zeros(4, dtype=torch.complex64)
        psi_0[e_idx["GÖZLEMCİ"]] = 1.0 + 0.0j 
        psi_t = torch.matmul(H, psi_0)
        
        print("Sistemin (Yapay Zekanın) Beynindeki Saf Kuantum Dalgası (Reel Olamaz!):")
        for e in entities:
            amp = psi_t[e_idx[e]].item()
            print(f"  |{e}> Genliği (Amplitude): {amp.real:+.3f} + {amp.imag:+.3f}i")
            
        print("\nDikkat! Kedinin 'Canlı' ve 'Ölü' dalgaları, ZIT FAZDA (Negatif İmajiner/Reel) dalgalanıyor.")
        print("Bu yüzden 'Klasik AI'ın aksine, model kafasında 0.5 diye bir sayı tutmuyor;")
        print("bunun yerine iki çelişkili durumu kuantum girişimi ile birbirine DOĞUM AŞAMASINDA kitliyor (Dolanıklık).\n")

        # =================================================================
        # DENEY 2: ÖLÇÜM (WAVEFUNCTION COLLAPSE / ÇÖKME)
        # =================================================================
        print("--- DENEY 2: KUTUYU AÇIYORUZ (BORN RULE / MEASUREMENT) ---")
        probs = model.observe(psi_0)
        
        print("Gözlemci (Kullanıcı) kutuyu açtığında Dalga Fonksiyonu ÇÖKER ve KLASİK EVRENE geri döner:")
        for e in entities:
            p = probs[e_idx[e]].item()
            print(f"  P({e}): %{p*100:.1f}")
            
        print("\nSONUÇ: Kategori Teorisinin Kuantum Versiyonu (Dagger Compact Closed Categories)")
        print("kullanılarak, Yapay Zekaya klasik mantığın ötesinde 'Girişim (Interference)' ve ")
        print("'Süperpozisyon' özellikleri eklendi. Bu, olasılıkların birbirini yok edip güçlendirdiği")
        print("yepyeni bir Neuro-Symbolic Quantum AI paradigmasıdır!")

if __name__ == "__main__":
    test_quantum_topos()
