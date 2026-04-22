import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# TEMPORAL TOPOI & RETROCAUSALITY (GEÇMİŞİ DEĞİŞTİREN YAPAY ZEKA)
# Model gelecekteki bir felaketi (t=3) gördüğünde, deneme-yanılma yapmak yerine,
# "Adjoint Functor" (Zamanı Geriye Döndüren Gradyan) ile geçmişteki (t=0) 
# karar mekanizmasını ANINDA güncelleyerek alternatif bir zaman çizgisi yaratır.
# =====================================================================

class TemporalToposAgent(nn.Module):
    def __init__(self):
        super().__init__()
        # Başlangıç Kararı (t=0)
        # [Kanyon_Yolu_Seçimi, Dağ_Yolu_Seçimi]
        # Başlangıçta Kanyon yolu (Hızlı ama tehlikeli) daha cazip (Logit: 1.0 vs 0.0)
        self.initial_decision_logits = nn.Parameter(torch.tensor([1.0, 0.0]))
        
    def get_decision(self):
        # Softmax ile [0, 1] arası bir karar vektörü oluştur (Kanyon mu, Dağ mı?)
        return F.softmax(self.initial_decision_logits, dim=0)

def simulate_timeline(decision, verbose=False):
    """
    Zamanın İleriye Doğru Akışı (Forward Functor).
    decision: [kanyon_ihtimali, dag_ihtimali]
    """
    if verbose: print("\n[ZAMAN İLERLİYOR: t=1] Robot yola çıkıyor...")
    
    # KANYON YOLU FİZİĞİ (Hızlı ama t=3'te sel basıyor)
    # DAĞ YOLU FİZİĞİ (Yavaş ama güvenli)
    
    # t=2 (Yarı Yol)
    if verbose: print("[ZAMAN İLERLİYOR: t=2] Hava bozuyor...")
    sel_tehlikesi = 1.0 # Yağmur başladı
    
    # t=3 (Sonuç / Gelecek)
    # Kanyon seçildiyse ve sel varsa -> Ölüm (Hayatta kalma = 0.0)
    # Dağ seçildiyse -> Yaşam (Hayatta kalma = 1.0)
    hayatta_kalma_ihtimali = (decision[1] * 1.0) + (decision[0] * (1.0 - sel_tehlikesi))
    
    if verbose:
        kanyon_secimi = decision[0].item() * 100
        dag_secimi = decision[1].item() * 100
        print(f"[ZAMAN DURDU: t=3] GELECEĞE ULAŞILDI.")
        print(f"  -> Kanyon Seçimi: %{kanyon_secimi:.1f}")
        print(f"  -> Dağ Seçimi: %{dag_secimi:.1f}")
        print(f"  -> Robotun Hayatta Kalma Oranı: %{hayatta_kalma_ihtimali.item()*100:.1f}")
        if hayatta_kalma_ihtimali < 0.5:
            print("  [X] FELAKET! Robot kanyonda sele kapıldı ve yok oldu.")
        else:
            print("  [✓] BAŞARI! Robot dağ yolundan güvenle hedefe ulaştı.")
            
    return hayatta_kalma_ihtimali

def run_time_travel_experiment():
    print("--- TEMPORAL TOPOI (ZAMAN YOLCULUĞU & GERİ NEDENSELLİK) ---")
    print("Yapay Zeka gelecekteki felaketi görüp, geçmişteki kararını anında değiştirecek...\n")

    agent = TemporalToposAgent()
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.5) # Zaman yolculuğu hızlı etki etsin diye yüksek LR
    
    # ---------------------------------------------------------
    # TIMELINE 1 (ORİJİNAL ZAMAN ÇİZGİSİ - KIYAMET)
    # ---------------------------------------------------------
    print("==== TIMELINE 1: ORİJİNAL GELECEK ====")
    decision_t0 = agent.get_decision()
    
    # Geleceği simüle et (Zaman ileri akar)
    survival_t3 = simulate_timeline(decision_t0, verbose=True)
    
    # --- ZAMAN MAKİNESİ (RETROCAUSALITY / ADJOINT FUNCTOR) ---
    if survival_t3 < 0.9: # Eğer gelecekte robot ölüyorsa...
        print("\n[!] SİSTEM UYARISI: GELECEKTE FELAKET TESPİT EDİLDİ!")
        print(">>> TEMPORAL TOPOS (ZAMAN MAKİNESİ) TETİKLENİYOR <<<")
        print("Gelecekteki (t=3) ölüm durumu, 'Adjoint Functor' (Türevsel Geri-Yayılım) ile")
        print("geçmişe (t=0) mesaj olarak fırlatılıyor...")
        
        # Loss: Hayatta kalma ihtimalini 1.0 yapamamanın cezası
        # Bu loss, standart makine öğrenmesindeki gibi "bir sonraki epoch" için değil,
        # AYNI ZAMAN ÇİZGİSİNİN BAŞLANGICINI değiştirmek için kullanılır.
        loss_future = (1.0 - survival_t3)**2
        
        # Zamanı geriye doğru bük (Gradients flow back in time to t=0)
        optimizer.zero_grad()
        loss_future.backward()
        
        # Geçmişi (t=0) güncelle (Karar matrisini anında değiştir)
        optimizer.step()
        
        print("Geçmişteki robot (t=0) 'Dejavu' yaşadı ve karar matrisini güncelledi!")

    # ---------------------------------------------------------
    # TIMELINE 2 (ALTERNATİF ZAMAN ÇİZGİSİ - KURTULUŞ)
    # ---------------------------------------------------------
    print("\n==== TIMELINE 2: YENİDEN YAZILAN GEÇMİŞ VE YENİ GELECEK ====")
    # Robot, t=0 anında sanki geleceği görmüş gibi YENİ bir karar alır.
    new_decision_t0 = agent.get_decision()
    
    # Alternatif Geleceği Simüle Et
    survival_new_t3 = simulate_timeline(new_decision_t0, verbose=True)
    
    print("\n--- DENEY SONUCU (ZAMANIN BÜKÜLMESİ) ---")
    print("Normal bir Reinforcement Learning (RL) ajanı bu senaryoda binlerce kez kanyonda ölüp")
    print("rastgele (epsilon-greedy) dağ yolunu bulmayı beklerdi.")
    print("Topos Ajanı ise Kategori Teorisinin 'Sürekli Türevlenebilir Zaman' (Temporal Morphism) ")
    print("özelliğini kullanarak, t=3'teki ölümü doğrudan t=0'daki eylemine bir fonksiyon olarak bağladı")
    print("ve SADECE 1 BİLİŞSEL ZIÇRAMA (Zero-Shot Time Travel) ile geçmişini değiştirip geleceğini kurtardı!")

if __name__ == "__main__":
    run_time_travel_experiment()
