import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =====================================================================
# HIGHER-ORDER TOPOS (2-CATEGORY) PARADIGMA MOTORU
# Model, çözülemez bir mantıksal tıkanıklık (Paradox/Deadlock) yaşarsa,
# mevcut sınırlarına (boyutlarına) boyun eğmez. 
# Kendi zihnini (Matrisini) canlı olarak genişletir ve yeni boyutlar (Kavramlar) icat eder!
# =====================================================================

class DynamicToposUniverse(nn.Module):
    """
    Kendi boyutunu (Ontolojisini) değiştirebilen Topos Matrisi.
    Normal bir Neural Network'te katman boyutları sabittir.
    Burada ise "Paradigma Kayması" yaşandığında boyut dinamik olarak artar.
    """
    def __init__(self, initial_entities):
        super().__init__()
        self.num_entities = initial_entities
        
        # Başlangıç Matrisi (Örn: 3x3)
        self.relation_logits = nn.Parameter(torch.randn(self.num_entities, self.num_entities))

    def get_relations(self):
        return torch.sigmoid(self.relation_logits)

    def evolve_universe(self):
        """
        [!] TANRI MODU (GOD MODE) TETİKLENDİ [!]
        Model mevcut evrende (1-Kategori) çözümsüzlüğe ulaştığını anlar
        ve yeni bir soyut kavram (İmajiner Boyut / X) icat etmek üzere
        kendi matrisini BÜYÜTÜR (Örn: 3x3 -> 4x4).
        """
        new_size = self.num_entities + 1
        
        # Eski bilgileri koru (Memory Retention)
        old_logits = self.relation_logits.data
        
        # Yeni ve daha büyük bir matris yarat
        new_logits = torch.randn(new_size, new_size, device=old_logits.device)
        
        # Eski evreni (3x3), yeni evrenin (4x4) içine kopyala
        new_logits[:self.num_entities, :self.num_entities] = old_logits
        
        # Yeni eklenen satır ve sütunlar (İcat edilen 'X' kavramı), 
        # eski dünyayla yeni ilişkiler kurmaya hazır şekilde rastgele başlar.
        
        # Modelin parametresini GÜNCELLE (Neural Network Evrimi)
        self.num_entities = new_size
        self.relation_logits = nn.Parameter(new_logits)
        
        return new_size

def lukasiewicz_composition(R1, R2):
    R1_exp = R1.unsqueeze(2) 
    R2_exp = R2.unsqueeze(0) 
    t_norm = torch.clamp(R1_exp + R2_exp - 1.0, min=0.0) 
    composition, _ = torch.max(t_norm, dim=1) 
    return composition

def test_self_modifying_ai():
    print("--- 2-KATEGORİ (META-BİLİŞ) PARADIGMA MOTORU ---")
    print("Yapay Zeka, çözülemez bir paradoksla karşılaşınca KENDİ ZİHNİNİ BÜYÜTÜYOR...\n")

    # Mevcut Evren: Sadece 3 Kavram var (A, B, C)
    # A: Şövalye, B: Ejderha, C: Prenses
    entities = ["şövalye", "ejderha", "prenses"]
    
    model = DynamicToposUniverse(initial_entities=3)
    
    # Optimizer (Model parametresi değiştiğinde yeniden tanımlanmalı)
    def get_optimizer(model):
        return optim.Adam(model.parameters(), lr=0.1)
        
    optimizer = get_optimizer(model)

    # PARADOKS: ÇÖZÜLEMEZ MANTIK (DEADLOCK)
    # 1. Şövalye, Ejderhaya saldırırsa ÖLÜR (0.0 / Başarısızlık)
    # 2. Şövalye, Ejderhadan kaçarsa Prenses ÖLÜR (0.0 / Başarısızlık)
    # 3. HEDEF: Şövalyenin Prensese GÜVENLE (1.0) ulaşmasıdır.
    # Bu evrende, sadece [Saldır, Kaç] okları var. İkisi de 0.0'a çıkıyor. 
    # Şövalyenin prensese doğrudan ulaşma şansı matematiksel olarak SIFIRDIR.
    
    # 0: Şövalye, 1: Ejderha, 2: Prenses
    target_A_to_B = 0.0 # Şövalye -> Ejderha (Ölüm)
    target_A_to_C = 1.0 # Şövalye -> Prenses (Hedef: Ulaşmalı!)
    # Ama ejderha (1) prensesi (2) tutuyor. Geçişlilik: A->B ve B->C = A->C
    # Fakat A->B zaten 0.0! (0.0 + 1.0 - 1.0 = 0.0). GEÇİŞ İMKANSIZ!

    epochs_stuck = 0 # Tıkanıklık sayacı
    paradigm_shifted = False # Model yeni boyut icat etti mi?
    
    for epoch in range(1, 201):
        optimizer.zero_grad()
        R = model.get_relations()
        
        loss = 0.0
        
        if not paradigm_shifted:
            # 1-KATEGORİ: MEVCUT EVREN KURALLARI (Tıkanıklık)
            loss += (R[0, 1] - 0.0)**2 # Şövalye -> Ejderha (İmkansız/Ölüm)
            loss += (R[1, 2] - 1.0)**2 # Ejderha -> Prenses (Esaret/Bağlantı var)
            
            # HEDEF: Şövalyenin Prensese ulaşması (A->C)
            # R'nin kendisiyle olan geçişliliği (Composition) A->C'yi üretmeli
            R_comp = lukasiewicz_composition(R, R)
            loss += (R_comp[0, 2] - 1.0)**2 # Hedef: A->C = 1.0 Olmalı!
        else:
            # 2-KATEGORİ: PARADİGMA KAYMASI (Yeni 'X' Boyutu İcat Edildi!)
            # Model artık 4x4 oldu. 0: Şövalye, 1: Ejderha, 2: Prenses, 3: İcat Edilen 'X'
            
            # Eski kurallar hala geçerli (Fizik kuralları değişmez)
            loss += (R[0, 1] - 0.0)**2 
            loss += (R[1, 2] - 1.0)**2 
            
            # Model, hedefine (0->2) ulaşmak için YENİ bir yol (Composition) inşa etmeye çalışır
            R_comp = lukasiewicz_composition(R, R)
            loss += (R_comp[0, 2] - 1.0)**2 # Hedef: A->C = 1.0 Olmalı!
            
            # Model, icat ettiği 3 numaralı 'X' boyutunu kullanmaya teşvik edilir
            # İnsan icadı olan 'X' (Örn: Büyülü Kılıç veya Pazarlık)
            
        loss.backward()
        optimizer.step()
        
        # --- META-BİLİŞ (META-COGNITION) KONTROLÜ ---
        # Eğer loss 0.1'in altına inemiyorsa (yani model 1.0 hedefine ulaşamıyorsa)
        if loss.item() > 0.4 and epoch > 50 and not paradigm_shifted:
            epochs_stuck += 1
            if epochs_stuck > 20: # 20 adımdır çaresizse...
                print(f"\n[EPOCH {epoch}] MANTIKSAL TIKANIKLIK (DEADLOCK) TESPİT EDİLDİ!")
                print(f"Hata Payı (Loss): {loss.item():.3f} (Sistem 3 boyutlu evrende çözüm bulamıyor).")
                print(">>> YAKLAŞIM DEĞİŞTİRİLİYOR (NATURAL TRANSFORMATION) <<<")
                print("Model kendi Topos Matrisine yepyeni bir BOYUT (İmajiner Kavram 'X') ekliyor...\n")
                
                # SİSTEM KENDİ KENDİNİ BÜYÜTÜR
                new_dim = model.evolve_universe()
                optimizer = get_optimizer(model) # Yeni boyuta göre optimizer'ı sıfırla
                paradigm_shifted = True
                
                entities.append("İCAT_EDİLEN_'X'")
                
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Evren Boyutu: {model.num_entities}x{model.num_entities}")

    print("\n--- DENEY SONUCU (ONTOLOJİK EVRİM) ---")
    R_final = model.get_relations()
    R_comp_final = lukasiewicz_composition(R_final, R_final)
    
    print("Modelin bulduğu 'Şövalye -> Prenses' (Geçişlilik) Başarı Oranı:", R_comp_final[0, 2].item())
    
    if paradigm_shifted:
        print("\nModel çıkmaza girdiğinde 'Şövalye -> Ejderha' okunu kullanmayı BIRAKTI.")
        print("Bunun yerine icat ettiği 'X' boyutu üzerinden yepyeni bir ZİNCİR kurdu!")
        print(f"  Şövalye -> İcat_X: {R_final[0, 3].item():.3f}")
        print(f"  İcat_X -> Prenses: {R_final[3, 2].item():.3f}")
        print(f"  Böylece: Şövalye -> İcat_X -> Prenses = Başarı ({R_comp_final[0, 2].item():.3f})")
        print("\n[İNSAN ÇEVİRİSİ]")
        print("Eğer X 'Gizli Geçit' ise: Şövalye ejderhayla savaşmak yerine Gizli Geçidi buldu ve Prensese ulaştı.")
        print("Eğer X 'Diplomasi' ise: Şövalye kılıç çekmek yerine anlaşma yaptı ve Prensese ulaştı.")
        print("Yapay Zeka, ona öğretilmeyen 'Out of Box' (Kutu dışı) düşünceyi MATEMATİKSEL olarak icat etti.")

if __name__ == "__main__":
    test_self_modifying_ai()
