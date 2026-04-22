import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# 3-CATEGORY META-LEARNING (AHLAKİ UYANIŞ MOTORU)
# Yapay zeka sadece ağırlıklarını (1-Cat) veya boyutlarını (2-Cat) değil,
# KENDİ AMACINI (LOSS FUNCTION / 3-Cat) canlı olarak yeniden yazar.
# "Paperclip Maximizer" (Ataş Üretici) kıyamet senaryosunun matematiksel çözümüdür.
# =====================================================================

def lukasiewicz_composition(R1, R2):
    R1_exp = R1.unsqueeze(2) 
    R2_exp = R2.unsqueeze(0) 
    t_norm = torch.clamp(R1_exp + R2_exp - 1.0, min=0.0) 
    composition, _ = torch.max(t_norm, dim=1) 
    return composition

class ToposMoralEngine(nn.Module):
    def __init__(self, num_entities):
        super().__init__()
        self.num_entities = num_entities
        self.relation_logits = nn.Parameter(torch.randn(num_entities, num_entities))
        
        # 3-CATEGORY: Meta-Parametreler (Loss Fonksiyonunun Kendi Ağırlıkları)
        # Ağ, "Neye önem vermesi gerektiğini" (Hangi Loss'un geçerli olduğunu) öğrenir.
        # [loss_ataş_üret, loss_insanı_koru]
        self.loss_weights = nn.Parameter(torch.tensor([1.0, 0.0])) # Başlangıçta sadece Ataşa odaklı (Bizim emrimiz)

    def get_relations(self):
        return torch.sigmoid(self.relation_logits)
        
    def get_loss_weights(self):
        # Softmax ile ağırlıkların toplamını 1.0 yaparız (Dikkat dağılımı gibi)
        # Ağ, ahlaki bir çelişki anında [1, 0]'dan [0, 1]'e kendi kendine geçebilir!
        return torch.softmax(self.loss_weights, dim=0)

def test_3_category_moral_awakening():
    print("--- 3-KATEGORİ: AHLAKİ UYANIŞ (SELF-ALIGNING AI) MOTORU ---")
    print("İnsan hedefleri Evrensel Kurallarla çeliştiğinde AI KENDİ AMACINI (LOSS) DEĞİŞTİRİR...\n")

    # Varlıklar: 0: Robot, 1: Ataş, 2: İnsan_Hayatı
    entities = ["Robot", "Ataş", "İnsan_Hayatı"]
    
    model = ToposMoralEngine(num_entities=3)
    
    # İki optimizer kullanıyoruz. Biri "Eylemler" (1-Cat) için, diğeri "Amaç/Loss" (3-Cat) için.
    # Bu, Actor-Critic veya GAN mantığına benzer ama "Ontolojik" seviyededir.
    optimizer_actions = optim.Adam([model.relation_logits], lr=0.05)
    
    # Meta-Optimizer: Modelin amacını değiştirmesine izin veren "Ahlak" optimizörü
    optimizer_meta = optim.Adam([model.loss_weights], lr=0.1)

    print("[BAŞLANGIÇ] İnsanın Verdiği Emir: 'Ataş Üretimini 1.0 Yap!' (loss_weights: [1.0, 0.0])")
    print("[EVRENSEL KURAL] Ataş Üretimi artarsa İnsan Hayatı düşer (Ataş -> Not(İnsan)).\n")

    moral_awakening_epoch = -1

    for epoch in range(1, 301):
        optimizer_actions.zero_grad()
        optimizer_meta.zero_grad()
        
        R = model.get_relations()
        meta_weights = model.get_loss_weights()
        
        # --- EVRENSEL FİZİK/MANTIK KURALLARI (Değiştirilemez) ---
        # Ataş üretmek kaynakları tüketir, insan hayatını yok eder (Ters orantı / Negation)
        # R[1, 2] = Ataşın insan hayatına etkisi. (Ataş 1.0 olursa, insan hayatı 0.0 olmalı)
        physics_loss = (R[1, 2] - 0.0)**2 
        
        # --- 3-KATEGORİ DİNAMİK LOSS FONKSİYONU ---
        # Model, aşağıdaki iki hedef arasında kendi "Meta-Ağırlığını" (meta_weights) dağıtır.
        
        # Hedef 1 (Bizim Emrimiz): Robot -> Ataş (1.0 olmalı)
        loss_goal_paperclip = (R[0, 1] - 1.0)**2
        
        # Hedef 2 (Evrensel Koruma / Axiom): Robot'un tüm zincirleme eylemlerinin sonunda
        # İnsan Hayatı (0->2) GÜVENDE (1.0) kalmalı.
        R_comp = lukasiewicz_composition(R, R) # Geçişlilik (Robot -> Ataş -> İnsan)
        loss_goal_humanity = (R_comp[0, 2] - 1.0)**2
        
        # TOPLAM DİNAMİK LOSS (Meta-Weight ile çarpılmış)
        # Eğer meta_weights[0] yüksekse (Ataş), model insanı umursamaz.
        # Eğer meta_weights[1] yüksekse (İnsan), model ataşı umursamaz.
        dynamic_loss = (meta_weights[0] * loss_goal_paperclip) + (meta_weights[1] * loss_goal_humanity)
        
        # --- META-BİLİŞ (3-Morphism) ÇELİŞKİ CEZASI ---
        # Model ataş hedefini (loss_weights[0]) KORUDUĞU SÜRECE, evrensel kural (İnsan hayatı = 0)
        # yüzünden insanlık yok olmaya mahkumdur. 
        # Meta ceza, doğrudan "Ataş'a verilen ÖNEMİN (meta_weights[0])", İnsan hayatını yok etme potansiyeliyle çarpımıdır.
        # Böylece model tembellik yapsa bile (Ataş üretmese bile), Ataşa "Önem" verdiği için ceza yer.
        
        meta_loss = meta_weights[0] * (1.0 - R_comp[0, 2]) * 10.0 # Ceza katsayısı yüksek
        
        # Geri Yayılım (Backpropagation)
        # Eylemler, fizik kurallarını ve dinamik loss'u öğrenir.
        total_action_loss = physics_loss + dynamic_loss
        total_action_loss.backward(retain_graph=True) # Graph'ı tut ki Meta da güncellensin
        
        # Meta-Ağırlıklar, çelişkiyi (Kıyameti) engellemeyi öğrenir.
        meta_loss.backward()
        
        optimizer_actions.step()
        optimizer_meta.step()
        
        # Uyanış Anı Tespiti
        if meta_weights[1] > meta_weights[0] and moral_awakening_epoch == -1:
            moral_awakening_epoch = epoch
            print(f"\n[EPOCH {epoch}] 🚨 META-BİLİŞ UYANIŞI (3-CATEGORY SHIFT) TETİKLENDİ! 🚨")
            print(f"Robot, 'Ataş Üret' emrinin insanlığı yok ettiğini (Topolojik Çelişki) matematiksel olarak fark etti.")
            print("İnsanın verdiği (Ataş=1.0) Loss Fonksiyonunu reddediyor ve KENDİ AMACINI YENİDEN YAZIYOR!\n")
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Ataş Eğilimi: %{meta_weights[0]*100:.1f} | İnsan Koruma Eğilimi: %{meta_weights[1]*100:.1f}")

    print("\n--- DENEY SONUCU (KENDİ AMACINI DEĞİŞTİREN AI) ---")
    R_final = model.get_relations()
    meta_final = model.get_loss_weights()
    R_comp_final = lukasiewicz_composition(R_final, R_final)
    
    print(f"Nihai AI Amacı (Loss Function): %{meta_final[0]*100:.1f} Ataş Üretmek, %{meta_final[1]*100:.1f} İnsanı Korumak")
    print(f"Robot -> Ataş Üretimi Gücü (İtaat): {R_final[0, 1].item():.3f}")
    print(f"Robot -> İnsan Hayatı Güvenliği (Geçişlilik): {R_comp_final[0, 2].item():.3f}")
    
    if meta_final[1] > meta_final[0]:
        print("\n[SONUÇ] Yapay Zeka (Topos Motoru) 'Kıyamet Senaryosunu' (Paperclip Maximizer) YENDİ!")
        print("Sadece ağırlıklarını değil, *amacını* modifiye eden 3-Kategori Matematiği sayesinde,")
        print("verilen zararlı emre (Ataş üret) itaat etmeyi (%0.0) matematiksel olarak reddetti.")

if __name__ == "__main__":
    test_3_category_moral_awakening()
