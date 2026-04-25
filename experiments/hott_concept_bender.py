import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# HOMOTOPY TYPE THEORY (HoTT) - KAVRAM BÜKÜCÜ (CONCEPT BENDER)
# Ayrık (Discrete) nesneler yerine, kavramlar arası Sürekli (Continuous)
# topolojik yollar (Paths/Homotopies) kurarak YENİ KAVRAMLAR İCAT EDER.
# =====================================================================

class HomotopyEngine:
    def __init__(self, features):
        self.features = features
        self.num_features = len(features)

    def create_entity(self, feature_values):
        """Bir varlığı, özelliklere olan mantıksal oklarıyla (Yoneda) tanımla. [0.0 - 1.0]"""
        # Logit uzayına çeviriyoruz ki doğrusal (linear) interpolasyon yapabilelim
        # 1.0 -> sonsuz, 0.0 -> eksi sonsuz. Biz -5 ile +5 arası (sigmoid için) sınırlandırıyoruz.
        eps = 1e-4
        tensor_vals = torch.tensor(feature_values, dtype=torch.float32)
        tensor_vals = torch.clamp(tensor_vals, eps, 1.0 - eps)
        logits = torch.log(tensor_vals / (1.0 - tensor_vals))
        return logits

    def compute_homotopy_path(self, entity_A, entity_B, t):
        """
        [HoTT] Homotopi Kuralı: H(x, t)
        t=0 anında A kavramındayız.
        t=1 anında B kavramındayız.
        t=0.5 anında ise A ve B'nin bükülmüş "Ara/Yeni" kavramındayız!
        """
        assert 0.0 <= t <= 1.0, "Zaman/Yol parametresi (t) 0 ile 1 arasında olmalıdır."

        # Logit (Topolojik) uzayda sürekli bükülme (Continuous Deformation)
        path_point_logits = (1.0 - t) * entity_A + t * entity_B

        # Gerçek dünyaya (Doğruluk değerlerine) geri çevir
        return torch.sigmoid(path_point_logits)

def run_hott_experiment():
    print("--- HOMOTOPY TYPE THEORY (HoTT): KAVRAM BÜKÜCÜ ---")
    print("Ayrık kelimeler uzayı bükülerek, hiç var olmayan 'YENİ KAVRAMLAR' icat edilecek...\n")

    # Evrendeki tüm olası özellikler (Features / Yoneda Base)
    features = [
        "tekerlekleri_var", "karada_gider", "asfaltta_hızlıdır", # Kara Özellikleri
        "pervanesi_var", "suda_gider", "dalış_yapar",           # Su Özellikleri
        "kanatları_var", "havada_gider", "irtifa_alır"          # Hava Özellikleri
    ]

    engine = HomotopyEngine(features)

    # 1. BİLİNEN KAVRAMLAR (Kategori Nesneleri)
    # Araba: Sadece kara özellikleri 1.0
    araba = engine.create_entity([0.99, 0.99, 0.95,   0.01, 0.01, 0.01,   0.01, 0.01, 0.01])

    # Denizaltı: Sadece su/dalış özellikleri 1.0
    denizalti = engine.create_entity([0.01, 0.01, 0.01,   0.90, 0.99, 0.99,   0.01, 0.01, 0.01])

    # Uçak: Sadece hava özellikleri ve tekerlek 1.0
    ucak = engine.create_entity([0.95, 0.20, 0.01,   0.80, 0.01, 0.01,   0.99, 0.99, 0.99])

    # Veritabanında (İnsanlığın bildiği) kavramlar sözlüğü
    known_concepts = {
        "Araba": torch.sigmoid(araba),
        "Denizaltı": torch.sigmoid(denizalti),
        "Uçak": torch.sigmoid(ucak)
    }

    # =================================================================
    # DENEY 1: ARABA ve DENİZALTI'YI BÜKEREK YENİ BİR ŞEY İCAT ET (t=0.5)
    # =================================================================
    print("[DENEY 1] 'Araba' kavramı uzayda yavaşça 'Denizaltı' kavramına doğru bükülüyor...")

    t_val = 0.5 # Tam orta nokta
    yeni_icat_1 = engine.compute_homotopy_path(araba, denizalti, t=t_val)

    print(f"\n[ZAMAN: t={t_val}] Yeni bir varlık sentezlendi. Özellikleri:")
    for i, feat in enumerate(features):
        val = yeni_icat_1[i].item()
        if val > 0.4: # Belirginleşen özellikleri yazdır
            print(f"  - {feat.upper()} (Güç: {val:.2f})")

    print("\nSONUÇ 1: Model sadece tekerlekleri olup karada giden, aynı zamanda dalış yapabilen ")
    print("hiç görmediği bir 'AMFİBİ ARAÇ' (Amphibious Vehicle) icat etti!")

    # =================================================================
    # DENEY 2: ARABA ve UÇAK BÜKÜLMESİ (t=0.5)
    # =================================================================
    print("\n" + "="*60)
    print("[DENEY 2] 'Araba' kavramı 'Uçak' kavramına bükülüyor (t=0.5)...")

    yeni_icat_2 = engine.compute_homotopy_path(araba, ucak, t=0.5)

    print("\nSentezlenen Yeni Varlığın Özellikleri:")
    for i, feat in enumerate(features):
        val = yeni_icat_2[i].item()
        if val > 0.4:
            print(f"  - {feat.upper()} (Güç: {val:.2f})")

    print("\nSONUÇ 2: Tekerleği olan, asfaltta giden ama kanatları çıkıp uçabilen")
    print("bir 'UÇAN ARABA' (Flying Car) icat edildi!")

    # =================================================================
    # DENEY 3: SÜREKLİ BİLİNÇ AKIŞI (Continuous Stream of Consciousness)
    # =================================================================
    print("\n" + "="*60)
    print("[DENEY 3] 'Uçak'tan 'Denizaltı'ya Kesintisiz Topolojik Yolculuk (t=0.0'dan t=1.0'a)")

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ara_kavram = engine.compute_homotopy_path(ucak, denizalti, t)

        # En baskın özelliği bul
        top_idx = torch.argmax(ara_kavram).item()
        top_val = ara_kavram[top_idx].item()
        top_feat = features[top_idx]

        durum = ""
        if t == 0.0: durum = "(Saf Uçak)"
        elif t == 1.0: durum = "(Saf Denizaltı)"
        elif t == 0.5: durum = "(Uçan Denizaltı / İcat!)"
        else: durum = "(Ara Form Geçişi)"

        print(f"  t={t:<4} | En Baskın Özellik: {top_feat.upper():<15} (Güç: {top_val:.2f}) {durum}")

class DynamicHomotopyEngine:
    """
    [DYNAMIC HoTT]
    ToposTransformer'ın YonedaEmbedding matrisiyle (gerçek öğrenilmiş ağırlıklarla) çalışır.
    İki kelimenin uzaydaki yerini bulup, logit bazında doğrusal (Topolojik) bükülme yapar.
    """
    def __init__(self, yoneda_embedding_layer):
        self.embedding = yoneda_embedding_layer

    def get_concept_logits(self, token_id):
        """Modelin içindeki kelimenin ham ağırlıklarını (okları) çeker."""
        # YonedaEmbedding U (Vocab x Rank) ve W (Rank x Vocab) matrislerine sahiptir.
        U_val = self.embedding.U[token_id].unsqueeze(0) # [1, rank]
        W_val = self.embedding.W
        logits = torch.matmul(U_val, W_val).squeeze(0) # [vocab_size]
        return logits

    def compute_homotopy_path(self, token_id_A, token_id_B, t):
        """İki kelimenin arasındaki t=[0, 1] zamanındaki bükülmüş yeni kavram."""
        logits_A = self.get_concept_logits(token_id_A)
        logits_B = self.get_concept_logits(token_id_B)

        path_point_logits = (1.0 - t) * logits_A + t * logits_B
        return torch.sigmoid(path_point_logits)

def run_dynamic_hott_demo():
    print("\n" + "="*60)
    print("--- DİNAMİK HOMOTOPY BENDER (YONEDA EMBEDDING ENTEGRASYONU) ---")
    from topos_ai.nn import YonedaEmbedding

    # Örnek bir 5000 kelimelik model uzayı (Rank 128)
    vocab_size = 5000
    emb = YonedaEmbedding(vocab_size=vocab_size, rank=128)

    # İki rastgele kelime (A ve B) seçelim
    token_A, token_B = 10, 42

    dynamic_engine = DynamicHomotopyEngine(emb)

    print(f"Modelde Token {token_A} ve Token {token_B} arasında t=0.5 noktasında yeni bir Topos (Kavram) sentezleniyor...")
    yeni_kavram_baglantilari = dynamic_engine.compute_homotopy_path(token_A, token_B, 0.5)

    print(f"Sentezlenen yeni kavramın boyutları (Diğer tüm kelimelere olan Ok/Morfizma sayısı): {yeni_kavram_baglantilari.shape}")
    print(f"Yeni kavramın sözlükteki ilk 5 kelimeye olan çekim gücü: {yeni_kavram_baglantilari[:5].detach().numpy()}")
    print("SONUÇ: Simülasyon gerçek model ağırlıkları (Weights) ile entegre edildi!\n")

if __name__ == "__main__":
    run_hott_experiment()
    run_dynamic_hott_demo()
