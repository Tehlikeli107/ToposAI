import torch
import torch.nn as nn

class ToposWitness(nn.Module):
    """
    Her tanığın (Local Truth) dünyasını temsil eden matris.
    Varlıklar arası özellikleri (Relations) tutar.
    """
    def __init__(self, num_entities, num_features):
        super().__init__()
        # Tanık, varlıkların hangi özelliklere sahip olduğunu gördü?
        # Satırlar: Varlıklar (Örn: Katil), Sütunlar: Özellikler (Örn: Şapkalı)
        self.knowledge = nn.Parameter(torch.randn(num_entities, num_features))
        
    def get_truth(self):
        return torch.sigmoid(self.knowledge)

def sheaf_gluing_condition(truth_A, truth_B, threshold=0.2):
    """
    İki farklı Local Truth (Topos) birleştirilebilir mi? (Sheaf Condition)
    Eğer iki tanık aynı varlık için farklı (çelişen) özellikler iddia ediyorsa,
    yapıştırma (gluing) işlemi başarısız olur.
    
    threshold: İki tanığın aynı özellik üzerindeki farklılığının tolere edilebilir sınırı.
    """
    # Kesişim noktalarını bul: İki tanığın da üzerinde fikir belirttiği (1.0'a veya 0.0'a yakın) özellikler.
    # Eğer biri "Emin değilim" (0.5 civarı) diyorsa, bu çelişki sayılmaz.
    
    # Certainty (Eminlik): Ne kadar 0.0 veya 1.0'a yakın?
    certainty_A = torch.abs(truth_A - 0.5) * 2.0 # [0, 1] arası
    certainty_B = torch.abs(truth_B - 0.5) * 2.0
    
    # Her iki tanığın da EMİN olduğu ortak alanlar (Intersection of Open Sets)
    overlap = certainty_A * certainty_B
    
    # Bu ortak alanlarda verdikleri cevapların farkı
    disagreement = torch.abs(truth_A - truth_B)
    
    # Kesişim alanındaki çelişki (Conflict on Overlap)
    conflict_score = torch.sum(overlap * disagreement).item()
    
    if conflict_score > threshold:
        return False, conflict_score # Sheaf koşulu ihlal edildi! Yapıştırılamaz.
    else:
        return True, conflict_score # Yapıştırılabilir.

def glue_truths(truth_A, truth_B):
    """
    Çelişmeyen iki Local Truth'u birleştirerek (Global Section / Gluing)
    daha geniş bir gerçeklik yarat.
    
    Mantık (Fuzzy OR): İkisinin bilgisinin maksimumunu al (Çünkü çelişmiyorlar).
    """
    return torch.max(truth_A, truth_B)

def test_sheaf_gluing():
    entities = ["KATİL"]
    features = ["şapkalı", "hızlı_koşar", "kırmızı_ceketli", "topallar", "gözlüklü"]
    
    f_idx = {f: i for i, f in enumerate(features)}
    
    # 3 Farklı Tanık (Local Topoi)
    # Başlangıçta hepsi her şeye "Emin değilim" (0.5) diyor
    alice = ToposWitness(1, len(features))
    bob = ToposWitness(1, len(features))
    charlie = ToposWitness(1, len(features))
    
    # Modellerin Logitlerini (knowledge) manuel olarak ayarlıyoruz (Eğitim simülasyonu)
    # Logit > 2.0 ise Sigmoid ~ 1.0 (Kesin Var)
    # Logit < -2.0 ise Sigmoid ~ 0.0 (Kesin Yok)
    # Logit 0.0 ise Sigmoid = 0.5 (Bilmiyorum)
    
    with torch.no_grad():
        # ALICE: Katil şapkalıydı(1) ve hızlı koşuyordu(1). Kırmızı ceket veya topallamayı GÖRMEDİ(0.5).
        alice.knowledge.fill_(0.0) 
        alice.knowledge[0, f_idx["şapkalı"]] = 5.0
        alice.knowledge[0, f_idx["hızlı_koşar"]] = 5.0
        
        # BOB: Katil hızlı koşuyordu(1) ve kırmızı ceketliydi(1).
        bob.knowledge.fill_(0.0)
        bob.knowledge[0, f_idx["hızlı_koşar"]] = 5.0
        bob.knowledge[0, f_idx["kırmızı_ceketli"]] = 5.0
        
        # CHARLIE: Katil kırmızı ceketliydi(1) ama topallıyordu(1). Yani hızlı KOŞMUYORDU(0).
        charlie.knowledge.fill_(0.0)
        charlie.knowledge[0, f_idx["kırmızı_ceketli"]] = 5.0
        charlie.knowledge[0, f_idx["topallar"]] = 5.0
        charlie.knowledge[0, f_idx["hızlı_koşar"]] = -5.0 # HIZLI KOŞMANIN ZITTI (ÇELİŞKİ)
        
    truth_A = alice.get_truth()
    truth_B = bob.get_truth()
    truth_C = charlie.get_truth()
    
    print("--- TOPOS SHEAF GLUING (KONSENSÜS / DEDEKTİF MOTORU) ---")
    print("Standart AI: Tüm iddiaların ortalamasını alıp 'Katil yarı hızlı yarı topal koşan biridir' der.\n")
    
    # TEST 1: Alice ve Bob Anlaşabilir mi?
    print("[TEST 1] Alice ve Bob'un İfadeleri Karşılaştırılıyor...")
    can_glue, conflict = sheaf_gluing_condition(truth_A, truth_B)
    print(f"  Çelişki Skoru: {conflict:.3f}")
    if can_glue:
        global_truth = glue_truths(truth_A, truth_B)
        print("  SONUÇ: SHEAF KOŞULU SAĞLANDI! (Yapıştırılıyor)")
        print("  Sentezlenen Ortak Gerçeklik:")
        for f, idx in f_idx.items():
            if global_truth[0, idx] > 0.8:
                print(f"    - {f.upper()} (Kesin)")
    else:
        print("  SONUÇ: Çelişki var, yapıştırılamaz.")
        
    print("\n" + "="*50 + "\n")
    
    # TEST 2: Bob ve Charlie Anlaşabilir mi?
    print("[TEST 2] Bob ve Charlie'nin İfadeleri Karşılaştırılıyor...")
    can_glue, conflict = sheaf_gluing_condition(truth_B, truth_C)
    print(f"  Çelişki Skoru: {conflict:.3f}")
    if can_glue:
        print("  SONUÇ: SHEAF KOŞULU SAĞLANDI! (Yapıştırılıyor)")
    else:
        print("  SONUÇ: SHEAF KOŞULU İHLAL EDİLDİ! (Yapıştırma Reddedildi)")
        print("  Sebep: Bob 'Hızlı Koşar' derken, Charlie 'Topallar/Hızlı Koşmaz' diyor.")
        print("  Ağ bu iki evreni sentezlemeyi (halüsinasyon üretmeyi) matematiksel olarak durdurdu.")

if __name__ == "__main__":
    test_sheaf_gluing()
