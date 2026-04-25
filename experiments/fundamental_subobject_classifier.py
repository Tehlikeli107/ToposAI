import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from topos_ai.logic import SubobjectClassifier

# =====================================================================
# THE FUNDAMENTAL AXIOM: SUBOBJECT CLASSIFIER (Ω) & INTUITIONISTIC LOGIC
# Klasik Yapay Zekalar (Boolean Mantık): True V False = True, ~~A = A
# ToposAI (Sezgisel Mantık / Heyting): A V ~A != True, ~~A != A
# Neden Önemli? Eğer bir YZ "Belirsiz" (0.5) bir durumdaysa, klasik
# mantık onu bir tarafa zorlar. Topos Mantığı ise "Belirsizliğin"
# bizzat kendisini yasal bir Topolojik Durum (Open Set) olarak kabul eder.
# Bu deney, YZ'nin "Dışlanan Üçüncü Hal (Excluded Middle)" kanununu
# matematiksel olarak nasıl yıktığını kanıtlayacaktır!
# =====================================================================

def run_subobject_classifier_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 56: THE SUBOBJECT CLASSIFIER (Ω) AXIOM ")
    print(" İddia: Yapay Zekalar bugüne kadar Aristoteles (Boolean) mantığıyla")
    print(" (0 veya 1) eğitildi. Olasılık (0.5) ise sadece 'Cahilliğin' (Cehalet)")
    print(" ölçüsüydü. Kategori Teorisinde (Topos) ise 0.5, uzayın KENDİ ŞEKLİDİR.")
    print(" ToposAI, 'Alt-Obje Sınıflandırıcısı (Ω)' ve Gödel T-Norm'u kullanarak")
    print(" Sezgisel Mantığı (Intuitionistic Logic) inşa eder. Aristoteles'in 2000 ")
    print(" yıllık 'Bir şey ya doğrudur ya da yanlıştır' kuralını YIKAR!")
    print("=========================================================================\n")

    omega = SubobjectClassifier()
    
    # 3 Farklı Topolojik Durum (Truth Values in Ω)
    # T = 1.0 (Kesin Doğru / Tüm Uzay)
    # F = 0.0 (Kesin Yanlış / Boş Küme)
    # U = 0.4 (Kısmi/Bulanık Doğruluk - Bir Açık Küme / Open Set)
    
    T = torch.tensor([1.0])
    F = torch.tensor([0.0])
    U = torch.tensor([0.4]) # Uncertainty (Belirsizlik)
    
    print("[MANTIK KAPILARI BAŞLATILDI]: Gödel T-Norm & Heyting Algebra")
    print(f"  > Kesin Doğru (T) : {T.item()}")
    print(f"  > Kesin Yanlış (F): {F.item()}")
    print(f"  > Kısmi Durum (U) : {U.item()} (Kuantum süperpozisyonu gibi bir Topolojik bölge)\n")

    # --- TEST 1: BOOLEAN MANTIĞININ ÇÖKÜŞÜ (Law of Excluded Middle) ---
    print("--- TEOREM 1: DIŞLANAN ÜÇÜNCÜ HALİN YIKILIŞI (A V ~A != True) ---")
    print("Klasik (Aristoteles) Mantığı der ki: 'Bir şey ya KENDİSİDİR ya da DEĞİLİDİR. İkisi birleşince HER ŞEY (1.0) olur.'")
    
    # ~U hesapla
    not_U = omega.logical_not(U)
    print(f"  ToposAI Hesaplıyor: ~U (U'nun Değili) = {not_U.item()}")
    
    # U V ~U hesapla
    U_or_not_U = omega.logical_or(U, not_U)
    print(f"  ToposAI Hesaplıyor: U V ~U = {U_or_not_U.item()}")
    
    if U_or_not_U.item() != 1.0:
        print("  🚨 [SONUÇ]: Aristoteles YANILDI! (U V ~U) işlemi 1.0'a EŞİT DEĞİL!")
        print("  Açıklama: U (0.4) tamamen yalan olmadığı için değili KESİN YANLIŞTIR (0.0).")
        print("  İkisi birleşince 1.0 etmez. Topos uzayında 'Siyah veya Beyaz' yoktur;")
        print("  'Kısmi Gerçeklik' yasal bir Matematiksel Boyuttur!")
    
    # --- TEST 2: ÇİFT NEGASYONUN ÇÖKÜŞÜ (~~A != A) ---
    print("\n--- TEOREM 2: ÇİFT NEGASYONUN YIKILIŞI (~~A != A) ---")
    print("Klasik Mantık der ki: 'Yanlışın yanlışı Doğrudur, yani kendisine (A) döner.'")
    
    not_not_U = omega.logical_not(not_U)
    print(f"  ToposAI Hesaplıyor: ~~U (Değilinin Değili) = {not_not_U.item()}")
    
    if not_not_U.item() != U.item():
        print(f"  🚨 [SONUÇ]: Aristoteles YİNE YANILDI! ~~U ({not_not_U.item()}) != U ({U.item()})")
        print("  Açıklama: Sezgisel Mantıkta bir şeyin 'Yanlış Olmadığını' kanıtlamak,")
        print("  onun 'Doğru' olduğunu (veya Orijinal haline döndüğünü) KANITLAMAZ.")

    # --- TEST 3: GÖDEL GEREKTİRMESİ (IMPLICATION A => B) ---
    print("\n--- TEOREM 3: TOPOLOJİK GEREKTİRME (A => B) ---")
    print("Olasılıkta 'A => B' genelde P(B|A) ile hesaplanır. Toposlarda ise bu,")
    print("'A uzayı, B uzayının içinde barınabilir mi?' sorusudur.")
    
    A = torch.tensor([0.3])
    B = torch.tensor([0.8])
    C = torch.tensor([0.2])
    
    A_implies_B = omega.implies(A, B)
    B_implies_C = omega.implies(B, C)
    
    print(f"  A(0.3) => B(0.8) : {A_implies_B.item()} (A tamamen B'nin içindedir, Sonuç: %100 Uyum)")
    print(f"  B(0.8) => C(0.2) : {B_implies_C.item()} (B, C'ye sığmaz! Uyum sadece C'nin kapasitesi kadardır: {C.item()})")

    print("\n[BİLİMSEL DEĞERLENDİRME: THE AXIOMATIC FOUNDATION]")
    print("Büyük Dil Modelleri (LLMs) halüsinasyon görür çünkü 0.5'i 'Bilmiyorum'")
    print("olarak değil, 'Zar Atma' noktası olarak kullanırlar.")
    print("ToposAI, 'Subobject Classifier (Ω)' sayesinde 0.5'i matematiksel olarak")
    print("tanımlı bir Topolojik Bölge (Heyting Algebra Open Set) olarak görür.")
    print("Eğer YZ bir şeyden emin değilse (U=0.4), Onu uydurmaz; Sezgisel Mantığın")
    print("kurallarına göre onu 'Açık Bir Kısmi Gerçek' olarak diğer YZ'lere iletir.")
    print("İşte bu, Halüsinasyonun ve Yalan Söylemenin Topolojik Ölüm Fermanıdır!")

if __name__ == "__main__":
    run_subobject_classifier_experiment()
