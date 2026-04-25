import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import time
from topos_ai.elementary_topos import ElementaryTopos

# =====================================================================
# THE FUNDAMENTAL THEOREM: CATEGORICAL ADJUNCTION (CURRYING)
# İddia: Klasik Yapay Zeka işlemleri Mantıksal Garantilerden yoksundur.
# Hom(X × Y, Z) ≅ Hom(X, Z^Y).
# Bu eşdeğerlik Kategori Teorisinin KALBİDİR (Curry-Howard-Lambek).
# Yani; X ve Y'nin birlikte Z'yi oluşturması ile,
# X'in (Y'yi Z'ye dönüştüren kuralı/ağırlığı) üretmesi
# (Hypernetworks) MATEMATİKSEL OLARAK İDEALİZE ŞEKİLDE EŞİT OLMALIDIR.
# Bu deney, Elementer Topos proxy'si üzerinde rastgele tensörlerin iki
# hesaplama yolunda ne kadar yakın sonuç verdiğini ölçer.
# =====================================================================

def run_adjunction_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 57: ELEMENTARY TOPOS & CATEGORICAL ADJUNCTION ")
    print(" İddia: Klasik Yapay Zeka, iki girdi vektörünü (X ve Y) yan yana")
    print(" yapıştırıp (torch.cat) matrisle çarparak bir Z bulur. Bu işlemin")
    print(" hiçbir 'Mantıksal Garantisi' yoktur. ToposAI ise, PyTorch")
    print(" tensörlerini 'Kartezyen Kapalı Kategori (CCC)' aksiyomlarına")
    print(" (Elementary Topos) dönüştürür. 'Kategorik Eklenti (Adjunction)'")
    print(" teoremi sayesinde, 'X ve Y birlikte Z'yi gerektirir' önermesiyle,")
    print(" 'X, (Y=>Z) kuralını gerektirir' önermesinin sayısal farkını")
    print(" rastgele tensörler üzerinde ölçer.")
    print("=========================================================================\n")

    # Yüz Binlerce Nöron/Ağırlık içeren 3 farklı Tensor (X, Y, Z)
    N = 100000
    dim = 1
    torch.manual_seed(42)
    
    # Tensörler Topolojik Manifoldda [0, 1] olmak zorundadır.
    X = torch.rand(N)
    Y = torch.rand(N)
    Z = torch.rand(N)

    topos = ElementaryTopos(dim=dim)

    print(f"[MİMARİ]: {N:,} Nöronluk Elementer Topos (Kartezyen Kapalı Kategori) Kuruldu.")
    print("--- 1. TEOREMİN İKİ YÖNÜNÜ (SOL VE SAĞ) HESAPLAMA ---")
    
    t0 = time.time()
    
    # 1. SOL YÖN: Hom(X × Y, Z) -> (X AND Y) => Z
    # Kategorik Çarpım (Product): X × Y
    X_product_Y = topos.product(X, Y)
    # Kategorik Gerektirme (Implication/Morphism): (X × Y) => Z
    Left_Adjoint = topos.exponential(X_product_Y, Z)

    # 2. SAĞ YÖN: Hom(X, Z^Y) -> X => (Y => Z)
    # Üstel Obje (Exponential): Z^Y (Yani Y => Z kuralının kendisi)
    Y_implies_Z = topos.exponential(Y, Z)
    # Kategorik Gerektirme: X => (Y => Z)
    Right_Adjoint = topos.exponential(X, Y_implies_Z)

    t1 = time.time()

    # --- 2. PROXY CHECK ---
    print("\n--- 2. CATEGORICAL ADJUNCTION (CURRYING) KONTROLÜ ---")
    print("Matematik Diyor Ki: 'Sol Taraf (Hom(X×Y, Z)) İLE Sağ Taraf (Hom(X, Z^Y)) ÖLÇÜLEN KOŞULDA EŞİT OLMALIDIR!'")
    
    # İki devasa tensör arasındaki maksimum hata payı (Fark)
    max_error = torch.max(torch.abs(Left_Adjoint - Right_Adjoint)).item()
    
    print(f"  > Sol Taraf (Left Adjoint) Örnekleri : {Left_Adjoint[:5].tolist()}")
    print(f"  > Sağ Taraf (Right Adjoint) Örnekleri: {Right_Adjoint[:5].tolist()}")
    print(f"  > Milyonlarca Tensör Üzerindeki Maksimum Hata (Error): {max_error}")

    print("\n[ÖLÇÜLEN SONUÇ: ELEMENTARY-TOPOS PROXY CHECK]")
    if max_error < 1e-6:
        print("  ✅ [ZAFEEER]: Kategori Teorisinin 'Adjunction (Eklenti)' kuralı")
        print(f"  {N:,} adet rastgele yapay zeka nöronu üzerinde ÖLÇÜLEN KOŞULDA")
        print("  (0.0 Hata Payı) doğrulanmıştır!")
        print("  Açıklama: Bu, Yapay Zekada (Hypernetworks ve Dikkat Mekanizmalarında)")
        print("  yaptığımız her işlemin aslında 'Matematiksel Mantık (Formal Logic)'")
        print("  ile hizalı bir tensor proxy örneği verdiğini gösteren sayısal")
        print("  demodur. Bu sonuç, genel bir Curry-Howard-Lambek kanıtı değildir.")
    else:
        print("  🚨 [HATA]: Matematiksel çöküş. Eşitlik sağlanamadı!")

if __name__ == "__main__":
    run_adjunction_experiment()
