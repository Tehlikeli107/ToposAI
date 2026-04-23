import torch
import torch.nn as nn

# =====================================================================
# THE ELEMENTARY TOPOS AXIOMS (CARTESIAN CLOSED CATEGORY)
# Amacı: Bir uzayın "Topos" olabilmesi için 3 temel şartı sağlaması 
# gerekir: Sonlu Limitler (Çarpım), Üstel Objeler (Exponentials) ve
# Subobject Classifier (Ω).
# Bu modül, PyTorch tensörlerini birer 'Topos Objesi' olarak kabul
# ederek, klasik matris çarpımları yerine Kategorik Çarpım (Product),
# Kategorik Toplam (Coproduct) ve Üstel Obje (Internal Hom) operasyonlarını
# Heyting Mantığı (Gödel T-Norm) ile tanımlar.
# =====================================================================

class ElementaryTopos(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 1. INITIAL OBJECT (0) - 'Big Bang' veya 'Kesin Yanlış'
        # Her objeye ondan sadece TEK BİR ok (Morphism) gider.
        self.initial_object = torch.zeros(dim)
        
        # 2. TERMINAL OBJECT (1) - 'Kara Delik' veya 'Kesin Doğru'
        # Her objeden ona sadece TEK BİR ok gider.
        self.terminal_object = torch.ones(dim)

    def product(self, X, Y):
        """
        [CATEGORICAL PRODUCT: X × Y]
        Mantıksal karşılığı: X AND Y.
        Topolojik karşılığı: Kesişim (Greatest Lower Bound).
        Gödel T-Norm (Min) ile modellenir.
        """
        return torch.minimum(X, Y)

    def coproduct(self, X, Y):
        """
        [CATEGORICAL COPRODUCT: X + Y]
        Mantıksal karşılığı: X OR Y.
        Topolojik karşılığı: Birleşim (Least Upper Bound).
        Gödel T-Conorm (Max) ile modellenir.
        """
        return torch.maximum(X, Y)

    def exponential(self, Y, Z):
        """
        [EXPONENTIAL OBJECT: Z^Y (Internal Hom)]
        Bir Topos'un en güçlü özelliğidir (Cartesian Closed).
        Fonksiyonların/Okların (Y -> Z) KENDİSİNİN DE BİR OBJE OLMASIDIR!
        Mantıksal karşılığı: Y => Z (Y, Z'yi gerektirir).
        Heyting Cebirindeki Implication ile modellenir:
        Eğer Y <= Z ise 1.0 (Kesin doğru), değilse Z.
        """
        condition = (Y <= Z).float()
        return condition * 1.0 + (1.0 - condition) * Z

    def subobject_classifier(self, X, Y):
        """
        [SUBOBJECT CLASSIFIER (Ω): X -> Ω]
        X'in Y'ye ne kadar 'Dahil (Subobject)' olduğunu ölçen morfizma.
        Kısmi kapsama (Fuzzy Subsethood).
        """
        # X <= Y ise 1.0, değilse Y
        return self.exponential(X, Y)

    def check_morphism(self, A, B):
        """
        [CATEGORICAL MORPHISM: A -> B]
        A'dan B'ye yasal bir ok (Morfizma) var mı?
        Heyting Mantığında (A <= B) durumu morfizmanın varlığını ispatlar.
        """
        # A, B'den küçük eşitse (Topolojik kapsama), aralarında 1.0'lık (Kesin) bir Ok vardır.
        return torch.all(A <= B).item()
