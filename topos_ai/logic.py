import torch
import torch.nn as nn

# =====================================================================
# TOPOI INTERNAL LOGIC: THE SUBOBJECT CLASSIFIER (Ω) & HEYTING ALGEBRA
# Amacı: Toposların içsel mantığı klasik (Boolean) değildir. Sezgisel
# (Intuitionistic) bir yapı olan Heyting Cebiri'ne uyar.
# Klasik mantıkta 'Değilinin Değili Kendisidir' (~~A == A) ve
# 'Bir şey ya Doğrudur ya Yanlış' (A V ~A == True).
# Sürekli Toposlarda (Continuous Topoi) bu kurallar ÇÖKER.
# Biz burada yapay zekaya klasik Olasılık (Probability) değil, 
# Gödel T-Norm'unu kullanarak Kesin Topolojik Mantık (Heyting) öğretiyoruz.
# =====================================================================

class SubobjectClassifier(nn.Module):
    """
    Topos Axiom 3: Her Topos'un bir Subobject Classifier'ı (Ω) vardır.
    Bu sınıf, tensörlerin (Ağırlıkların) Topos içindeki 'Hakikat Değerlerini'
    (Truth Values) hesaplar.
    """
    def __init__(self):
        super().__init__()
        # Topos'taki 'True' (T) terminal objeden Ω'ya giden morfizmadır.
        self.truth_morphism = 1.0
        self.false_morphism = 0.0

    def logical_and(self, A, B):
        """Topolojik Kesişim (Meet / Infimum): Gödel T-Norm -> min(A, B)"""
        return torch.minimum(A, B)

    def logical_or(self, A, B):
        """Topolojik Birleşim (Join / Supremum): Gödel T-Conorm -> max(A, B)"""
        return torch.maximum(A, B)

    def implies(self, A, B, hardness=50.0):
        """
        [HEYTING IMPLICATION: A => B]
        Sezgisel mantığın kalbi! "A, B'yi gerektirir" morfizması.
        A <= B ise sonuç 1.0'a (True) yaklaşır.
        A > B ise sonuç B'nin kendisine eşit olur (Lukasiewicz gibi 1-A+B DEĞİL!).
        """
        # Geri yayılım (Backprop) için pürüzsüz (Smooth) sigmoid yaklaşımı
        # Bu sayede her durumda A ve B gradyan (türev) alır.
        sigma = torch.sigmoid((B - A) * hardness)
        return sigma + (1.0 - sigma) * B

    def logical_not(self, A, hardness=50.0):
        """
        [INTUITIONISTIC NEGATION: ~A]
        Toposlarda "Değil (Not)" kavramı, A'nın "Yanlış"ı (0.0) gerektirmesidir: (A => False)
        Bu çok serttir! Bir şeyin değili, sadece o şey TAMAMEN YANLIŞ (0.0) ise True (1.0) olur.
        İçinde en ufak bir doğruluk payı (Örn: 0.1) varsa, değili ANINDA 0.0 (False) olur!
        (Klasik olasılıktaki 1.0 - 0.1 = 0.9 mantığını reddeder).
        """
        # Gradient kesilmemesi için pürüzsüz yaklaşım (A == 0 ise 1, A > 0 ise 0)
        return torch.sigmoid(-A * hardness)

class HeytingNeuralLayer(nn.Module):
    """
    Klasik Linear(x) + ReLU yerine, veriyi Toposların Subobject Classifier'ı (Ω)
    üzerinden süzerek mantıksal çıkarım yapan aksiyomatik katman.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        # Ağırlıklar (Premises/Öncüller) [0, 1] aralığında olmalıdır
        self.weight = nn.Parameter(torch.rand(out_features, in_features))
        self.omega = SubobjectClassifier()

    def forward(self, x):
        # x: [Batch, in_features], weight: [out_features, in_features]
        # x'i [0,1] aralığına sıkıştır (Fiziksel uzaydan Mantıksal uzaya geçiş)
        x_logical = torch.sigmoid(x)
        w_logical = torch.sigmoid(self.weight)
        
        # O(batch x out_features) Python döngüsü YERİNE, Broadcasting ile Vektörize İşlem (GPU)
        x_exp = x_logical.unsqueeze(1) # [Batch, 1, in_features]
        w_exp = w_logical.unsqueeze(0) # [1, out_features, in_features]
        
        # Her bir nöron (Çıktı), girdilerin ağırlıkları "gerektirmesi" (Implication)
        # üzerine kurulu bir Heyting Mantık Kapısıdır.
        # Nöron j = AND_i ( x_i => w_ji )
        
        # [Batch, out_features, in_features] boyutunda tüm implicationlar
        implications = self.omega.implies(x_exp, w_exp)
        
        # Tüm öncüllerin Topolojik Kesişimi (Meet/Min) in_features boyutunda
        out = implications.min(dim=-1).values # [Batch, out_features]
                
        return out
