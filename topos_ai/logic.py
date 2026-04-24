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

    def implies(self, A, B):
        """
        [HEYTING IMPLICATION: A => B]
        Sezgisel mantığın kalbi! "A, B'yi gerektirir" morfizması.
        A <= B ise sonuç 1.0 (True) olur.
        A > B ise sonuç B'nin kendisine eşit olur (Lukasiewicz gibi 1-A+B DEĞİL!).
        Bu, uzayın topolojik açık kümelerini (Open Sets) ifade eder.
        """
        # Matematiksel olarak kesin koşul (İleri geçiş için)
        condition_exact = (A <= B).float()
        
        # Geri yayılım (Backprop) için yumuşatılmış türevlenebilir yüzey
        tau = 50.0
        condition_soft = torch.sigmoid(tau * (B - A))
        
        # Straight-Through Estimator (STE)
        # İleri geçişte KESİN (Exact) Mantık, Geri yayılımda YUMUŞAK (Soft) Türev!
        condition = condition_exact.detach() - condition_soft.detach() + condition_soft
        
        return condition * 1.0 + (1.0 - condition) * B

    def logical_not(self, A):
        """
        [INTUITIONISTIC NEGATION: ~A]
        Toposlarda "Değil (Not)" kavramı, A'nın "Yanlış"ı (0.0) gerektirmesidir: (A => False)
        Bu çok serttir! Bir şeyin değili, sadece o şey TAMAMEN YANLIŞ (0.0) ise True (1.0) olur.
        İçinde en ufak bir doğruluk payı (Örn: 0.1) varsa, değili ANINDA 0.0 (False) olur!
        (Klasik olasılıktaki 1.0 - 0.1 = 0.9 mantığını reddeder).
        """
        false_tensor = torch.zeros_like(A)
        return self.implies(A, false_tensor)

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
        
        batch_size = x.size(0)
        out_features = self.weight.size(0)
        out = torch.zeros(batch_size, out_features, device=x.device)
        
        # Her bir nöron (Çıktı), girdilerin ağırlıkları "gerektirmesi" (Implication)
        # üzerine kurulu bir Heyting Mantık Kapısıdır.
        # Nöron j = AND_i ( x_i => w_ji )
        for b in range(batch_size):
            for j in range(out_features):
                # x'in w'yi gerektirmesi (Eğer Girdi, Beklentiden küçük eşitse sorun yok)
                implications = self.omega.implies(x_logical[b], w_logical[j])
                # Tüm öncüllerin Topolojik Kesişimi (Meet/Min)
                out[b, j] = torch.min(implications)
                
        return out
