import torch
import torch.nn as nn

# =====================================================================
# THE YONEDA LEMMA (CATEGORICAL EMBEDDINGS)
# Amacı: Kategori Teorisinin en önemli teoremi olan Yoneda Lemma'yı
# PyTorch'a taşımak.
# Teorem: X ≅ Hom(-, X). Bir X objesi (Örn: Devasa bir Veri Vektörü),
# kendisinin ne olduğuyla değil; Evrendeki diğer referans noktalarına
# (Probes / A) olan uzaklıkları ve ilişkileriyle (Morphisms) %100
# kusursuz tanımlanır.
# Bu modül, objelerin içsel özelliklerini (Features) SİLER ve
# onları sadece birer 'İlişki Vektörüne (Contravariant Functor)'
# dönüştürür.
# =====================================================================

class YonedaUniverse(nn.Module):
    """
    Evrendeki referans noktalarını (Probes/A) temsil eder.
    Yoneda Lemma'nın "Hom(-, X)" kısmındaki "-" (Her şey) burasıdır.
    """
    def __init__(self, num_probes, dim):
        super().__init__()
        # Evrenin farklı köşelerine dağılmış Referans Noktaları
        # Bunlara "Sonda (Probe)" veya "Gözlemci" diyebiliriz.
        self.probes = nn.Parameter(torch.randn(num_probes, dim))

    def get_morphisms(self, X):
        """
        [THE HOM-FUNCTOR: Hom(A, X)]
        X'in (Bilinmeyen Obje), evrendeki tüm Probe'lara (A) olan 
        Uzaklık / Benzerlik ilişkilerini (Morfizmalarını) çıkarır.
        X'in KENDİ ÖZELLİKLERİ SİLİNİR, geriye sadece "İlişki Ağı" kalır.
        """
        # X: [Batch, Dim], Probes: [Num_Probes, Dim]
        # Morfizma = Uzaklık kareleri (Gradientler kaybolmasın)
        # Çıktı: [Batch, Num_Probes]
        distances = torch.cdist(X, self.probes, p=2) ** 2
        return distances

class YonedaReconstructor(nn.Module):
    """
    [THE YONEDA INVERSE: Hom(-, X) -> X]
    Yoneda Lemma'nın asıl mucizesi: Sadece "İlişki Ağını" (Morfizmaları)
    alarak, objenin "Gerçek Fiziksel Koordinatlarını (X)" GERİ BULMAK!
    """
    def __init__(self, num_probes, dim):
        super().__init__()
        # Makine "Gerçekliği" tahmin etmeye çalışacak
        self.estimated_X = nn.Parameter(torch.zeros(1, dim))

    def forward(self, true_morphisms, universe: YonedaUniverse):
        """
        Tahmin edilen X'in morfizmalarını, Gerçek X'in morfizmalarıyla
        kıyaslar. Eğer Morfizmalar eşitse (İzomorfik Functor), 
        Yoneda Lemma'ya göre OBJELER KESİNLİKLE EŞİT OLMALIDIR!
        """
        # Makinenin tahmin ettiği uydurma X'in Evrenle olan ilişkisi
        estimated_morphisms = universe.get_morphisms(self.estimated_X)
        
        # Eğer bu iki ilişki ağı birbirine uyarsa, X'i kusursuz bulduk demektir!
        loss = torch.nn.functional.mse_loss(estimated_morphisms, true_morphisms)
        return loss, self.estimated_X
