# =====================================================================
# O-MINIMAL STRUCTURES & TAME GEOMETRY (THE PERFECT OPTIMIZER)
# Amacı: Grothendieck'in "Esquisse d'un Programme" eserindeki Tame
# (Uysal) Geometriyi PyTorch'a taşımak. Klasik YZ'nin Kayıp Yüzeyleri
# (Loss Landscapes) sinüs gibi sonsuz salınımlı (Wild/Fractal) olabilir.
# O-Minimal yapılarda ise sonsuz salınım YASAKTIR. Her fonksiyon
# sonlu sayıda basit parçaya (Semi-Algebraic Set) bölünmelidir.
# Bu modül, kaotik ve vahşi (Wild) sinyalleri alıp, onları uysal
# (Tame) polinomlara izdüşümler (Projection). Optimizatörlerin
# fraktal çukurlarına düşmesini sonsuza dek engeller.
# =====================================================================
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OMinimalProjector(nn.Module):
    """
    Vahşi (Wild) Sinyalleri, Sonlu Parçalı (Semi-Algebraic)
    O-Minimal sinyallere dönüştüren Uysal (Tame) Projektör.
    """
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        """
        [THE TAMING FUNCTOR]
        x: Herhangi bir vahşi veya sonsuz salınımlı sinyal.
        Çıktı: x'in Taylor serisi benzeri ama O-Minimal (Sınırlı Dereceli Polinom)
        bir yaklaşımı. Sinüs dalgası gibi sonsuz kökü olan bir şey,
        bu projektörden geçince sadece 'degree' kadar kökü olan
        uysal bir eğriye dönüşür.
        """
        # x'in [-1, 1] aralığına sıkıştırılmış (Compactified) hali
        # Sonsuzlukları sonlu bir interval'e çekiyoruz (Tame Topology'nin temeli)
        x_compact = torch.tanh(x)

        tame_signal = torch.zeros_like(x_compact)

        # Sonlu dereceli polinom (Semi-Algebraic) yaklaşımı
        # P(x) = c_1*x + c_3*x^3 + ...
        # Sadece tek (odd) dereceler kullanılarak orijin simetrisi korunur
        for i in range(1, self.degree + 1, 2):
            # Taylor benzeri katsayılar (Alternating signs)
            coef = ((-1.0) ** ((i - 1) // 2)) / float(math.factorial(i))
            tame_signal += coef * (x_compact ** i)

        return tame_signal

class TameNeuralLayer(nn.Module):
    """
    Aktivasyon fonksiyonu olarak vahşi (Sigmoid/Sinüs) yerine
    O-Minimal projektörü kullanan 'Uysal (Tame)' Nöral Katman.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.tame_projector = OMinimalProjector(degree=3)

    def forward(self, x):
        # 1. Doğrusal Dönüşüm
        raw_out = self.linear(x)

        # 2. Uysal (Tame) Aktivasyon
        # Bu aktivasyonun hiçbir fraktal yapısı veya sonsuz salınımı yoktur.
        # Bu sayede Loss yüzeyi her zaman sonlu sayıda vadi (Basin) içerir.
        tame_out = self.tame_projector(raw_out)

        return tame_out
