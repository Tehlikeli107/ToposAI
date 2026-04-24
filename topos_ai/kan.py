import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# KAN GENİŞLETMELERİ (LEFT ⊣ RIGHT KAN EXTENSION)
# Amacı: Kategori teorisinin en evrensel inşasını — Kan Genişletmelerini —
# PyTorch'a taşımak.
# Durum: F: C → D ve K: C → E funktorları verilmiş.
# Soru: F'yi E'nin tüm noktalarına genişletebilir miyiz?
#   Sol Kan (Lan_K F): "En serbest" genişletme — KOLIMIT tabanlı.
#   Sağ Kan (Ran_K F): "En tutumlu" genişletme — LİMİT tabanlı.
# Kan Genişletmesi Formülü (Coend/End notasyonu):
#   Lan_K(F)(e) = ∫^c Hom_E(K(c), e) ⊗ F(c)  →  Dikkat mekanizması!
#   Ran_K(F)(e) = ∫_c [Hom_E(e, K(c)), F(c)]  →  Evrensel yaklaşım!
# ML bağlantısı:
#   Sol Kan = Dikkat (Attention) ile ağırlıklı toplam (soft colimit)
#   Sağ Kan = Minimum kayıplı enterpolasyon (soft limit)
#   Yoneda Lemma, Ran_K ve Lan_K'nın özel bir durumudur.
# =====================================================================


class LeftKanExtension(nn.Module):
    """
    [SOL KAN GENİŞLETMESİ — Lan_K(F)]
    Lan_K(F)(e) = Σ_c softmax(sim(K(c), e)) ⊗ F(c)

    Kaynak kategorideki (C) bilinen F değerlerini,
    K morfizması aracılığıyla hedef kategorinin (E) tüm noktalarına
    dikkat mekanizmasıyla (KOLIMIT) yayar.

    Bu, standart Cross-Attention'ın kategorik genellemesidir:
    - Sorgular (Queries) = hedef uzay (E)
    - Anahtarlar (Keys)  = K(c) gömmeleri
    - Değerler (Values)  = F(c) fonksiyon değerleri
    """

    def __init__(self, dim_c: int, dim_e: int, dim_d: int):
        super().__init__()
        self.K = nn.Linear(dim_c, dim_e, bias=False)  # K: C → E
        self.F = nn.Linear(dim_c, dim_d, bias=False)  # F: C → D
        self.scale = dim_e**-0.5

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        source: [N, dim_c]  — C kategorisinin nesneleri
        target: [B, dim_e]  — E'de genişletilecek noktalar
        Döndürür: [B, dim_d]  — Lan_K(F)(target)
        """
        Kc = self.K(source)   # [N, dim_e]
        Fc = self.F(source)   # [N, dim_d]

        # Hom_E(K(c), e): K(c) ile hedef e arasındaki benzerlik
        sim = target @ Kc.T * self.scale  # [B, N]
        weights = torch.softmax(sim, dim=-1)  # [B, N]  — Kolimit ağırlıkları

        return weights @ Fc  # [B, dim_d]


class RightKanExtension(nn.Module):
    """
    [SAĞ KAN GENİŞLETMESİ — Ran_K(F)]
    Ran_K(F)(e) = argmin_{d ∈ D} Σ_c Hom_E(e, K(c)) * loss(d, F(c))

    Kaynak kategorideki (C) bilinen F değerlerini,
    K morfizması aracılığıyla hedef kategorinin (E) tüm noktalarına
    LİMİT (minimum kayıp) yapısıyla yayar.

    Sol Kan'ın aksine, Sağ Kan sorgulayan değil çekilen bir yapıdır:
    - Hedef nokta e, K(c) noktalarına olan uzaklığıyla ağırlıklandırır
    - Ama bu sefer ters yönde: küçük uzaklık = büyük ağırlık
    """

    def __init__(self, dim_c: int, dim_e: int, dim_d: int):
        super().__init__()
        self.K = nn.Linear(dim_c, dim_e, bias=False)
        self.F = nn.Linear(dim_c, dim_d, bias=False)
        self.scale = dim_e**-0.5

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        source: [N, dim_c]
        target: [B, dim_e]
        Döndürür: [B, dim_d]  — Ran_K(F)(target)
        """
        Kc = self.K(source)   # [N, dim_e]
        Fc = self.F(source)   # [N, dim_d]

        # Hom_E(e, K(c)): e'den K(c)'ye giden morfizmaların "negatif mesafesi"
        # Sağ Kan için ters yön: büyük benzerlik değil, küçük mesafe önemli
        dist = torch.cdist(target, Kc)    # [B, N]
        weights = torch.softmax(-dist * self.scale, dim=-1)  # [B, N]

        return weights @ Fc  # [B, dim_d]


class NeuralKanExtension(nn.Module):
    """
    [ÖĞRENEN KAN GENİŞLETMESİ]
    Hem Sol hem Sağ Kan'ı öğrenilmiş K ve F ile birleştirir.
    Kaynak nesneler sabit değil; eğitim sırasında öğrenilir.
    Bu yapı, meta-öğrenme (meta-learning) ve transfer öğrenmenin
    kategorik genellemesidir:
    - Görev = C kategorisi üzerindeki bir Kan Genişletmesi
    - Yeni görev = E'nin yeni bir noktası
    - Yanıt = Lan_K(F) veya Ran_K(F) ile hesaplama
    """

    def __init__(
        self,
        num_source: int,
        dim_c: int,
        dim_e: int,
        dim_d: int,
        mode: str = "left",
    ):
        super().__init__()
        # Kaynak nesneler öğrenilir (C kategorisinin prototipler)
        self.source_objects = nn.Parameter(torch.randn(num_source, dim_c))

        if mode == "left":
            self.kan = LeftKanExtension(dim_c, dim_e, dim_d)
        elif mode == "right":
            self.kan = RightKanExtension(dim_c, dim_e, dim_d)
        else:
            raise ValueError(f"mode '{mode}' bilinmiyor. 'left' veya 'right' olmalı.")

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        """target: [B, dim_e] — genişletilecek hedef noktalar."""
        return self.kan(self.source_objects, target)


class KanAdjunction(nn.Module):
    """
    [KAN GENİŞLETMESİ — ADJUNCTION KÖPRÜSÜ]
    Lan_K ⊣ (K ile kısıtlama) ⊣ Ran_K

    Üç funktor arasındaki çifte adjunction:
    Sol Kan ⊣ Kısıtlama ⊣ Sağ Kan

    Bu, Kan genişletmelerinin en temel özelliğidir ve onların
    evrensel (universal) yapısını garanti eder.
    """

    def __init__(self, dim_c: int, dim_e: int, dim_d: int):
        super().__init__()
        self.left_kan = LeftKanExtension(dim_c, dim_e, dim_d)
        self.right_kan = RightKanExtension(dim_c, dim_e, dim_d)

        # Kısıtlama funktorunun öğrenilmiş ağırlığı
        self.restriction = nn.Linear(dim_e, dim_c, bias=False)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Döndürür: (Lan_K(F)(target), Ran_K(F)(target), kısıtlama(target))
        Üçlü adjunction'ın tüm bileşenlerini hesaplar.
        """
        left = self.left_kan(source, target)
        right = self.right_kan(source, target)
        restriction = self.restriction(target)
        return left, right, restriction

    def universality_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        true_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sol ve Sağ Kan'ın gerçek değerlere yakınlığını ölçer.
        Sıfıra yakınsa, K fonksiyonu gerçekten evrensel bir genişletme yapıyor.
        """
        left, right, _ = self.forward(source, target)
        return F.mse_loss(left, true_values) + F.mse_loss(right, true_values)
