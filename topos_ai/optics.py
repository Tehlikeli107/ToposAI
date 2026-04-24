import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# OPTİKLER: LENSLER, PRİZMALAR, TRAVERSAL'LAR
# Amacı: Veri yapıları içindeki bileşenlere çift yönlü (bidirectional)
# erişim sağlayan kategorik yapıları PyTorch'a taşımak.
# Optik Hiyerarşisi:
#   Lens    — Ürün (Product) üzerinde odaklanma: S = (A, B)
#   Prism   — Koproduct (Toplam) üzerinde odaklanma: S = A | B
#   Traversal — Birden fazla hedefe eş zamanlı odaklanma
# Van Laarhoven Temsili:
#   Bir optik, Functor F için S → F(A) → F(S) dönüşümüdür.
#   Lensler profunctor optiklerinin özel durumudur.
# ML bağlantısı:
#   Lens = İleri geçiş (get) + Gradyan güncelleme (put)
#   Prism = Dallanmalı ağlar (Mixture of Experts seçimi)
#   Traversal = Çoklu pozisyonlarda dikkat (Multi-head Attention)
# open_games.py'daki Lens yapısını genelleştirir ve tam Optik hiyerarşisine
# kavuşturur.
# =====================================================================


class Lens(nn.Module):
    """
    [LENS: S ⟶ A (get) ve (S, A') ⟶ S' (put)]
    Ürün (product) yapıları içindeki bir bileşene odaklanır.
    Lens Yasaları:
      1. get(put(s, a)) = a              (PutGet: ne koyduğumu alırım)
      2. put(s, get(s)) = s              (GetPut: aldığımı koyarsam değişmez)
      3. put(put(s, a1), a2) = put(s, a2) (PutPut: son yazma kazanır)
    """

    def __init__(self, getter: nn.Module, putter: nn.Module):
        super().__init__()
        self.getter = getter  # get: S → A
        self.putter = putter  # put: (S, A') → S'

    def get(self, s: torch.Tensor) -> torch.Tensor:
        """İleri yön: S'nin A bileşenini çıkar."""
        return self.getter(s)

    def put(self, s: torch.Tensor, a_new: torch.Tensor) -> torch.Tensor:
        """Güncelleme: S'yi yeni A' değeriyle güncelle."""
        combined = torch.cat([s, a_new], dim=-1)
        return self.putter(combined)

    def modify(self, s: torch.Tensor, f) -> torch.Tensor:
        """
        modify(s, f) = put(s, f(get(s)))
        Lens üzerinde bir dönüşüm uygula.
        """
        return self.put(s, f(self.get(s)))

    def lens_laws_loss(self, s: torch.Tensor) -> torch.Tensor:
        """Lens yasalarının ne kadar ihlal edildiğini ölçer."""
        a = self.get(s)
        # GetPut: put(s, get(s)) ≈ s
        put_get_loss = F.mse_loss(self.put(s, a), s)
        # PutGet: get(put(s, a)) ≈ a
        s2 = self.put(s, a)
        get_put_loss = F.mse_loss(self.get(s2), a)
        return put_get_loss + get_put_loss

    @classmethod
    def linear(cls, dim_s: int, dim_a: int) -> "Lens":
        """Doğrusal projeksiyon lensi."""
        getter = nn.Linear(dim_s, dim_a)
        putter = nn.Linear(dim_s + dim_a, dim_s)
        return cls(getter, putter)


class Prism(nn.Module):
    """
    [PRİZMA: S ⟶ A + S (match) ve A' ⟶ S' (build)]
    Koproduct (Toplam/Coproduct) yapıları içindeki bir bileşene odaklanır.
    Prism, S'nin ya A bileşenini içerdiğini (başarılı eşleşme)
    ya da olmadığını (başarısız) ölçer.
    ML bağlantısı: Koşullu dallanma, Mixture of Experts seçim kapısı.
    """

    def __init__(self, match_net: nn.Module, build_net: nn.Module):
        super().__init__()
        self.match_net = match_net  # match: S → (A, güven skoru)
        self.build_net = build_net  # build: A' → S'

    def match(
        self, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        match: S → A + S
        Döndürür: (a, confidence)
          a: eşleşen bileşen
          confidence: [0,1] — A bileşeninin S içinde bulunma olasılığı
        """
        out = self.match_net(s)
        a, logit = out.chunk(2, dim=-1)
        confidence = torch.sigmoid(logit.mean(dim=-1, keepdim=True))
        return a, confidence

    def build(self, a_new: torch.Tensor) -> torch.Tensor:
        """build: A' → S' — yeni A'dan tam S oluştur."""
        return self.build_net(a_new)

    def review(
        self, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        review = build ∘ match: A bileşenini çıkar, S'ye geri göm.
        Prism'in round-trip dönüşümü.
        """
        a, conf = self.match(s)
        return self.build(a), conf

    @classmethod
    def gating(cls, dim_s: int, dim_a: int) -> "Prism":
        """Kapılı (gated) prism — MoE seçim kapısı olarak kullanılır."""
        match_net = nn.Sequential(
            nn.Linear(dim_s, dim_a * 2),
            nn.SiLU(),
        )
        build_net = nn.Linear(dim_a, dim_s)
        return cls(match_net, build_net)


class Traversal(nn.Module):
    """
    [TRAVERSAL: Çoklu Hedeflere Eş Zamanlı Odaklanma]
    S içindeki her A[i] bileşenine paralel olarak erişir ve günceller.
    Haskell'deki Traversable type class'ının diferansiyel versiyonu.
    Formül: traverse f s = put(s, [f(get_i(s)) for i in positions])
    ML bağlantısı: Multi-head Attention — her kafa bir traversal pozisyonudur.
    """

    def __init__(self, dim_s: int, dim_a: int, num_positions: int):
        super().__init__()
        self.num_positions = num_positions
        self.dim_a = dim_a

        # Her pozisyon için ayrı getter ve putter
        self.getters = nn.ModuleList(
            [nn.Linear(dim_s, dim_a) for _ in range(num_positions)]
        )
        self.putter = nn.Linear(num_positions * dim_a, dim_s)

    def get_all(self, s: torch.Tensor) -> torch.Tensor:
        """Tüm pozisyonlardan bileşen çıkar. [B, num_pos, dim_a]"""
        parts = [getter(s) for getter in self.getters]  # num_pos × [B, dim_a]
        return torch.stack(parts, dim=1)  # [B, num_pos, dim_a]

    def put_all(self, s: torch.Tensor, parts: torch.Tensor) -> torch.Tensor:
        """
        parts: [B, num_pos, dim_a]
        Tüm güncellenmiş bileşenleri birleştirip S'ye yaz.
        """
        flat = parts.reshape(parts.shape[0], -1)  # [B, num_pos * dim_a]
        return self.putter(flat)

    def traverse(self, s: torch.Tensor, f) -> torch.Tensor:
        """
        traverse f s: her pozisyona f uygula, sonucu S'ye yaz.
        f: [B, dim_a] → [B, dim_a]
        """
        parts = self.get_all(s)                           # [B, num_pos, dim_a]
        modified = torch.stack(
            [f(parts[:, i, :]) for i in range(self.num_positions)],
            dim=1,
        )  # [B, num_pos, dim_a]
        return self.put_all(s, modified)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Tüm pozisyonları oku ve bir araya getir (dikkat olmadan)."""
        return self.get_all(s)  # [B, num_pos, dim_a]


class VanLaarhovenLens(nn.Module):
    """
    [VAN LAARHOVEN TEMSİLİ]
    Lens'i profunctor kodlamasıyla temsil eder:
    type Lens s t a b = forall f. Functor f => (a -> f b) -> s -> f t

    Bu temsil, lensler üzerindeki kompozisyonu otomatik yapar:
    lens1 ∘ lens2 = lens1 . lens2 (fonksiyon bileşimi olarak)

    ML'de: Katman bileşimlerinin gradyan akışını Van Laarhoven
    temsiliyle açıklamak, neden derin ağların çalıştığını gösterir.
    """

    def __init__(self, focus_net: nn.Module, reconstruct_net: nn.Module):
        super().__init__()
        self.focus = focus_net        # a → f(b)  kısmı
        self.reconstruct = reconstruct_net  # s → f(t) kısmı

    def forward(self, s: torch.Tensor, f=None) -> torch.Tensor:
        """
        lens(f)(s) = reconstruct(s, focus(f(get(s))))
        f=None ise kimlik dönüşümü kullan.
        """
        focused = self.focus(s)
        if f is not None:
            focused = f(focused)
        return self.reconstruct(torch.cat([s, focused], dim=-1))

    def compose(self, other: "VanLaarhovenLens") -> "VanLaarhovenLens":
        """
        İki Van Laarhoven lensini kompoze et.
        (self ∘ other)(f) = self(other(f))
        """
        outer_focus = self.focus
        outer_recon = self.reconstruct
        inner_focus = other.focus
        inner_recon = other.reconstruct

        composed_focus = nn.Sequential(inner_focus, outer_focus)
        composed_recon = nn.Sequential(
            nn.Linear(
                inner_recon.in_features if hasattr(inner_recon, "in_features") else 64,
                outer_recon.out_features if hasattr(outer_recon, "out_features") else 64,
            )
        )
        return VanLaarhovenLens(composed_focus, composed_recon)
