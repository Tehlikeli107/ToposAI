import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# MONADS (T, η, μ) — KLEİSLİ KATEGORİSİ
# Amacı: Kategori teorisinin en güçlü yapılarından olan Monadları
# PyTorch'a taşımak.
# Bir Monad (T, η, μ): C kategorisi üzerindeki bir endofunktor T,
#   - Unit (Return): η: Id_C → T  (saf değeri sarmala)
#   - Join (Flatten): μ: T∘T → T  (çift sarmalamayı düzleştir)
# Monad Yasaları:
#   - Sol birim: μ ∘ T(η) = Id_T
#   - Sağ birim: μ ∘ η_T  = Id_T
#   - İlişkilik: μ ∘ T(μ)  = μ ∘ μ_T
# Kleisli Kategorisi: Morfizmalar A → T(B) şeklinde, kompozisyon
# 'bind (>>=)' operatörüyle yapılır.
# ML bağlantısı: Geri yayılım (Backprop) = Continuation Monadı!
#   Gradyanlar, continuation passthrough (CPS) ile akar.
#   Olasılıksal katmanlar = Giry (Olasılık) Monadı.
# =====================================================================


class Monad(nn.Module):
    """Soyut Monad arayüzü. Alt sınıflar T, unit ve bind'i tanımlar."""

    def T(self, x: torch.Tensor) -> torch.Tensor:
        """Endofunktor T: X → T(X)"""
        raise NotImplementedError

    def unit(self, x: torch.Tensor) -> torch.Tensor:
        """η: x → T(x)  (return / pure)"""
        raise NotImplementedError

    def join(self, ttx: torch.Tensor) -> torch.Tensor:
        """μ: T(T(x)) → T(x)  (join / flatten)"""
        raise NotImplementedError

    def bind(self, tx: torch.Tensor, f) -> torch.Tensor:
        """
        (>>=): T(X) → (X → T(Y)) → T(Y)
        Kleisli kompozisyonunun temel operatörü.
        bind(tx, f) = join(T(f)(tx))
        """
        return self.join(self.T(f(tx)))

    def kleisli_compose(self, f, g):
        """
        (>=>): (A → T(B)) → (B → T(C)) → (A → T(C))
        İki Kleisli morfizmasını birleştirir.
        """
        return lambda x: self.bind(f(x), g)

    def monad_laws_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Monad yasalarını kayıp fonksiyonu olarak ölçer.
        Sıfıra yakınsa, yapı gerçek bir Monad'dır.
        """
        tx = self.unit(x)
        # Sol birim: join(unit(T(x))) ≈ T(x)
        left_unit = F.mse_loss(self.join(self.unit(tx)), tx)
        # Sağ birim: join(T(unit(x))) ≈ T(x)
        right_unit = F.mse_loss(self.join(self.T(self.unit(x))), tx)
        return left_unit + right_unit


class GiryMonad(Monad):
    """
    [GİRY MONADI — OLASILİK MONADİ]
    T(X) = X üzerindeki olasılık dağılımları uzayı (Olasılık Simpleksi).
    η(x) = Dirac delta (x üzerinde yoğunlaşmış dağılım)
    μ   = Marjinalleştirme (Beklenti / Expectation)

    ML bağlantısı: Softmax çıkışları Giry Monad'ının nesneleridir.
    Kleisli morfizması A → D(B), stokastik bir dönüşümdür (Markov çekirdeği).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def T(self, x: torch.Tensor) -> torch.Tensor:
        """Dağılımı yeniden normalize et (Giry funktorunun etki alanı)."""
        return torch.softmax(x, dim=-1)

    def unit(self, x: torch.Tensor) -> torch.Tensor:
        """η: sürekli vektör → en yakın Dirac delta (one-hot yaklaşımı)."""
        idx = x.argmax(dim=-1)
        return F.one_hot(idx, num_classes=x.shape[-1]).float()

    def join(self, ttx: torch.Tensor) -> torch.Tensor:
        """
        μ: D(D(X)) → D(X)
        Dağılım üzerindeki dağılımı marjinalleştir.
        ttx: [B, dim, dim] — satırlar dış dağılım, sütunlar iç dağılım.
        """
        if ttx.dim() == 3:
            # Σ_y p(y) * q(x | y)
            return torch.einsum("bij,bi->bj", ttx, ttx.sum(dim=-1, keepdim=True).expand_as(ttx))
        return ttx

    def markov_compose(
        self, K1: torch.Tensor, K2: torch.Tensor
    ) -> torch.Tensor:
        """
        İki Markov çekirdeğini (stokastik matris) Kleisli olarak birleştirir.
        K1: [A, B], K2: [B, C]  →  K1 >=> K2: [A, C]
        """
        return K1 @ K2


class ContinuationMonad(Monad):
    """
    [CONTİNUATİON MONADI — BACKPROP'UN MATEMATİKSEL TEMELİ]
    T(X) = (X → R) → R
    Geri yayılım (Backprop), continuation passing style (CPS) dönüşümüdür:
    Her ileri geçiş fonksiyonu f: X → Y, Kleisli morfizmasına
    f_CPS: X → (Y → R) → R dönüştürülür.
    Bu monad, gradyan akışını kategorik olarak açıklar.

    Pratik uygulama: Gradyan transformatörü olarak çalışır.
    """

    def __init__(self, transform: nn.Module | None = None):
        super().__init__()
        # Continuation dönüşümü (isteğe bağlı, varsayılan: kimlik)
        self.transform = transform or nn.Identity()

    def T(self, x: torch.Tensor) -> torch.Tensor:
        """T(x) = continuation sarmalayıcı (burada: transform(x))."""
        return self.transform(x)

    def unit(self, x: torch.Tensor) -> torch.Tensor:
        """η(x)(k) = k(x) — saf değeri continuation'a koy."""
        return self.T(x)  # k = T olarak düşün

    def join(self, ttx: torch.Tensor) -> torch.Tensor:
        """μ: T(T(x)) → T(x) — çift sarmalamayı düzleştir."""
        return self.T(ttx)

    def cps_transform(self, f, x: torch.Tensor):
        """
        Bir fonksiyonu f CPS (Continuation Passing Style) biçimine dönüştürür.
        f_cps(x)(k) = k(f(x))
        Gradyan akışı bu yapıyla kategorik olarak modellenir.
        """
        fx = f(x)
        return lambda k: k(fx)


class WriterMonad(Monad):
    """
    [WRITER MONADI — BİRİKİMLİ LOG/METRİK]
    T(X) = X × W  (W bir monoid, örneğin: log tensörü, kayıp toplamı)
    η(x) = (x, ε)  — ε: monoidin birim elemanı (sıfır tensör)
    μ((x, w1), w2) = (x, w1 + w2)  — logları birleştir

    ML bağlantısı: Eğitim sırasında ara kayıpları veya aktivasyon
    istatistiklerini otomatik olarak biriktirmek için kullanılır.
    """

    def __init__(self, log_dim: int = 1):
        super().__init__()
        self.log_dim = log_dim

    def unit(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """η(x) = (x, 0) — boş log ile sar."""
        log = torch.zeros(*x.shape[:-1], self.log_dim, device=x.device)
        return x, log

    def T(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.unit(x)

    def join(
        self,
        ttx: tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """μ: ((x, w1), w2) → (x, w1 + w2)"""
        (x, w1), w2 = ttx
        return x, w1 + w2

    def tell(
        self, x: torch.Tensor, log_entry: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mevcut değere bir log kaydı ekle."""
        return x, log_entry

    def bind(self, tx, f):
        """Writer bind: logları toplar."""
        x, w1 = tx
        y, w2 = f(x)
        return y, w1 + w2


class KleisliLayer(nn.Module):
    """
    [KLEİSLİ KATEGORİSİ'NDE MORFİZMA: X → T(Y)]
    Olasılıksal Doğrusal Katman.
    Klasik nn.Linear(x) yerine (μ_Y, σ_Y) olasılık dağılımı döndürür.
    Bu, Kleisli kategorisinde bir morfizmadır:
        f: X → D(Y)  (X'ten Y üzerindeki dağılıma)
    Kleisli kompozisyonu ile katmanlar zinciri,
    otomatik olarak doğru olasılıksal davranış sergiler.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.mean_map = nn.Linear(in_features, out_features)
        self.log_std = nn.Parameter(torch.zeros(out_features))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Kleisli morfizması: x → (μ, σ) dağılımı döndür."""
        mu = self.mean_map(x)
        sigma = torch.exp(self.log_std).expand_as(mu)
        return mu, sigma

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Dağılımdan yeniden parametreli örnekleme (reparameterization trick)."""
        mu, sigma = self.forward(x)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def kl_divergence(self, x: torch.Tensor) -> torch.Tensor:
        """KL(q(y|x) || N(0,1)) — bilgi-teorik düzenlileştirici."""
        mu, sigma = self.forward(x)
        return -0.5 * (1 + 2 * sigma.log() - mu.pow(2) - sigma.pow(2)).sum(-1).mean()
