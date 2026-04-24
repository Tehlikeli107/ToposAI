import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# ADJOINT FUNCTORS (F ⊣ G)
# Amacı: Kategori teorisinin en temel yapılarından biri olan Adjoint
# (Eşlenik) Funktor çiftlerini PyTorch'a taşımak.
# F ⊣ G anlamı: F: C → D sol eşlenik, G: D → C sağ eşlenik.
# Her adjunction bir Unit (η: Id_C → G∘F) ve Counit (ε: F∘G → Id_D)
# ile tanımlanır. Üçgen özdeşlikler (Triangle Identities) ise modelin
# kararlılığını (Stability) garanti eder.
# Sinir ağlarında: F = Encoder (gömücü), G = Decoder (yeniden kurucu).
# Adjunction unit: veriyi kodlayıp geri çözdüğünde kendine yakın olmalı.
# Adjunction counit: hedef uzaydaki noktayı geri çevirip yeniden
# hedef uzaya gönderince kendine yakın olmalı.
# =====================================================================


class AdjointPair(nn.Module):
    """
    F ⊣ G: Soyut Adjoint Funktor Çifti.
    Alt sınıflar F ve G'yi tanımlamalıdır.
    """

    def __init__(self, F: nn.Module, G: nn.Module):
        super().__init__()
        self.F = F  # Sol eşlenik: C → D
        self.G = G  # Sağ eşlenik: D → C

    def unit(self, x: torch.Tensor) -> torch.Tensor:
        """η_x: x → G(F(x))  — kaynak uzayında 'serbest inşa et ve unut'."""
        return self.G(self.F(x))

    def counit(self, y: torch.Tensor) -> torch.Tensor:
        """ε_y: F(G(y)) → y  — hedef uzayında 'unut ve serbest inşa et'."""
        return self.F(self.G(y))

    def triangle_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        [TRIANGLE IDENTITIES]
        G(ε_{F(x)}) ∘ η_{G(F(x))} = Id_C  →  unit(x) ≈ x
        F(η_{G(y)}) ∘ ε_{F(G(y))} = Id_D  →  counit(y) ≈ y
        Bu iki kayıp, adjunction'ın matematiksel geçerliliğini zorlar.
        """
        loss_unit = F.mse_loss(self.unit(x), x)
        loss_counit = F.mse_loss(self.counit(y), y)
        return loss_unit, loss_counit

    def hom_isomorphism(
        self, f_morphism: torch.Tensor, g_morphism: torch.Tensor
    ) -> torch.Tensor:
        """
        Temel Adjunction Özdeşliği: Hom_D(F(c), d) ≅ Hom_C(c, G(d))
        İki morfizma vektörünü hizalayarak bu izomorfizmi ölçer.
        Sonuç sıfıra yakınsa iki taraf gerçekten adjoint demektir.
        """
        return torch.norm(f_morphism - g_morphism)


class LinearAdjoint(AdjointPair):
    """
    Hilbert Uzayı Adjointi: F = W (linear), G = W^T (transpozisi).
    Sonlu boyutlu vektör uzaylarında F ⊣ G = W ⊣ Wᵀ.
    Sinir ağlarındaki ağırlık geri-aktarımının (Weight Tying) matematiksel
    temeli budur.
    """

    def __init__(self, dim_in: int, dim_out: int):
        self.W = nn.Parameter(torch.randn(dim_out, dim_in) * 0.1)
        F_module = _LinearMap(self.W, transpose=False)
        G_module = _LinearMap(self.W, transpose=True)
        super().__init__(F_module, G_module)
        # W'yi her iki modülün de görebilmesi için tekrar kaydet
        self.register_parameter("W", self.W)


class _LinearMap(nn.Module):
    def __init__(self, W: nn.Parameter, transpose: bool):
        super().__init__()
        self.W = W
        self.transpose = transpose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            return x @ self.W  # [B, dim_out] → [B, dim_in]
        return x @ self.W.T   # [B, dim_in]  → [B, dim_out]


class NeuralAdjoint(AdjointPair):
    """
    Genel Nöral Adjoint: F ve G keyfi MLP'ler.
    Encoder-Decoder mimarilerinin kategorik genellemesi.
    Triangle loss eğitim sırasında adjunction yapısını korur.
    """

    def __init__(
        self,
        dim_c: int,
        dim_d: int,
        hidden: int = 128,
    ):
        F_net = nn.Sequential(
            nn.Linear(dim_c, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim_d),
        )
        G_net = nn.Sequential(
            nn.Linear(dim_d, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim_c),
        )
        super().__init__(F_net, G_net)


class FreeForgetfulAdjoint(AdjointPair):
    """
    Serbest-Unutkan (Free ⊣ Forgetful) Adjunction.
    Algebraik yapılardaki en klasik adjoint örüntüsü:
    - F (Serbest Funktor): Ham veriyi yapısal temsile kaldırır.
    - G (Unutkan Funktor): Yapıyı silip ham vektöre döner.
    Örnek: Kelime → Gömme (F), Gömme → Kelime (G) çiftleri.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        F_net = nn.Embedding(vocab_size, embed_dim)  # Free: index → dense
        G_net = nn.Linear(embed_dim, vocab_size, bias=False)  # Forgetful: dense → logits
        super().__init__(F_net, G_net)

    def unit(self, idx: torch.Tensor) -> torch.Tensor:
        """η: kelime indeksi → G(F(idx)) = logitler → softmax → geri."""
        emb = self.F(idx)               # [B, embed_dim]
        logits = self.G(emb)            # [B, vocab_size]
        return torch.softmax(logits, dim=-1)

    def counit(self, emb: torch.Tensor) -> torch.Tensor:
        """ε: gömme → F(G(emb)) = en olası kelime gömmesi."""
        logits = self.G(emb)            # [B, vocab_size]
        idx = logits.argmax(dim=-1)     # [B]
        return self.F(idx)              # [B, embed_dim]
