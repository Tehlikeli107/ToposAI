import torch
import torch.nn as nn
import torch.nn.functional as F


class AdjointPair(nn.Module):
    """A pair of maps with unit/counit round-trip diagnostics."""

    def __init__(self, F_map: nn.Module, G_map: nn.Module):
        super().__init__()
        self.F = F_map
        self.G = G_map

    def unit(self, x: torch.Tensor) -> torch.Tensor:
        """eta_x: x -> G(F(x))."""
        return self.G(self.F(x))

    def counit(self, y: torch.Tensor) -> torch.Tensor:
        """epsilon_y: F(G(y)) -> y."""
        return self.F(self.G(y))

    def triangle_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Round-trip losses for the two triangle identities."""
        loss_unit = F.mse_loss(self.unit(x), x)
        loss_counit = F.mse_loss(self.counit(y), y)
        return loss_unit, loss_counit

    def hom_isomorphism(
        self, f_morphism: torch.Tensor, g_morphism: torch.Tensor
    ) -> torch.Tensor:
        """Measure agreement between two Hom-set representations."""
        return torch.norm(f_morphism - g_morphism)


class _LinearMap(nn.Module):
    def __init__(self, W: nn.Parameter, transpose: bool):
        super().__init__()
        self.W = W
        self.transpose = transpose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            return x @ self.W
        return x @ self.W.T


class LinearAdjoint(AdjointPair):
    """Finite-dimensional Hilbert-space adjoint: W paired with W.T."""

    def __init__(self, dim_in: int, dim_out: int):
        W = nn.Parameter(torch.randn(dim_out, dim_in) * 0.1)
        super().__init__(
            _LinearMap(W, transpose=False),
            _LinearMap(W, transpose=True),
        )

    @property
    def W(self) -> nn.Parameter:
        """Shared parameter used by both linear directions."""
        return self.F.W


class NeuralAdjoint(AdjointPair):
    """Encoder/decoder pair with adjunction-style round-trip losses."""

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
    """Embedding/free map paired with a linear forgetful/logit map."""

    def __init__(self, vocab_size: int, embed_dim: int):
        F_net = nn.Embedding(vocab_size, embed_dim)
        G_net = nn.Linear(embed_dim, vocab_size, bias=False)
        super().__init__(F_net, G_net)
        self.vocab_size = vocab_size

    def unit(self, idx: torch.Tensor) -> torch.Tensor:
        """idx -> logits -> probability distribution over the vocabulary."""
        emb = self.F(idx)
        logits = self.G(emb)
        return torch.softmax(logits, dim=-1)

    def counit(self, emb: torch.Tensor) -> torch.Tensor:
        """emb -> most likely token -> token embedding."""
        logits = self.G(emb)
        idx = logits.argmax(dim=-1)
        return self.F(idx)

    def triangle_loss(
        self, idx: torch.Tensor, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compare each round trip in the space where it actually lives."""
        probs = self.unit(idx)
        target = F.one_hot(idx.long(), num_classes=self.vocab_size).to(
            device=probs.device,
            dtype=probs.dtype,
        )
        loss_unit = F.mse_loss(probs, target)
        loss_counit = F.mse_loss(self.counit(emb), emb)
        return loss_unit, loss_counit
