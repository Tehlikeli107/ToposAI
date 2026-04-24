import torch
import torch.nn as nn
import torch.nn.functional as F


class Monad(nn.Module):
    """Minimal monad interface for tensor-valued examples."""

    def T(self, x: torch.Tensor):
        raise NotImplementedError

    def unit(self, x: torch.Tensor):
        raise NotImplementedError

    def join(self, ttx):
        raise NotImplementedError

    def bind(self, tx, f):
        """Kleisli bind: T(X) -> (X -> T(Y)) -> T(Y)."""
        return self.join(self.T(f(tx)))

    def kleisli_compose(self, f, g):
        return lambda x: self.bind(f(x), g)

    def monad_laws_loss(self, x: torch.Tensor) -> torch.Tensor:
        tx = self.unit(x)
        left_unit = F.mse_loss(self.join(self.unit(tx)), tx)
        right_unit = F.mse_loss(self.join(self.T(self.unit(x))), tx)
        return left_unit + right_unit


class GiryMonad(Monad):
    """Finite probability-simplex approximation of the Giry monad."""

    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def T(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=-1)

    def unit(self, x: torch.Tensor) -> torch.Tensor:
        idx = x.argmax(dim=-1)
        return F.one_hot(idx, num_classes=x.shape[-1]).float()

    def join(self, ttx: torch.Tensor) -> torch.Tensor:
        """
        [STRICT MONAD JOIN (Giry Monad Marginalization)]
        Kategori Teorisinde Giry Monadı'nın Join işlemi bir Lebesgue İntegrali
        (beklenen değer)dir. Eski sistemdeki Softmax ve Normalize işlemleri
        Associativity (Birleşme) yasasını ihlal ediyordu.
        Yeni sistem, katı tensör daraltması (Tensor Contraction / Einsum) ile 
        Monad yasasını (μ ∘ μ_T = μ ∘ Tμ) %100 korur.
        """
        if ttx.dim() == 3:
            # ttx shape: [B_dış, B_iç, X_elem]
            
            # İçteki (Inner) dağılımların dış (Outer) olasılıklarla çarpılıp
            # toplanması (Marginalization / Law of Total Probability).
            # Softmax Gürültüsü SİLİNDİ. Doğrudan olasılık çarpımı (Einsum)
            
            # 1. Dış boyutların iç dağılımlar üzerindeki mutlak olasılıkları (Sum to 1)
            outer_weights = ttx.sum(dim=-1) # [B_dış, B_iç]
            outer_weights = outer_weights / (outer_weights.sum(dim=-1, keepdim=True) + 1e-12)
            
            # 2. İç boyutların kendi içindeki bağıl olasılıkları (Sum to 1)
            inner_probs = ttx / (ttx.sum(dim=-1, keepdim=True) + 1e-12) # [B_dış, B_iç, X_elem]
            
            # 3. Toplam Olasılık Yasası (Strict Law of Total Probability)
            # P(X) = Σ_y P(X|Y) * P(Y)
            marginal = torch.einsum('ij,ijk->ik', outer_weights, inner_probs)
            
            return marginal
            
        elif ttx.dim() == 2:
            return ttx
            
        return ttx

    def markov_compose(
        self, K1: torch.Tensor, K2: torch.Tensor
    ) -> torch.Tensor:
        return K1 @ K2


class ContinuationMonad(Monad):
    """Small CPS-inspired wrapper used as a gradient-flow toy model."""

    def __init__(self, transform: nn.Module | None = None):
        super().__init__()
        self.transform = transform or nn.Identity()

    def T(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

    def unit(self, x: torch.Tensor) -> torch.Tensor:
        return self.T(x)

    def join(self, ttx: torch.Tensor) -> torch.Tensor:
        return self.T(ttx)

    def cps_transform(self, f, x: torch.Tensor):
        fx = f(x)
        return lambda k: k(fx)


class WriterMonad(Monad):
    """Writer monad over tensor values and additive tensor logs."""

    def __init__(self, log_dim: int = 1):
        super().__init__()
        self.log_dim = log_dim

    def unit(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log = torch.zeros(*x.shape[:-1], self.log_dim, device=x.device)
        return x, log

    def T(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.unit(x)

    def join(
        self,
        ttx: tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (x, w1), w2 = ttx
        return x, w1 + w2

    def tell(
        self, x: torch.Tensor, log_entry: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, log_entry

    def bind(self, tx, f):
        x, w1 = tx
        y, w2 = f(x)
        return y, w1 + w2


class KleisliLayer(nn.Module):
    """Probabilistic linear layer: x -> (mu, sigma)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.mean_map = nn.Linear(in_features, out_features)
        self.log_std = nn.Parameter(torch.zeros(out_features))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mean_map(x)
        sigma = torch.exp(self.log_std).expand_as(mu)
        return mu, sigma

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        mu, sigma = self.forward(x)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def kl_divergence(self, x: torch.Tensor) -> torch.Tensor:
        mu, sigma = self.forward(x)
        return -0.5 * (1 + 2 * sigma.log() - mu.pow(2) - sigma.pow(2)).sum(-1).mean()
