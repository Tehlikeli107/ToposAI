import torch
import torch.nn as nn
import torch.nn.functional as F


class LeftKanExtension(nn.Module):
    """
    Soft left-Kan-extension proxy.

    Given source objects c, learned embeddings K(c), and learned values F(c),
    this module evaluates targets with a cross-attention style weighted sum:

        sum_c softmax(sim(K(c), e)) * F(c)

    This is a differentiable analogy to the categorical construction, not a
    proof that an arbitrary learned model satisfies a universal property.
    """

    def __init__(self, dim_c: int, dim_e: int, dim_d: int):
        super().__init__()
        self.K = nn.Linear(dim_c, dim_e, bias=False)
        self.F = nn.Linear(dim_c, dim_d, bias=False)
        self.scale = dim_e**-0.5

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: [num_source, dim_c]
            target: [batch, dim_e]

        Returns:
            [batch, dim_d]
        """
        Kc = self.K(source)  # [num_source, dim_e]
        Fc = self.F(source)  # [num_source, dim_d]
        
        # [STRICT COLIMIT (SUPREMUM) FOR LEFT KAN]
        # Softmax yerine Kategori teorisindeki Universal Colimit (En Büyük Alt Sınır) kullanılır.
        sim = target @ Kc.T * self.scale  # [batch, num_source]
        
        # Olasılık/ağırlıklarımızı 0 ile 1 arasına Sigmoid ile sıkıştırıyoruz (Colimit ağırlığı)
        weights = torch.sigmoid(sim)  # [batch, num_source]
        
        # Tensör çarpımı (Softmax gibi ortalama DEĞİL, her bir elemanı ağırlıklandır)
        weighted_Fc = weights.unsqueeze(-1) * Fc.unsqueeze(0)  # [batch, num_source, dim_d]
        
        # Colimit: Tüm kaynaklardan gelen (weighted_Fc) bileşenlerin SUPREMUM'unu (MAX) al.
        # Böylece evrensellik aksiyomu korunur.
        supremum, _ = torch.max(weighted_Fc, dim=1)  # [batch, dim_d]
        return supremum


class RightKanExtension(nn.Module):
    """
    Soft right-Kan-extension proxy.

    Targets are compared to learned K(c) points with Euclidean distance and
    values are averaged with soft nearest-neighbor weights. This is useful for
    interpolation experiments, but it is not a categorical end construction.
    """

    def __init__(self, dim_c: int, dim_e: int, dim_d: int):
        super().__init__()
        self.K = nn.Linear(dim_c, dim_e, bias=False)
        self.F = nn.Linear(dim_c, dim_d, bias=False)
        self.scale = dim_e**-0.5

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: [num_source, dim_c]
            target: [batch, dim_e]

        Returns:
            [batch, dim_d]
        """
        Kc = self.K(source)
        Fc = self.F(source)
        dist = torch.cdist(target, Kc)
        weights = torch.softmax(-dist * self.scale, dim=-1)
        return weights @ Fc


class NeuralKanExtension(nn.Module):
    """
    Learned source-object wrapper around the soft Kan proxies.
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
        self.source_objects = nn.Parameter(torch.randn(num_source, dim_c))

        if mode == "left":
            self.kan = LeftKanExtension(dim_c, dim_e, dim_d)
        elif mode == "right":
            self.kan = RightKanExtension(dim_c, dim_e, dim_d)
        else:
            raise ValueError(f"mode '{mode}' is unknown. Use 'left' or 'right'.")

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        return self.kan(self.source_objects, target)


class KanAdjunction(nn.Module):
    """
    Bundle left proxy, right proxy, and a learned restriction map.

    The object mirrors the shape of the classical Lan_K -| Res_K -| Ran_K
    story, but the learned maps below do not certify a universal property by
    themselves. Use losses and benchmarks as empirical diagnostics only.
    """

    def __init__(self, dim_c: int, dim_e: int, dim_d: int):
        super().__init__()
        self.left_kan = LeftKanExtension(dim_c, dim_e, dim_d)
        self.right_kan = RightKanExtension(dim_c, dim_e, dim_d)
        self.restriction = nn.Linear(dim_e, dim_c, bias=False)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        Empirical fit loss for the two soft Kan proxies.

        A small value means both learned proxy maps match the supplied targets
        on this batch. It does not prove an actual categorical universal
        property.
        """
        left, right, _ = self.forward(source, target)
        return F.mse_loss(left, true_values) + F.mse_loss(right, true_values)
