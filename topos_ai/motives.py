import torch
import torch.nn as nn


class MotifFunctor(nn.Module):
    """Map one data domain into a shared learned latent motive space."""

    def __init__(self, in_dim, motive_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, motive_dim),
        )

    def forward(self, x):
        return self.net(x)


class UniversalMotiveEngine(nn.Module):
    """
    Two-domain latent-alignment toy model inspired by motives/Langlands.

    The module learns maps A -> M and B -> M and compares the resulting
    distributions with an RBF maximum mean discrepancy loss. A low MMD value
    indicates empirical latent alignment for the sampled batches; it is not a
    proof that two mathematical domains share a Grothendieck motive.
    """

    def __init__(self, dim_A, dim_B, motive_dim=16):
        super().__init__()
        self.functor_A = MotifFunctor(dim_A, motive_dim)
        self.functor_B = MotifFunctor(dim_B, motive_dim)

    def topological_mmd_loss(self, X_A, X_B):
        """
        Compute an RBF-kernel MMD loss between learned latent projections.
        """
        M_A = self.functor_A(X_A)
        M_B = self.functor_B(X_B)

        xx = torch.cdist(M_A, M_A, p=2) ** 2
        yy = torch.cdist(M_B, M_B, p=2) ** 2
        xy = torch.cdist(M_A, M_B, p=2) ** 2

        sigma = 1.0
        kernel_xx = torch.exp(-xx / (2 * sigma**2)).mean()
        kernel_yy = torch.exp(-yy / (2 * sigma**2)).mean()
        kernel_xy = torch.exp(-xy / (2 * sigma**2)).mean()

        mmd_loss = kernel_xx + kernel_yy - 2 * kernel_xy
        return mmd_loss, M_A, M_B
