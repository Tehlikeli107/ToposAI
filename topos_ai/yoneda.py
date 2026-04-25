import torch
import torch.nn as nn


class YonedaUniverse(nn.Module):
    """
    Yoneda-inspired probe universe.

    This module does not implement the categorical Yoneda lemma directly.
    It stores reference probes and represents an object by its squared
    Euclidean distances to those probes. That makes it a useful
    relation-vector reconstruction toy model, not a proof of Yoneda.
    """

    def __init__(self, num_probes, dim):
        super().__init__()
        self.probes = nn.Parameter(torch.randn(num_probes, dim))

    def get_morphisms(self, X):
        """Return squared distances from X to every probe."""
        return torch.cdist(X, self.probes, p=2) ** 2


class YonedaReconstructor(nn.Module):
    """
    Reconstruct coordinates whose probe-distance vector matches a target.

    The optimization variable is the coordinate vector itself, so this is an
    inverse-distance reconstruction baseline rather than a categorical inverse.
    """

    def __init__(self, num_probes, dim):
        super().__init__()
        self.estimated_X = nn.Parameter(torch.zeros(1, dim))

    def forward(self, true_morphisms, universe: YonedaUniverse):
        estimated_morphisms = universe.get_morphisms(self.estimated_X)
        loss = torch.nn.functional.mse_loss(estimated_morphisms, true_morphisms)
        return loss, self.estimated_X
