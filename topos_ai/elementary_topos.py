import torch
import torch.nn as nn

from .logic import godel_implication


class ElementaryTopos(nn.Module):
    """
    Tensor-level proxy for a few elementary-topos operations.

    The implementation models products, coproducts, exponentials, and a
    subobject-classifier-style inclusion score with fuzzy logic on tensors in
    [0, 1]. It is useful as a differentiable teaching/experimentation layer,
    not as a certification that an arbitrary tensor space is an elementary
    topos.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.initial_object = torch.zeros(dim)
        self.terminal_object = torch.ones(dim)

    def product(self, X, Y):
        """Fuzzy product/meet proxy: min(X, Y)."""
        return torch.minimum(X, Y)

    def coproduct(self, X, Y):
        """Fuzzy coproduct/join proxy: max(X, Y)."""
        return torch.maximum(X, Y)

    def exponential(self, Y, Z):
        """
        Goedel-Heyting internal hom on the unit interval.

        This is the residual of meet/min: Y => Z is 1 when Y <= Z and Z
        otherwise.
        """
        return godel_implication(Y, Z)

    def subobject_classifier(self, X, Y):
        """Return the fuzzy inclusion score X => Y."""
        return self.exponential(X, Y)

    def check_morphism(self, A, B):
        """
        Return whether A is pointwise below B in this tensor proxy.

        This is a concrete order check on tensors, not a categorical proof
        search over an arbitrary category.
        """
        return torch.all(A <= B).item()
