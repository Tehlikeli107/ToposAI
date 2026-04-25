import torch
import torch.nn as nn


def godel_implication(A, B):
    """Exact Goedel-Heyting implication on the unit interval."""
    return torch.where(A <= B, torch.ones_like(B), B)


class StrictGodelImplication(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, hardness=50.0):
        ctx.save_for_backward(A, B)
        ctx.hardness = hardness
        return godel_implication(A, B)

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        hardness = ctx.hardness

        with torch.enable_grad():
            A_soft = A.detach().requires_grad_(True)
            B_soft = B.detach().requires_grad_(True)

            sigma = torch.sigmoid((B_soft - A_soft) * hardness)
            soft_impl = sigma + (1.0 - sigma) * B_soft

            soft_impl.backward(grad_output)

        return A_soft.grad, B_soft.grad, None


class SubobjectClassifier(nn.Module):
    """
    Tensor proxy for truth values in a Goedel-Heyting algebra.

    Meet is min, join is max, implication is the Heyting residual for min, and
    negation is the pseudocomplement A => false. These are numerical operators
    for experiments, not a proof assistant for arbitrary topoi.
    """

    def __init__(self):
        super().__init__()
        self.truth_morphism = 1.0
        self.false_morphism = 0.0

    def logical_and(self, A, B):
        """Meet: min(A, B)."""
        return torch.minimum(A, B)

    def logical_or(self, A, B):
        """Join: max(A, B)."""
        return torch.maximum(A, B)

    def implies(self, A, B, hardness=50.0):
        """
        Goedel-Heyting implication with exact forward values.

        The custom backward uses a smooth boundary approximation so neural
        experiments are trainable around A ~= B.
        """
        return StrictGodelImplication.apply(A, B, hardness)

    def logical_not(self, A, hardness=50.0):
        """Heyting pseudocomplement: not A is A => false."""
        return self.implies(A, torch.zeros_like(A), hardness)


class HeytingNeuralLayer(nn.Module):
    """Linear-like layer built from Goedel-Heyting implication."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(out_features, in_features))
        self.omega = SubobjectClassifier()

    def forward(self, x):
        x_logical = torch.sigmoid(x)
        w_logical = torch.sigmoid(self.weight)

        x_exp = x_logical.unsqueeze(1)
        w_exp = w_logical.unsqueeze(0)
        implications = self.omega.implies(x_exp, w_exp)
        return implications.min(dim=-1).values
