import torch
import torch.nn as nn


class StrictGodelImplication(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, hardness=50.0):
        ctx.save_for_backward(A, B)
        ctx.hardness = hardness
        
        # [KESİN/STRICT İLERİ YÖN - TOPOS TEORİSİ (A -> B)]
        return torch.where(A <= B, torch.tensor(1.0, dtype=A.dtype, device=A.device), B)

    @staticmethod
    def backward(ctx, grad_output):
        # [YUMUŞAK/SOFT GERİ YÖN] 
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
    Fuzzy truth-value operations inspired by Heyting algebras.

    The methods operate on tensors and return differentiable approximations of
    meet, join, implication, and negation. They are numerical operators, not a
    formal proof engine for internal topos logic.
    """

    def __init__(self):
        super().__init__()
        self.truth_morphism = 1.0
        self.false_morphism = 0.0

    def logical_and(self, A, B):
        """Meet proxy: min(A, B)."""
        return torch.minimum(A, B)

    def logical_or(self, A, B):
        """Join proxy: max(A, B)."""
        return torch.maximum(A, B)

    def implies(self, A, B, hardness=50.0):
        """
        Strict Goedel-style implication with Hybrid Autograd.

        The hard operator is 1 when A <= B and B otherwise (Zero Modus Ponens Violation).
        The soft formulation is used purely for gradients.
        """
        return StrictGodelImplication.apply(A, B, hardness)

    def logical_not(self, A, hardness=50.0):
        """
        [STRICT TOPOLOGICAL NEGATION]
        Eski bulanık 'sigmoid(0.5 - A)' yaklaşımı silinmiştir.
        Lawvere-Tierney topolojisindeki İdempotent aksiyomunu (j(j(p)) == j(p))
        sağlayabilmek için katı bir tamamlayıcı (1.0 - A) gereklidir.
        """
        return 1.0 - A


class HeytingNeuralLayer(nn.Module):
    """
    Linear-like layer built from the fuzzy implication proxy.
    """

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
