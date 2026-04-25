import torch
import torch.nn as nn

from .logic import SubobjectClassifier


class LawvereTierneyTopology(nn.Module):
    """
    Soft Lawvere-Tierney-style operators over the repository's fuzzy logic.

    The methods below expose double-negation and closed-topology analogues for
    tensors in [0, 1]. `check_axioms` reports numerical residuals for the three
    usual topology equations; zero residual is a diagnostic on the chosen fuzzy
    operator, not a theorem about arbitrary neural outputs.
    """

    def __init__(self):
        super().__init__()
        self.omega = SubobjectClassifier()

    def double_negation_topology(self, P):
        """Apply j(p) = not(not(p)) using the local fuzzy negation."""
        return self.omega.logical_not(self.omega.logical_not(P))

    def closed_topology(self, P, C):
        """Apply j_c(p) = p OR c with the local fuzzy disjunction."""
        return self.omega.logical_or(P, C)

    def check_axioms(self, P, Q, C, topology_func):
        """
        Return max absolute residuals for the three topology equations:

        1. j(True) == True
        2. j(j(p)) == j(p)
        3. j(p AND q) == j(p) AND j(q)
        """
        T = torch.ones_like(P)

        if topology_func.__name__ == "closed_topology":

            def j(x):
                return topology_func(x, C)

        else:
            j = topology_func

        ax1_diff = torch.abs(j(T) - T).max().item()
        ax2_diff = torch.abs(j(j(P)) - j(P)).max().item()

        P_and_Q = self.omega.logical_and(P, Q)
        left_side = j(P_and_Q)
        right_side = self.omega.logical_and(j(P), j(Q))
        ax3_diff = torch.abs(left_side - right_side).max().item()

        return ax1_diff, ax2_diff, ax3_diff
