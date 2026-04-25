import math

import torch
import torch.nn as nn


class OMinimalProjector(nn.Module):
    """
    Bounded polynomial activation inspired by tame geometry.

    This is not a proof that a network is o-minimal, nor does it guarantee a
    finite number of loss basins. It simply maps inputs through tanh and a
    finite odd polynomial, which is useful as a tame regularization toy model.
    """

    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        x_compact = torch.tanh(x)
        tame_signal = torch.zeros_like(x_compact)

        for i in range(1, self.degree + 1, 2):
            coef = ((-1.0) ** ((i - 1) // 2)) / float(math.factorial(i))
            tame_signal += coef * (x_compact**i)

        return tame_signal


class TameNeuralLayer(nn.Module):
    """Linear layer followed by the bounded polynomial projector."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.tame_projector = OMinimalProjector(degree=3)

    def forward(self, x):
        return self.tame_projector(self.linear(x))
