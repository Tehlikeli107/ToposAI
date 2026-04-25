import torch


class OpenGame:
    """
    Minimal open-game optic with play and coplay callbacks.

    The class stores the latest observation/action pair so a local coplay
    function can update strategy parameters from a regret signal.
    """

    def __init__(self, name, play_functor, coplay_functor, params=None):
        self.name = name
        self.play_functor = play_functor
        self.coplay_functor = coplay_functor
        self.params = params if params is not None else torch.tensor([0.5], requires_grad=False)
        self.history_X = None
        self.history_Y = None

    def play(self, X):
        """Run the forward strategy map."""
        self.history_X = X.clone()
        Y = self.play_functor(X, self.params)
        self.history_Y = Y.clone()
        return Y

    def coplay(self, R, lr=0.1):
        """Run the backward regret map and update local parameters."""
        S, param_regret = self.coplay_functor(self.history_X, self.history_Y, R, self.params)
        self.params = torch.clamp(self.params + lr * param_regret, min=0.01, max=0.99)
        return S


class ComposedOpenGame:
    """Sequential composition of two open-game blocks."""

    def __init__(self, game_A: OpenGame, game_B: OpenGame):
        self.game_A = game_A
        self.game_B = game_B

    def play(self, X):
        Y_A = self.game_A.play(X)
        Y_B = self.game_B.play(Y_A)
        return Y_B

    def coplay(self, R_B, lr=0.1):
        S_B = self.game_B.coplay(R_B, lr)
        S_A = self.game_A.coplay(S_B, lr)
        return S_A
