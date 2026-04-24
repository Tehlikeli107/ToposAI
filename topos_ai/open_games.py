import torch

# =====================================================================
# CATEGORICAL CYBERNETICS (OPEN GAMES)
# Amacı: Çoklu ajanların (Multi-Agent) bulunduğu ortamlarda,
# her bir ajanın stratejisini (Play) ve pişmanlığını (Coplay/Regret)
# Kategori Teorisindeki 'Lens (Optic)' yapılarıyla modellemek.
# Açık Oyunlar (Open Games), karmaşık oyunları devasa ağaçlarla
# (Minimax) çözmek yerine, küçük oyunları Legolar gibi birbirine
# bağlayarak (Composition) Merkeziyetsiz Nash Dengesi (Decentralized
# Nash Equilibrium) bulmayı sağlar.
# =====================================================================

class OpenGame:
    """
    Bir Açık Oyun (Open Game) 4 portu olan bir Kategori Kutusudur (Morphism):
    - X: Oyunun gözlemlediği durum (State in)
    - Y: Oyuncunun yaptığı hamle (Play out)
    - R: Ortamdan gelen ödül/pişmanlık (Utility/Regret in)
    - S: Önceki oyuncuya aktarılan pişmanlık (Coplay out)
    """
    def __init__(self, name, play_functor, coplay_functor, params=None):
        self.name = name
        # play_functor(X, params) -> Y
        self.play_functor = play_functor
        # coplay_functor(X, Y, R, params) -> S, ve params_gradient (Öğrenme için Regret)
        self.coplay_functor = coplay_functor

        # Ajanın stratejisini (Ağırlıklarını) belirleyen Kategori Değişkenleri
        self.params = params if params is not None else torch.tensor([0.5], requires_grad=False)
        self.history_X = None
        self.history_Y = None

    def play(self, X):
        """İleri Yön (Strateji Üretimi)"""
        self.history_X = X.clone()
        Y = self.play_functor(X, self.params)
        self.history_Y = Y.clone()
        return Y

    def coplay(self, R, lr=0.1):
        """
        Geri Yön (Pişmanlık ve Öğrenme).
        Burada PyTorch Autograd kullanılmaz! Sadece lokal "Regret (Pişmanlık)"
        sinyalleri Kategori kurallarına göre işlenir.
        """
        S, param_regret = self.coplay_functor(self.history_X, self.history_Y, R, self.params)

        # Ajan stratejisini (parametresini) pişmanlığına göre günceller
        self.params = self.params + lr * param_regret

        # Sınırlandırma (Topolojik Manifold Kuralları, örn: Olasılık [0, 1])
        self.params = torch.clamp(self.params, min=0.01, max=0.99)

        return S

class ComposedOpenGame:
    """
    İki Açık Oyunu (Game A ve Game B) seri (Sequential) olarak bağlayan yapı.
    (String Diagrams in Category Theory)
    """
    def __init__(self, game_A: OpenGame, game_B: OpenGame):
        self.game_A = game_A
        self.game_B = game_B

    def play(self, X):
        # A'nın hamlesi (Y_A), B'nin gözlemi (X_B) olur.
        Y_A = self.game_A.play(X)
        Y_B = self.game_B.play(Y_A)
        return Y_B

    def coplay(self, R_B, lr=0.1):
        # B'ye gelen ödül/pişmanlık (R_B), B tarafından işlenir.
        # B'nin ürettiği Coplay (S_B), A'nın ödülü/pişmanlığı (R_A) olur.
        S_B = self.game_B.coplay(R_B, lr)
        S_A = self.game_A.coplay(S_B, lr)
        return S_A
