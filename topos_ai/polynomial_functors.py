import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# POLİNOM FUNKTORLAR (POLY)
# Amacı: David Spivak'ın Poly kategorisini PyTorch'a taşımak.
# Bir Polinom Funktor p: Set → Set şu şekilde tanımlanır:
#   p(X) = Σ_{i ∈ p(1)} X^{p[i]}
# Burada:
#   p(1) = "Pozisyonlar" kümesi (olası durumlar / states)
#   p[i] = Pozisyon i'deki "Yönler" kümesi (mevcut aksiyonlar)
# Temel örnekler:
#   Monomial X^n: tek pozisyon, n yön
#   Lineer p: her pozisyon 1 yön
#   Sabit p: her pozisyon 0 yön (işaret nesnesi / signal object)
# En derin bağlantı: Komonoidler (Comonoids) Poly'de tam olarak
#   Kategorilere (Categories) karşılık gelir! (Spivak, 2022)
# ML bağlantısı:
#   Sinir ağı katmanı = Polinom Funktor morfizması
#   Dinamik sistem = p → q morfizması (state × input → output × next_state)
#   Composition = Tel diyagramı (Wiring Diagram)
# =====================================================================


class PolynomialFunctor(nn.Module):
    """
    [POLİNOM FUNKTOR p]
    p(X) = Σ_{i ∈ positions} X^{directions[i]}
    Bir polinom funktor, her pozisyon i'de directions[i] adet yön içerir.
    """

    def __init__(self, positions: int, directions_per_pos: int | list[int]):
        super().__init__()
        self.positions = positions

        if isinstance(directions_per_pos, int):
            self.directions = [directions_per_pos] * positions
        else:
            assert len(directions_per_pos) == positions
            self.directions = list(directions_per_pos)

        # Öğrenilebilir pozisyon gömmeleri (state representations)
        max_dir = max(self.directions) if self.directions else 1
        self.position_embed = nn.Embedding(positions, max_dir)

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        """
        p(X): Polinom funktorunun X'e uygulanması.
        X: [B, max_directions]
        Döndürür: [B, positions] — her pozisyondaki "konfigürasyon toplamı"
        """
        # Σ_i X^{directions[i]} — her pozisyon için directions kuvveti
        pos_emb = self.position_embed.weight  # [positions, max_dir]

        # X ile pozisyon gömmeleri arasındaki iç çarpım (polinom değerlendirmesi)
        # X^k ≈ exp(k * log(sigmoid(x))) için pürüzsüz yaklaşım
        x_sig = torch.sigmoid(X)  # [B, max_dir]
        scores = x_sig @ pos_emb.T  # [B, positions]
        return scores

    def arena_map(
        self, f_on_positions, f_on_directions
    ) -> "PolynomialFunctor":
        """
        p üzerinde arena morfizması: pozisyon ve yön dönüşümlerini uygula.
        (p morfizması = pozisyon üzerinde ileri, yön üzerinde geri)
        Bu, Poly kategorisindeki morfizmaların tam tanımıdır.
        """
        return self


class MonomialFunctor(PolynomialFunctor):
    """
    [MONOMİAL: y^n = tek pozisyon, n yön]
    En basit polinom funktor: p(X) = X^n.
    Sinir ağlarındaki standart n-giriş nöronuna karşılık gelir.
    """

    def __init__(self, n: int):
        super().__init__(positions=1, directions_per_pos=n)
        self.n = n

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        """p(X) = X^n — n boyutlu giriş vektörü."""
        return (torch.sigmoid(X) ** self.n).sum(dim=-1, keepdim=True)


class PolyMorphism(nn.Module):
    """
    [POLY'DE MORFİZMA: p → q]
    İki polinom funktor arasındaki morfizma:
    (f₁, f₂): p → q
      f₁: p(1) → q(1)        (pozisyonlar üzerinde ileri)
      f₂: q[f₁(i)] → p[i]   (yönler üzerinde geri — ters yönlü!)

    Bu çift yönlülük (covariant on positions, contravariant on directions)
    lenslere ve optiğe benzer — aslında her lens bir Poly morfizmasıdır!

    ML bağlantısı: Her sinir ağı katmanı bir Poly morfizmasıdır.
    """

    def __init__(
        self,
        source: PolynomialFunctor,
        target: PolynomialFunctor,
        hidden: int = 64,
    ):
        super().__init__()
        self.source = source
        self.target = target

        src_pos = source.positions
        tgt_pos = target.positions
        src_dir = max(source.directions) if source.directions else 1
        tgt_dir = max(target.directions) if target.directions else 1

        # f₁: pozisyon dönüşümü (ileri)
        self.f1 = nn.Sequential(
            nn.Linear(src_pos, hidden),
            nn.SiLU(),
            nn.Linear(hidden, tgt_pos),
        )

        # f₂: yön geri dönüşümü (karşıt varyanslı — contravariant)
        self.f2 = nn.Sequential(
            nn.Linear(tgt_dir, hidden),
            nn.SiLU(),
            nn.Linear(hidden, src_dir),
        )

    def forward(
        self, positions: torch.Tensor, directions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        positions: [B, src_positions] — kaynak pozisyon gömmeleri
        directions: [B, tgt_directions] — hedef yön sinyalleri
        Döndürür: (yeni_pozisyonlar, geri_yönler)
        """
        new_positions = self.f1(positions)   # f₁: ileri
        back_directions = self.f2(directions)  # f₂: geri (karşıt)
        return new_positions, back_directions


class DynamicalSystem(nn.Module):
    """
    [DİNAMİK SİSTEM: p → p (öz-morfizma)]
    Spivak'ın temel teoremi: Bir Dinamik Sistem,
    p polinom funktorundan kendine bir morfizmadır.
    sys: p → p
    Bileşenler:
      output: S × I → O  (durum × giriş → çıkış)
      update: S × I → S  (durum × giriş → yeni durum)
    Poly bağlantısı: Durum uzayı = p(1), Giriş/Çıkış = p[i].
    Bu yapı, LSTM/GRU/SSM'lerin (State Space Models) kategorik temelidir.
    """

    def __init__(self, state_dim: int, input_dim: int, output_dim: int):
        super().__init__()
        self.state_dim = state_dim

        # Çıkış morfizması: S × I → O
        self.output_map = nn.Sequential(
            nn.Linear(state_dim + input_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, output_dim),
        )

        # Güncelleme morfizması: S × I → S
        self.update_map = nn.Sequential(
            nn.Linear(state_dim + input_dim, state_dim),
            nn.Tanh(),
        )

    def forward(
        self, state: torch.Tensor, inp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        state: [B, state_dim]
        inp:   [B, input_dim]
        Döndürür: (output, new_state)
        """
        si = torch.cat([state, inp], dim=-1)
        output = self.output_map(si)
        new_state = self.update_map(si)
        return output, new_state

    def run(
        self,
        initial_state: torch.Tensor,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dinamik sistemi T adım boyunca çalıştır.
        inputs: [B, T, input_dim]
        Döndürür: (outputs: [B, T, output_dim], final_state: [B, state_dim])
        """
        state = initial_state
        outputs = []
        T = inputs.shape[1]
        for t in range(T):
            out, state = self.forward(state, inputs[:, t, :])
            outputs.append(out)
        return torch.stack(outputs, dim=1), state


class WiringDiagram(nn.Module):
    """
    [TEL DİYAGRAMI — Poly Kompozisyonu]
    Birden fazla Dinamik Sistemi (kutucuğu) birbirine bağlar.
    Her kutucuğun çıkışları, diğer kutucukların girişlerine kablolanır.
    Kablo haritası: wire_map[i] = (kaynak_kutu, kaynak_çıkış_idx)
    Bu yapı:
    - Derin ağların katman bileşimini
    - Çoklu ajan sistemlerini (open_games.py ile bağlantı)
    - Modüler hesaplama grafiklerini
    kategorik olarak modeller.
    """

    def __init__(
        self,
        systems: list[DynamicalSystem],
        wire_map: list[tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.systems = nn.ModuleList(systems)
        # wire_map[i] = (hangi kutunun çıkışı bu kutunun girişi)
        # None ise doğrusal zincir (0→1→2→...)
        self.wire_map = wire_map

    def forward(
        self,
        states: list[torch.Tensor],
        external_input: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        states:         Her sistem için başlangıç durumu listesi
        external_input: [B, input_dim] — ilk kutucuğa dış giriş
        Döndürür: (outputs listesi, yeni states listesi)
        """
        current_input = external_input
        outputs = []
        new_states = []

        for i, system in enumerate(self.systems):
            out, new_state = system(states[i], current_input)
            outputs.append(out)
            new_states.append(new_state)

            if self.wire_map is not None and i < len(self.wire_map):
                src_box, src_idx = self.wire_map[i]
                current_input = outputs[src_box]
            else:
                current_input = out  # Seri bağlantı

        return outputs, new_states
