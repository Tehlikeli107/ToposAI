import torch
import torch.nn as nn
import torch.nn.functional as F


class Lens(nn.Module):
    """
    Structural lens over a tensor slice.

    `get` reads a contiguous component of the state and `put` writes a new
    component back into the same slice. For this structural lens, the standard
    get-put and put-get lens laws hold exactly up to tensor equality.
    """

    def __init__(self, dim_s: int, start_idx: int, dim_a: int):
        super().__init__()
        self.dim_s = dim_s
        self.start = start_idx
        self.end = start_idx + dim_a
        assert self.end <= dim_s, "Lens boundaries exceed state dimension."

    def get(self, s: torch.Tensor) -> torch.Tensor:
        return s[..., self.start:self.end]

    def put(self, s: torch.Tensor, a_new: torch.Tensor) -> torch.Tensor:
        s_new = s.clone()
        s_new[..., self.start:self.end] = a_new
        return s_new

    def modify(self, s: torch.Tensor, f) -> torch.Tensor:
        return self.put(s, f(self.get(s)))

    def lens_laws_loss(self, s: torch.Tensor) -> torch.Tensor:
        """Return a diagnostic loss for the get-put and put-get laws."""
        a = self.get(s)
        get_put_loss = F.mse_loss(self.put(s, a), s)
        put_get_loss = F.mse_loss(self.get(self.put(s, a)), a)
        return get_put_loss + put_get_loss

    @classmethod
    def linear(cls, dim_s: int, dim_a: int) -> "Lens":
        """
        Backward-compatible constructor for the first `dim_a` coordinates.

        Despite the legacy name, this returns a structural slice lens rather
        than a learned linear projection.
        """
        return cls(dim_s=dim_s, start_idx=0, dim_a=dim_a)


class Prism(nn.Module):
    """Neural prism-style matcher/builder for conditional branches."""

    def __init__(self, match_net: nn.Module, build_net: nn.Module):
        super().__init__()
        self.match_net = match_net
        self.build_net = build_net

    def match(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.match_net(s)
        a, logit = out.chunk(2, dim=-1)
        confidence = torch.sigmoid(logit.mean(dim=-1, keepdim=True))
        return a, confidence

    def build(self, a_new: torch.Tensor) -> torch.Tensor:
        return self.build_net(a_new)

    def review(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, conf = self.match(s)
        return self.build(a), conf

    @classmethod
    def gating(cls, dim_s: int, dim_a: int) -> "Prism":
        match_net = nn.Sequential(
            nn.Linear(dim_s, dim_a * 2),
            nn.SiLU(),
        )
        build_net = nn.Linear(dim_a, dim_s)
        return cls(match_net, build_net)


class Traversal(nn.Module):
    """Parallel access/update over multiple learned positions."""

    def __init__(self, dim_s: int, dim_a: int, num_positions: int):
        super().__init__()
        self.num_positions = num_positions
        self.dim_a = dim_a
        self.getters = nn.ModuleList(
            [nn.Linear(dim_s, dim_a) for _ in range(num_positions)]
        )
        self.putter = nn.Linear(num_positions * dim_a, dim_s)

    def get_all(self, s: torch.Tensor) -> torch.Tensor:
        parts = [getter(s) for getter in self.getters]
        return torch.stack(parts, dim=1)

    def put_all(self, s: torch.Tensor, parts: torch.Tensor) -> torch.Tensor:
        flat = parts.reshape(parts.shape[0], -1)
        return self.putter(flat)

    def traverse(self, s: torch.Tensor, f) -> torch.Tensor:
        parts = self.get_all(s)
        modified = torch.stack(
            [f(parts[:, i, :]) for i in range(self.num_positions)],
            dim=1,
        )
        return self.put_all(s, modified)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.get_all(s)


class VanLaarhovenLens(nn.Module):
    """
    Practical neural lens inspired by the Van Laarhoven representation.

    With arbitrary `nn.Module` blocks we cannot recover the fully parametric
    representation, so composition is implemented as sequential neural
    composition while preserving the same call surface.
    """

    def __init__(self, focus_net: nn.Module, reconstruct_net: nn.Module):
        super().__init__()
        self.focus = focus_net
        self.reconstruct = reconstruct_net

    def forward(self, s: torch.Tensor, f=None) -> torch.Tensor:
        focused = self.focus(s)
        if f is not None:
            focused = f(focused)
        return self.reconstruct(torch.cat([s, focused], dim=-1))

    def compose(self, other: "VanLaarhovenLens") -> "ComposedVanLaarhovenLens":
        return ComposedVanLaarhovenLens(first=other, second=self)


class ComposedVanLaarhovenLens(nn.Module):
    """Sequential composition wrapper for practical neural lenses."""

    def __init__(self, first: VanLaarhovenLens, second: VanLaarhovenLens):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, s: torch.Tensor, f=None) -> torch.Tensor:
        return self.second(self.first(s, f=f))
