import torch
import torch.nn as nn
from .nn import MultiUniverseToposAttention

class ToposTransformerBlock(nn.Module):
    def __init__(self, d_model, num_universes):
        super().__init__()
        self.muta = MultiUniverseToposAttention(d_model, num_universes)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.muta(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class ToposTransformer(nn.Module):
    """Uçtan uca eğitilebilir, Dot-Product içermeyen tam donanımlı Topos Dil Modeli."""
    def __init__(self, vocab_size, d_model=64, num_universes=4, num_layers=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)
        self.blocks = nn.ModuleList([ToposTransformerBlock(d_model, num_universes) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, SeqLen = idx.shape
        pos = torch.arange(0, SeqLen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        # Geleceği Görme Engeli
        mask = torch.tril(torch.ones(SeqLen, SeqLen, device=idx.device)).view(1, 1, SeqLen, SeqLen)
        for block in self.blocks:
            x = block(x, mask)
            
        return self.fc_out(self.norm(x))
