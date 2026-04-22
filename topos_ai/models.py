import torch
import torch.nn as nn
from .nn import MultiUniverseToposAttention, YonedaEmbedding, SwiGLU, precompute_freqs_cis

class ToposTransformerBlock(nn.Module):
    def __init__(self, d_model, num_universes):
        super().__init__()
        self.muta = MultiUniverseToposAttention(d_model, num_universes)
        self.norm1 = nn.LayerNorm(d_model)
        # SwiGLU: Llama-3/PaLM standardı
        hidden_dim = int(8 * d_model / 3)
        self.ffn = SwiGLU(d_model, hidden_dim, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None):
        muta_out, kv_cache = self.muta(self.norm1(x), freqs_cis, mask, kv_cache)
        x = x + muta_out
        x = x + self.ffn(self.norm2(x))
        return x, kv_cache

class ToposTransformer(nn.Module):
    """
    Uçtan uca eğitilebilir, Dot-Product içermeyen tam donanımlı Topos Dil Modeli.
    Meta Llama-3 (RoPE, SwiGLU, KV-Cache) teknolojileriyle entegre edilmiştir.
    """
    def __init__(self, vocab_size, d_model=64, num_universes=4, num_layers=2, max_seq_len=2048):
        super().__init__()
        self.yoneda_emb = YonedaEmbedding(vocab_size)
        self.yoneda_proj = nn.Linear(vocab_size, d_model)
        
        # Absolute pos_emb kaldırıldı. Yerine precomputed RoPE (Rotary) açıları:
        self.freqs_cis = precompute_freqs_cis(d_model // num_universes, max_seq_len * 2)
        
        self.blocks = nn.ModuleList([ToposTransformerBlock(d_model, num_universes) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, idx, kv_caches=None):
        B, SeqLen = idx.shape
        
        yoneda_repr = self.yoneda_emb(idx)
        x = self.yoneda_proj(yoneda_repr)
        
        # RoPE açılarını o anki seq_len'e göre al
        freqs_cis = self.freqs_cis[:SeqLen].to(x.device)
        
        # Geleceği Görme Engeli (Causal Mask). Sadece Training için. Inference'da KV-Cache tek token gelir.
        if SeqLen > 1:
            mask = torch.tril(torch.ones(SeqLen, SeqLen, device=idx.device)).view(1, 1, SeqLen, SeqLen)
            mask = torch.where(mask == 0, float('-inf'), 0.0) # Decay yerine Absolute Causal (GPT) standardı
        else:
            mask = None
            
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            x, new_kv_cache = block(x, freqs_cis, mask, kv_cache)
            new_kv_caches.append(new_kv_cache)
            
        logits = self.fc_out(self.norm(x))
        return logits, new_kv_caches
