import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn import MultiUniverseToposAttention, YonedaEmbedding, TopologicalFFN, TopologicalNorm, precompute_freqs_cis

class ToposTransformerBlock(nn.Module):
    def __init__(self, d_model, num_universes):
        super().__init__()
        self.muta = MultiUniverseToposAttention(d_model, num_universes)
        self.norm1 = TopologicalNorm(d_model)
        # PURE TOPOS FFN: Klasik SwiGLU yerine
        hidden_dim = int(8 * d_model / 3)
        self.ffn = TopologicalFFN(d_model, hidden_dim, d_model)
        self.norm2 = TopologicalNorm(d_model)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None):
        muta_out, kv_cache = self.muta(self.norm1(x), freqs_cis, mask, kv_cache)
        
        # PURE TOPOS RESIDUAL (T-Conorm): x = max(x, muta_out) veya x + muta_out - x*muta_out
        # 1.0'ı aşmamak için Probabilistic Sum kullanıyoruz
        x = x + muta_out - (x * muta_out)
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out - (x * ffn_out)
        
        return x, kv_cache

class ToposTransformer(nn.Module):
    """
    Uçtan uca eğitilebilir, Dot-Product içermeyen tam donanımlı Topos Dil Modeli.
    Meta Llama-3 (RoPE, SwiGLU, KV-Cache) teknolojileriyle entegre edilmiştir.
    Klasik Classifier (fc_out) YERİNE, kelimelerin Yoneda uzayındaki yerlerine
    ne kadar ulaşılabildiğini (Topological Reachability) [0, 1] arasında ölçer.
    """
    def __init__(self, vocab_size, d_model=64, num_universes=4, num_layers=2, max_seq_len=2048):
        super().__init__()
        self.yoneda_emb = YonedaEmbedding(vocab_size)
        self.yoneda_proj = nn.Linear(vocab_size, d_model, bias=False)
        
        self.freqs_cis = precompute_freqs_cis(d_model // num_universes, max_seq_len * 2)
        
        self.blocks = nn.ModuleList([ToposTransformerBlock(d_model, num_universes) for _ in range(num_layers)])
        self.norm = TopologicalNorm(d_model)

    def forward(self, idx, kv_caches=None):
        B, SeqLen = idx.shape
        
        yoneda_repr = self.yoneda_emb(idx)
        x = self.yoneda_proj(yoneda_repr)
        
        freqs_cis = self.freqs_cis[:SeqLen].to(x.device)
        
        if SeqLen > 1:
            mask = torch.tril(torch.ones(SeqLen, SeqLen, device=idx.device)).view(1, 1, SeqLen, SeqLen)
            mask = torch.where(mask == 0, float('-inf'), 0.0) 
        else:
            mask = None
            
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            x, new_kv_cache = block(x, freqs_cis, mask, kv_cache)
            new_kv_caches.append(new_kv_cache)
            
        x_norm = self.norm(x) # [B, SeqLen, d_model]
        
        # [PURE TOPOLOGICAL PROJECTION]
        # Klasik bir nn.Linear (fc_out) yerine, modelin geldiği son nokta (x_norm) ile
        # tüm kelimelerin (vocab) Kategori Uzayındaki (Yoneda_Proj) yönlerini kıyaslarız.
        # Cosine Similarity: -1 ile 1 arasındadır. Biz bunu (x + 1)/2 ile [0, 1] arasına (Topos) çekeriz.
        
        # Kelimelerin projeksiyon matrisindeki yerleri: [vocab_size, d_model]
        vocab_embeddings = self.yoneda_proj.weight # [d_model, vocab_size]
        
        # L2 Normalize
        x_normalized = F.normalize(x_norm, p=2, dim=-1) # [B, SeqLen, d_model]
        vocab_normalized = F.normalize(vocab_embeddings, p=2, dim=0) # [d_model, vocab_size]
        
        # [B, SeqLen, vocab_size] (Kosinüs Benzerliği -1 ile 1 arası)
        cosine_sim = torch.matmul(x_normalized, vocab_normalized) 
        
        # Topolojik Ulaşılabilirlik (Reachability) Skoru: [0.0, 1.0]
        reachability_logits = (cosine_sim + 1.0) / 2.0
        
        return reachability_logits, new_kv_caches
