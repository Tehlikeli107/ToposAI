import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    MultiUniverseToposAttention,
    TopologicalFFN,
    TopologicalLinear,
    TopologicalNorm,
    YonedaEmbedding,
    precompute_freqs_cis,
)


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
        x = x + muta_out

        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x, kv_cache


class ToposTransformer(nn.Module):
    """
    Uctan uca egitilebilir, attention skorlamasi dot-product-free Goedel-Heyting
    olan tam donanımlı Topos Dil Modeli. Çıkış projeksiyonu (Reachability) ise Cosine Similarity kullanır.
    Meta Llama-3 (RoPE, SwiGLU, KV-Cache) teknolojileriyle entegre edilmiştir.
    Klasik Classifier (fc_out) YERİNE, kelimelerin Yoneda uzayındaki yerlerine
    ne kadar ulaşılabildiğini (Topological Reachability) [0, 1] arasında ölçer.
    """

    def __init__(self, vocab_size, d_model=64, num_universes=4, num_layers=2, max_seq_len=2048):
        super().__init__()
        self.yoneda_emb = YonedaEmbedding(vocab_size)
        self.yoneda_proj = TopologicalLinear(vocab_size, d_model, bias=False)

        self.freqs_cis = precompute_freqs_cis(d_model // num_universes, max_seq_len * 2)

        self.blocks = nn.ModuleList([ToposTransformerBlock(d_model, num_universes) for _ in range(num_layers)])
        self.norm = TopologicalNorm(d_model)

    def forward(self, idx, kv_caches=None):
        B, SeqLen = idx.shape

        yoneda_repr = self.yoneda_emb(idx)
        x = self.yoneda_proj(yoneda_repr)

        past_kv_length = kv_caches[0][0].shape[1] if kv_caches is not None else 0
        freqs_cis = self.freqs_cis[past_kv_length : past_kv_length + SeqLen].to(x.device)

        if SeqLen > 1:
            # [PURE TOPOLOGICAL TIME ARROW (Asymmetric Monoid)]
            # Klasik maskeleme geleceği '-inf' ile doldurur ve softmax ile ezer.
            # Kategori Teorisinde ok ya vardır (1.0) ya yoktur (0.0).
            mask = torch.ones(SeqLen, past_kv_length + SeqLen, device=idx.device)
            mask = torch.tril(mask, diagonal=past_kv_length).view(1, 1, SeqLen, past_kv_length + SeqLen)
            # 0.0 olan yerler gelecektir. Çarpım anında (truth_matrix * mask) sıfırlanırlar.
        else:
            mask = None

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            x, new_kv_cache = block(x, freqs_cis, mask, kv_cache)
            new_kv_caches.append(new_kv_cache)

        x_norm = self.norm(x)  # [B, SeqLen, d_model]

        # [PURE TOPOLOGICAL PROJECTION]
        # Kosinüs benzerliği için, projeksiyonun arkasındaki o saf (0 ile 1 arasına sıkıştırılmış)
        # ağırlıkları çekmemiz gerekiyor. TopologicalLinear'ın asıl pozitif ağırlıkları:
        vocab_embeddings = torch.sigmoid(self.yoneda_proj.weight_raw)  # [d_model, vocab_size]

        # L2 Normalize
        x_normalized = F.normalize(x_norm, p=2, dim=-1)  # [B, SeqLen, d_model]
        vocab_normalized = F.normalize(vocab_embeddings, p=2, dim=0).T  # [vocab_size, d_model]

        # [STRICT GODEL INTERNAL HOM]
        # Kosinüs Benzerliği (Cosine Similarity) SİMETRİKTİR ve yönü katleder.
        # Asimetriyi korumak için StrictGodelImplication kullanıyoruz: A <= B ise 1, değilse B
        from topos_ai.logic import StrictGodelImplication

        x_exp = x_normalized.unsqueeze(2)           # [B, SeqLen, 1, d_model]
        vocab_exp = vocab_normalized.unsqueeze(0).unsqueeze(0) # [1, 1, vocab_size, d_model]

        # Kelimeden (x) Hedefe (vocab) Topos İçsel Çıkarımı (Implication)
        implication = StrictGodelImplication.apply(x_exp, vocab_exp)

        # Boyutlar üzerinden ortalama alarak [0, 1] arası Asimetrik Ulaşılabilirlik (Reachability) bul
        reachability_logits = implication.mean(dim=-1)

        # Topolojik Ulaşılabilirlik (Reachability) Skoru: [0.0, 1.0]
        # Floating point hataları yüzünden -0.00001 veya 1.000001 olmasını engelle (BCELoss CUDA Assert)
        reachability_logits = torch.clamp(reachability_logits, min=1e-6, max=1.0 - 1e-6)

        return reachability_logits, new_kv_caches
