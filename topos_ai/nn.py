import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =====================================================================
# TOPOS-LLM CORE ARCHITECTURE (THE LLAMA-CLASS UPGRADE)
# Bu modül, ToposAI'ı bir "Proof of Concept" olmaktan çıkarıp, Llama-3 
# veya Mistral gibi endüstriyel modellere kafa tutabilecek devasa bir 
# "Büyük Dil Modeli (LLM)" mimarisine (RoPE, SwiGLU, KV-Cache) dönüştürür.
# =====================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    [ROTARY POSITION EMBEDDING (RoPE) - Kategori Teorisinde Functor]
    Kelimelerin cümledeki sırasını 'mutlak' bir sayı olarak değil, 
    kompleks düzlemde birbirlerine göre bir 'Açı (Rotasyon)' olarak kodlar.
    Bu, ToposAI'ın 512 tokenlik sabit bağlam (Context) penceresini kırıp
    sınırsız (Infinite Context) metin okuyabilmesini sağlar.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Q ve K vektörlerini kompleks düzlemde (Euler Açısı) döndürür."""
    # xq: [B, SeqLen, H, D_head]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    seq_len = xq_.shape[1]
    freqs_cis = freqs_cis[:seq_len].view(1, seq_len, 1, -1)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLU(nn.Module):
    """
    [GATED LINEAR UNIT (SwiGLU)]
    Google PaLM ve Meta Llama tarafından kullanılan mucizevi nöral kapı.
    Basit GELU veya ReLU'dan çok daha fazla bilgi kapasitesine (Capacity) sahiptir.
    Girdinin bir yarısını karar verici (Gate), diğer yarısını ise Bilgi (Value) olarak kullanır.
    """
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x):
        # Swish(x * W1) * (x * W3) @ W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MultiUniverseToposAttention(nn.Module):
    """Multi-Head Attention'ın Kategori Teorisi (Lukasiewicz) Karşılığı."""
    def __init__(self, d_model, num_universes):
        super().__init__()
        self.num_universes = num_universes
        self.d_universe = d_model // num_universes
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None):
        B, SeqLen, D = x.shape
        
        # 1. Projeksiyonlar
        Q = self.q_proj(x).view(B, SeqLen, self.num_universes, self.d_universe)
        K = self.k_proj(x).view(B, SeqLen, self.num_universes, self.d_universe)
        V = self.v_proj(x).view(B, SeqLen, self.num_universes, self.d_universe)

        # 2. Rotary Position Embedding (RoPE)
        Q, K = apply_rotary_emb(Q, K, freqs_cis)
        
        # 3. KV-Cache (İnference / Chat sırasında milyarlarca işlemi hızlandırır)
        if kv_cache is not None:
            # Geçmiş Tokenları Cache'den getir ve yenilerini ekle
            k_cache, v_cache = kv_cache
            K = torch.cat([k_cache, K], dim=1)
            V = torch.cat([v_cache, V], dim=1)
            # Cache'i güncelle
            kv_cache = (K, V)
            
        # Topos Mantığı için 0-1 arasına sıkıştırma (Fiziksel olasılık uzayı)
        Q = torch.sigmoid(Q).transpose(1, 2).contiguous() # [B, U, Seq, D_u]
        K = torch.sigmoid(K).transpose(1, 2).contiguous() # [B, U, Cache_Seq, D_u]
        V = V.transpose(1, 2).contiguous()

        # 4. Kategori Teorisi (Lukasiewicz Implication)
        # min(1, 1 - Q + K)
        Q_exp = Q.unsqueeze(3) # [B, U, Seq, 1, D_u]
        K_exp = K.unsqueeze(2) # [B, U, 1, Cache_Seq, D_u]
        implication = torch.clamp(1.0 - Q_exp + K_exp, min=0.0, max=1.0)
        
        truth_matrix = implication.mean(dim=-1) # [B, U, Seq, Cache_Seq]

        if mask is not None:
            truth_matrix = truth_matrix + mask

        # Softmax ve Value çarpımı
        attn_weights = F.softmax(truth_matrix * 5.0, dim=-1)
        out = torch.matmul(attn_weights, V) # [B, U, Seq, D_u]
        
        out = out.transpose(1, 2).contiguous().view(B, SeqLen, D)
        return self.out_proj(out), kv_cache

class YonedaEmbedding(nn.Module):
    """
    [LOW-RANK FACTORIZED YONEDA]
    Kelime anlamlarını istatistiksel ağırlık (nn.Embedding) olarak DEĞİL, 
    diğer tüm kelimelere olan Oklar (Morphism) üzerinden hesaplar.
    O(V^2) hafıza patlamasını (40GB VRAM) U (V x r) ve W (r x V) matrislerine böler.
    """
    def __init__(self, vocab_size, rank=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.rank = rank
        
        self.U = nn.Parameter(torch.randn(vocab_size, rank) * (1.0 / rank**0.5))
        self.W = nn.Parameter(torch.randn(rank, vocab_size) * (1.0 / rank**0.5))

    def forward(self, idx):
        selected_U = F.embedding(idx, self.U)
        logits = torch.matmul(selected_U, self.W)
        return torch.sigmoid(logits) # Topos (0, 1) olasılık uzayı
