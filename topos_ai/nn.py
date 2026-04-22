import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiUniverseToposAttention(nn.Module):
    """Multi-Head Attention'ın Kategori Teorisi Karşılığı. Evrenlere (Local Truths) Böler."""
    def __init__(self, d_model, num_universes):
        super().__init__()
        self.num_universes = num_universes
        self.d_universe = d_model // num_universes
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, SeqLen, D = x.shape
        Q = torch.sigmoid(self.q_proj(x)).view(B, SeqLen, self.num_universes, self.d_universe).transpose(1, 2).contiguous()
        K = torch.sigmoid(self.k_proj(x)).view(B, SeqLen, self.num_universes, self.d_universe).transpose(1, 2).contiguous()
        V = self.v_proj(x).view(B, SeqLen, self.num_universes, self.d_universe).transpose(1, 2).contiguous()

        Q_exp = Q.unsqueeze(3) 
        K_exp = K.unsqueeze(2) 
        implication = torch.clamp(1.0 - Q_exp + K_exp, max=1.0)
        truth_matrix = implication.mean(dim=-1)

        if mask is not None:
            # Topological Decay Mask (Yumuşak Causal Mask)
            idx_q = torch.arange(SeqLen, device=x.device).unsqueeze(1)
            idx_k = torch.arange(SeqLen, device=x.device).unsqueeze(0)
            distance = idx_k - idx_q # Geleceğe doğru pozitif, geçmişe doğru negatif
            
            # Gelecekteki tokenların ağırlığını mesafeye göre YUMUŞAK (Soft) olarak düşür (Decay)
            decay_mask = torch.where(distance > 0, -distance.float() * 1.5, 0.0).view(1, 1, SeqLen, SeqLen)
            truth_matrix = truth_matrix + decay_mask

        attn_weights = F.softmax(truth_matrix * 5.0, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, SeqLen, D)
        return self.out_proj(out)

class YonedaEmbedding(nn.Module):
    """
    [LOW-RANK FACTORIZED YONEDA]
    Sabit vektör (nn.Embedding) kullanmaz. Anlamı, kelimelerin birbirleriyle olan okları 
    (morphism) üzerinden hesaplar. O(V^2) hafıza patlamasını engellemek için devasa 
    V x V matrisini, V x r ve r x V (Low-Rank) olarak iki alt uzaya sıkıştırır.
    """
    def __init__(self, vocab_size, rank=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.rank = rank
        
        # O(V^2) yerine O(V * r) hafıza kullanımı
        self.U = nn.Parameter(torch.randn(vocab_size, rank) * (1.0 / rank**0.5))
        self.W = nn.Parameter(torch.randn(rank, vocab_size) * (1.0 / rank**0.5))

    def get_morphisms(self):
        """Testler ve Benchmarkerlar için Yoneda Olasılık Matrisi"""
        logits = torch.matmul(self.U, self.W)
        return torch.sigmoid(logits)

    def forward(self, idx):
        # Memory Efficient Routing
        selected_U = F.embedding(idx, self.U)
        logits = torch.matmul(selected_U, self.W)
        return torch.sigmoid(logits)

class DynamicToposUniverse(nn.Module):
    """Paradoks anında kendi boyutunu genişleten (Self-Modifying) Kategori Matrisi."""
    def __init__(self, initial_entities):
        super().__init__()
        self.num_entities = initial_entities
        self.relation_logits = nn.Parameter(torch.randn(initial_entities, initial_entities))

    def evolve_universe(self):
        new_size = self.num_entities + 1
        old_logits = self.relation_logits.data
        new_logits = torch.randn(new_size, new_size, device=old_logits.device)
        new_logits[:self.num_entities, :self.num_entities] = old_logits
        self.num_entities = new_size
        self.relation_logits = nn.Parameter(new_logits)
        return new_size
