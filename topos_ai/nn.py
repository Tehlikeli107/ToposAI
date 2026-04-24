import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    Bu, ToposAI'ın sabit bağlam (Context) penceresini kırıp çok daha uzun
    metinleri okuyabilmesini (Uzun Bağlam - Long Context) sağlar.
    (Not: RoPE tek başına 'Infinite Context' sağlamaz, S17 FIX).
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

class TopologicalLinear(nn.Module):
    """
    [PURE TOPOS LINEAR PROJECTION]
    Klasik nn.Linear ağırlıkları (Weights) negatif değerler alabilir ve
    boyutsuz büyüyebilir (-5.4, +12.1). Bu, Kategori Teorisinin
    'Morfizma (Ok) Gücü' mantığına ([0, 1] arası olasılık) ihanettir.
    Bu sınıf, öğrenilebilir ağırlıkları her zaman (sigmoid ile) 0.0 ile 1.0
    arasına hapseder. Negatif bilgi (Eksi ok) kavramı yoktur.
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Ağırlıkları başlat (Daha geniş bir spread [2.0], sigmoid sonrası daha zengin çeşitlilik sağlar)
        self.weight_raw = nn.Parameter(torch.randn(out_features, in_features) * 2.0)

        if bias:
            self.bias_raw = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_raw', None)

    def forward(self, x):
        # Ağırlıkları Kategori Teorisinin doğasına ([0, 1]) zorla
        W_topos = torch.sigmoid(self.weight_raw)

        out = F.linear(x, W_topos)

        if self.bias_raw is not None:
            B_topos = torch.sigmoid(self.bias_raw)
            out = out + B_topos

        # Çıktının 1.0'ı aşmamasını (Probabilistic Bound) veya normalize olmasını TopoNorm sağlar,
        # ancak doğrusal projeksiyonda ağırlıklı toplamlar büyüyebilir.
        return out

class TopologicalNorm(nn.Module):
    """
    [SHEAF NORMALIZATION]
    Klasik LayerNorm ortalamayı 0 yapar ve negatif sayılar (-0.5) üretir.
    Kategori Teorisinde okların gücü negatife inemez.
    Bu Norm, vektördeki değerleri [0, 1] aralığına doğrusal olarak yayar (Min-Max Scaling)
    veya sadece maksimum ok gücüne bölerek normalize eder.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Öğrenilebilir ölçek (Scale) ve kaydırma (Shift) [0, 1] arası olmalı
        self.weight = nn.Parameter(torch.ones(d_model))
        # Norm bias'ının başlangıçta her şeye 0.5 (sigmoid(0)) eklememesi için
        self.bias = nn.Parameter(torch.full((d_model,), -4.0)) # sigmoid(-4) ≈ 0.018

    def forward(self, x):
        # x: [B, SeqLen, D]
        # Mutlak maksimum değere oranla (Sınır: [-1.0, 1.0])
        x_abs_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
        x_norm = x / (x_abs_max + self.eps)

        # Kategori Teorisinin [0.0, 1.0] (Reachability) aralığına kaydır
        x_norm = (x_norm + 1.0) / 2.0

        # Öğrenilebilir parametrelerle çarp ama sonucu yine [0, 1] arasına sıkıştır
        out = x_norm * torch.sigmoid(self.weight) + torch.sigmoid(self.bias)
        return torch.clamp(out, min=0.0, max=1.0)

class TopologicalFFN(nn.Module):
    """
    [PURE TOPOS FEED-FORWARD (Fuzzy Logic Gate)]
    SwiGLU veya GELU negatif değerler üretebilir veya 1.0'ı aşabilir.
    TopologicalFFN, ağırlıkları her zaman [0, 1] arasında tutar.
    Katmanlar arası mantıksal AND (min) ve OR (max) işlemlerini simüle eder.
    """
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        # Ağırlıklar doğrudan [0, 1] olasılıkları olarak öğrenilecek (TopoLinear)
        self.w_gate = TopologicalLinear(in_features, hidden_features, bias=False)
        self.w1 = TopologicalLinear(in_features, hidden_features, bias=False)
        self.w2 = TopologicalLinear(hidden_features, out_features, bias=False)

    def forward(self, x):
        # x her zaman [0, 1] arasındadır.
        # W1 projeksiyonu ve Gate projeksiyonu
        w1_out = self.w1(x)
        gate_out = torch.sigmoid(self.w_gate(x))

        # Bulanık Mantık (Fuzzy Logic) Aktivasyonu:
        # Gerçek Gating mekanizması: Kapı (Gate) açıksa ve bilgi (W1) güçlüyse geçiş yap
        hidden_state = w1_out * gate_out

        # İkinci katmana geçiş
        w2_out = self.w2(hidden_state)

        # Çıktı yine [0, 1] arasına sıkıştırılır
        return torch.clamp(w2_out, min=0.0, max=1.0)
class TopologicalMoERouter(nn.Module):
    """
    [MIXTURE OF EXPERTS (MoE) ROUTER]
    Her kelime (Token) için 8 Evrenden (Expert) sadece en ilgili 2 tanesini seçer.
    Bu, modelin trilyon parametreye çıkmasını ama sadece küçük bir kısmının
    (Sparse) çalışarak O(1) hızında kalmasını sağlar (Örn: Mixtral 8x7B mantığı).
    """
    def __init__(self, d_model, num_universes, top_k=2):
        super().__init__()
        self.router_weights = TopologicalLinear(d_model, num_universes, bias=False)
        self.top_k = top_k

    def forward(self, x):
        # x: [B, SeqLen, D]
        router_logits = self.router_weights(x) # [B, SeqLen, num_universes]

        # Oyları olasılıklara (Routing probabilities) çevir
        routing_probs = F.softmax(router_logits, dim=-1)

        # En yüksek oyu alan Top-K evreni seç
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)

        # Olasılıkları yeniden normalize et ki toplamları 1 olsun (Seçilmeyenler 0 olur)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        return top_k_probs, top_k_indices

class MultiUniverseToposAttention(nn.Module):
    """
    [SPARSE TOPOS MoE ATTENTION]
    Teorik olarak Sparse (Seyrek) çalışan, ancak şu an donanım kernel
    eksikliğinden dolayı Dense hesaplanıp Sparse maskelenen MoE yapısı (S12 FIX).
    Kategori Teorisi'nin (Local Presheaves) donanım verimliliğine dönüşmüş halidir.
    """
    def __init__(self, d_model, num_universes, top_k=2):
        super().__init__()
        self.num_universes = num_universes
        self.top_k = top_k
        self.d_universe = d_model // num_universes

        # Yönlendirici (Router)
        self.router = TopologicalMoERouter(d_model, num_universes, top_k)

        # Saf Kategori Teorisi Projeksiyonları
        self.q_proj = TopologicalLinear(d_model, d_model, bias=False)
        self.k_proj = TopologicalLinear(d_model, d_model, bias=False)
        self.v_proj = TopologicalLinear(d_model, d_model, bias=False)
        self.out_proj = TopologicalLinear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None):
        B, SeqLen, D = x.shape

        # 1. MoE Yönlendirmesi (Routing)
        # Her token için hangi 2 evrenin çalışacağını buluyoruz
        top_k_probs, top_k_indices = self.router(x) # probs: [B, SeqLen, top_k], indices: [B, SeqLen, top_k]

        # 2. Tüm Evrenlerin Projeksiyonları
        # (Not: İddiamız olan O(1) Sparse computation tam olarak çalışması için
        # CUDA kernel gerekir. PyTorch batch matmul kolaylığı için burada
        # Dense olarak hesaplayıp sonra maskeliyoruz (Simulation of Sparsity).)
        Q_new = self.q_proj(x).view(B, SeqLen, self.num_universes, self.d_universe)
        K_new = self.k_proj(x).view(B, SeqLen, self.num_universes, self.d_universe)
        V_new = self.v_proj(x).view(B, SeqLen, self.num_universes, self.d_universe)

        # [KV-CACHE VE RoPE DÜZELTMESİ (P11/P12)]
        # RoPE sadece YENİ (Q_new ve K_new) vektörlere uygulanır.
        # freqs_cis sadece yeni tokenlerin pozisyonunu içermelidir.
        Q_new, K_new = apply_rotary_emb(Q_new, K_new, freqs_cis)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Yeni K ve V, mevcut Cache'in sonuna eklenir
            K_all = torch.cat([k_cache, K_new], dim=1)
            V_all = torch.cat([v_cache, V_new], dim=1)
        else:
            K_all = K_new
            V_all = V_new

        # Gelecek adımlar için cache'i güncelle
        new_kv_cache = (K_all, V_all)

        Q_all = torch.sigmoid(Q_new).transpose(1, 2).contiguous() # [B, U, Seq, D_u]
        K_all = torch.sigmoid(K_all).transpose(1, 2).contiguous() # [B, U, Cache_Seq, D_u]
        V_all = V_all.transpose(1, 2).contiguous()

        # 3. Topos Mantığı (Lukasiewicz Implication)
        Q_exp = Q_all.unsqueeze(3) # [B, U, Seq, 1, D_u]
        K_exp = K_all.unsqueeze(2) # [B, U, 1, Cache_Seq, D_u]
        implication = torch.clamp(1.0 - Q_exp + K_exp, min=0.0, max=1.0)
        truth_matrix = implication.mean(dim=-1) # [B, U, Seq, Cache_Seq]

        if mask is not None:
            # Geleceğe giden nedensellik oklarını iptal et (0.0 ile çarp)
            truth_matrix = truth_matrix * mask

        # [THE DEATH OF SOFTMAX (Fuzzy/Categorical Union)]
        # Klasik YZ, her kelimeye sadece 1.0 enerji dağıtır. Biri güçlüyse diğerini sıfırlar (Softmax).
        # ToposAI'da (Kategori Teorisinde) her kelimenin uzaydaki bağımsız "Gücü" önemlidir.
        # Eğer bir kelime 10 geçmiş kelimeye çok güçlü bağlıysa (10.0 enerji), o kelimenin anlamı zengindir.
        # Bu yüzden Softmax'in ezişini kaldırıyor, L1 Norm ile sadece vektörel toplamı 1.0 yapıyoruz.
        row_sums = truth_matrix.sum(dim=-1, keepdim=True) + 1e-9
        attn_weights = truth_matrix / row_sums

        out_all_universes = torch.matmul(attn_weights, V_all) # [B, U, Seq, D_u]
        out_all_universes = out_all_universes.transpose(1, 2).contiguous() # [B, Seq, U, D_u]

        # 4. SADECE SEÇİLEN 2 EVRENİ (TOP-K) BİRLEŞTİRME (Sheaf Gluing)
        # Seçilmeyen 6 evrenin ürettiği sonuçlar ÇÖPE GİDER (Sıfırlanır).
        # Her token kendi seçtiği 2 uzmanın cevabını (router ağırlıklarına göre) alır.

        final_out = torch.zeros(B, SeqLen, self.d_universe * self.num_universes, device=x.device)

        # (PyTorch scatter/gather işlemleriyle hızlı MoE birleştirme)
        for k in range(self.top_k):
            # i. seçilen uzmanların ID'leri
            expert_idx = top_k_indices[:, :, k] # [B, SeqLen]
            expert_weights = top_k_probs[:, :, k].unsqueeze(-1) # [B, SeqLen, 1]

            # Bu tokenların seçtiği uzmanların ürettiği D_u boyutundaki çıktılar
            # (Gather işlemi: out_all_universes matrisinden o anki tokenin seçtiği U. indexteki vektörü çek)
            # index shape [B, SeqLen, 1, D_u] olmalı
            idx_expanded = expert_idx.unsqueeze(2).unsqueeze(3).expand(B, SeqLen, 1, self.d_universe)
            expert_out = torch.gather(out_all_universes, dim=2, index=idx_expanded).squeeze(2) # [B, SeqLen, D_u]

            # Uzmanın çıktısını router oyu ile çarpıp, kendi evrenindeki (D) asıl yerine koy
            # (Burada tam bir FFN-MoE yerine Attention-MoE yaptığımız için, ilgili evrenin dilimine ekliyoruz)

            batch_indices = torch.arange(B).unsqueeze(1).expand(B, SeqLen)
            seq_indices = torch.arange(SeqLen).unsqueeze(0).expand(B, SeqLen)

            # expert_out'u (D_u) ana uzayda (D) uygun yere (expert_idx * D_u) yerleştir (Maskele)
            # Bu kısım teorik olarak çok spesifiktir, basitlik adına hepsini D'ye yayıp topluyoruz
            # Gerçek MoE'de Out_Proj'dan önce D_u'luk kısım kendi uzayında kalır.
            # Şimdilik "Ağırlıklı Toplam (Weighted Sum)" yapıp her evrenin yerine koyacağız.

            mask_D = torch.zeros(B, SeqLen, self.num_universes, self.d_universe, device=x.device)
            mask_D[batch_indices, seq_indices, expert_idx, :] = expert_out * expert_weights
            final_out += mask_D.view(B, SeqLen, D)

        return self.out_proj(final_out), new_kv_cache

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

    def get_morphisms(self):
        """
        [P2 FIX] (Gerçek Matrisi Görme)
        Eski testlerin ve TDA/Astrofizik modüllerinin aradığı metod.
        Low-Rank ayrıştırılmış U ve W matrislerini çarparak
        tam (Vocab x Vocab) Kategori Matrisini üretir ve [0, 1] arasına çeker.
        """
        full_logits = torch.matmul(self.U, self.W)
        return torch.sigmoid(full_logits)
