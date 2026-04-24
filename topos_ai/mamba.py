import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import TopologicalLinear, TopologicalNorm

# =====================================================================
# TOPOS-MAMBA (CATEGORICAL STATE SPACE MODELS)
# Problem: Attention (Transformer) mekanizması O(N^2) olduğu için
# uzun metinlerde (100K+ token) GPU'yu kilitler ve yavaştır.
# Çözüm: Kategori Teorisini "Sürekli Dinamik Sistemlere (SSM)" uygularız.
# ToposMamba, her kelimeyi diğerleriyle kıyaslamaz. Bunun yerine
# [0, 1] aralığında sınırlı bir "Topolojik Hafıza (State)" tutar.
# Her yeni kelime (Morphism), bu hafızayı Lukasiewicz/Fuzzy mantığıyla
# günceller (Fold). Karmaşıklık O(N)'dir. Sonsuz bağlam (Context) destekler.
# =====================================================================

class ToposMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # [KATEGORİ GEÇİŞ (MORPHISM) MATRİSLERİ]
        # A: Geçmişi ne kadar unutacağımız (Forget Gate / Subobject Classifier)
        # B: Yeni bilginin sisteme nasıl gireceği (Input Functor)
        # C: İçsel durumun dışarı nasıl yansıyacağı (Output Functor)

        self.A = nn.Parameter(torch.rand(d_model, d_state))
        self.B = TopologicalLinear(d_model, d_state, bias=False)
        self.C = TopologicalLinear(d_state, d_model, bias=False)

        self.norm = TopologicalNorm(d_model)

    def forward(self, x, state=None):
        """
        x: [Batch, SeqLen, D_model]
        state: [Batch, D_model, D_state] (Opsiyonel önceki hafıza)
        """
        B_batch, SeqLen, D = x.shape

        # Topolojik Hafıza (Başlangıçta Sıfır/Boşluk VEYA Önceki State)
        if state is None:
            h = torch.zeros(B_batch, self.d_model, self.d_state, device=x.device)
        else:
            h = state

        outs = []

        # A matrisini [0, 1] arasına çek (Kategori Uzayı Sınırları)
        # A_sig = 1.0 demek hafızayı %100 koru, 0.0 demek geçmişi tamamen sil
        A_sig = torch.sigmoid(self.A) # [D_model, d_state]
        A_exp = A_sig.unsqueeze(0)    # [1, D_model, d_state]

        # [O(N) TOPOLOGICAL SCAN - LINEAR TIME]
        for t in range(SeqLen):
            x_t = x[:, t, :] # [B_batch, D_model]

            # 1. Yeni bilginin State uzayına izdüşümü
            input_morphism = self.B(x_t) # [B_batch, d_state]
            input_morphism = input_morphism.unsqueeze(1).expand(-1, self.d_model, -1) * x_t.unsqueeze(-1)

            # 2. [LUKASIEWICZ STATE UPDATE]
            h = torch.clamp((h * A_exp) + input_morphism, min=0.0, max=1.0)

            # 3. Çıktı Functor'ı (C)
            C_weights = torch.sigmoid(self.C.weight_raw) # [d_model, d_state]

            # State'i C ağırlıkları ile maskele ve topla (Fuzzy Intersection)
            y_t = torch.sum(h * C_weights.unsqueeze(0), dim=-1) # [B_batch, D_model]
            y_t = self.norm(y_t)

            outs.append(y_t.unsqueeze(1))

        # O(N) sürede sentezlenen yeni boyut
        y_final = torch.cat(outs, dim=1) # [B_batch, SeqLen, D_model]

        # Residual Connection ve yeni state (Geleceğe aktarım için)
        return torch.clamp(y_final + x, min=0.0, max=1.0), h

class ToposMambaLM(nn.Module):
    """
    Uçtan uca O(N) Karmaşıklıklı (Attention/KV-Cache İÇERMEYEN),
    Sonsuz bağlam kapasiteli (Infinite Context) Yeni Nesil Dil Modeli.
    """
    def __init__(self, vocab_size, d_model=128, d_state=32, num_layers=4):
        super().__init__()
        from .nn import TopologicalLinear, YonedaEmbedding

        self.yoneda_emb = YonedaEmbedding(vocab_size)
        self.yoneda_proj = TopologicalLinear(vocab_size, d_model, bias=False)

        self.blocks = nn.ModuleList([
            ToposMambaBlock(d_model=d_model, d_state=d_state)
            for _ in range(num_layers)
        ])

        self.norm = TopologicalNorm(d_model)

    def forward(self, idx, states=None):
        B, SeqLen = idx.shape

        # 1. Kelimeleri Kategori Matrisine Çevir
        yoneda_repr = self.yoneda_emb(idx)
        x = self.yoneda_proj(yoneda_repr)

        new_states = []
        # 2. O(N) Topos-Mamba Katmanlarından Geçir (Sonsuz Hafıza Akışı)
        for i, block in enumerate(self.blocks):
            state_i = states[i] if states is not None else None
            x, new_h = block(x, state=state_i)
            new_states.append(new_h)

        x_norm = self.norm(x)

        # 3. Yeniden Kelimeye İzdüşüm (Reachability Projection)
        vocab_embeddings = torch.sigmoid(self.yoneda_proj.weight_raw)

        x_normalized = F.normalize(x_norm, p=2, dim=-1)
        vocab_normalized = F.normalize(vocab_embeddings, p=2, dim=0)

        cosine_sim = torch.matmul(x_normalized, vocab_normalized)

        reachability_logits = (cosine_sim + 1.0) / 2.0
        reachability_logits = torch.clamp(reachability_logits, min=1e-6, max=1.0 - 1e-6)

        return reachability_logits, new_states
