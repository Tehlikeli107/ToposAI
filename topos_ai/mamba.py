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
        
    def forward(self, x):
        """
        x: [Batch, SeqLen, D_model]
        """
        B_batch, SeqLen, D = x.shape
        
        # Topolojik Hafıza (Başlangıçta Sıfır/Boşluk)
        # [B_batch, D_model, d_state]
        h = torch.zeros(B_batch, self.d_model, self.d_state, device=x.device)
        
        outs = []
        
        # A matrisini [0, 1] arasına çek (Kategori Uzayı Sınırları)
        # A_sig = 1.0 demek hafızayı %100 koru, 0.0 demek geçmişi tamamen sil
        A_sig = torch.sigmoid(self.A) # [D_model, d_state]
        A_exp = A_sig.unsqueeze(0)    # [1, D_model, d_state]
        
        # [O(N) TOPOLOGICAL SCAN - LINEAR TIME]
        # Gerçek bir CUDA Mamba implementasyonunda bu döngü Parallel Associative Scan
        # ile donanımda tek seferde yapılır. Biz burada kavramsal (Simülasyon)
        # O(N) State geçişini (RNN gibi) yazıyoruz.
        
        for t in range(SeqLen):
            x_t = x[:, t, :] # [B_batch, D_model]
            
            # 1. Yeni bilginin State uzayına izdüşümü
            # x_t_exp: [B_batch, 1, D_model] x B.weight: [D_state, D_model] -> [B_batch, D_model, D_state]
            input_morphism = self.B(x_t) # [B_batch, d_state]
            input_morphism = input_morphism.unsqueeze(1).expand(-1, self.d_model, -1) * x_t.unsqueeze(-1)
            
            # 2. [LUKASIEWICZ STATE UPDATE]
            # h(t) = h(t-1) * A + B * x(t) (Klasik SSM)
            # ToposAI SSM: Geçmişi (h) A functor'ı ile zayıflat ve yeni bilgiyi (input_morphism) ekle.
            # 1.0 sınırını aşmaması için clamp (Kategori mantığı).
            h = torch.clamp((h * A_exp) + input_morphism, min=0.0, max=1.0)
            
            # 3. Çıktı Functor'ı (C)
            # h matrisini C ile çarparak orijinal d_model boyutuna geri indir
            # h: [B_batch, D_model, d_state] -> Output: [B_batch, D_model]
            
            # C'nin pozitif ağırlıkları (Topolojik)
            C_weights = torch.sigmoid(self.C.weight_raw) # [d_model, d_state]
            
            # State'i C ağırlıkları ile maskele ve topla (Fuzzy Intersection)
            y_t = torch.sum(h * C_weights.unsqueeze(0), dim=-1) # [B_batch, D_model]
            
            # Değerleri normalize et
            y_t = self.norm(y_t)
            
            outs.append(y_t.unsqueeze(1))
            
        # O(N) sürede sentezlenen yeni boyut
        y_final = torch.cat(outs, dim=1) # [B_batch, SeqLen, D_model]
        
        # Residual Connection (Topolojik Bütünlük)
        return torch.clamp(y_final + x, min=0.0, max=1.0)

class ToposMambaLM(nn.Module):
    """
    Uçtan uca O(N) Karmaşıklıklı (Attention/KV-Cache İÇERMEYEN),
    Sonsuz bağlam kapasiteli (Infinite Context) Yeni Nesil Dil Modeli.
    """
    def __init__(self, vocab_size, d_model=128, d_state=32, num_layers=4):
        super().__init__()
        from .nn import YonedaEmbedding, TopologicalLinear
        
        self.yoneda_emb = YonedaEmbedding(vocab_size)
        self.yoneda_proj = TopologicalLinear(vocab_size, d_model, bias=False)
        
        self.blocks = nn.ModuleList([
            ToposMambaBlock(d_model=d_model, d_state=d_state) 
            for _ in range(num_layers)
        ])
        
        self.norm = TopologicalNorm(d_model)
        
    def forward(self, idx):
        B, SeqLen = idx.shape
        
        # 1. Kelimeleri Kategori Matrisine Çevir
        yoneda_repr = self.yoneda_emb(idx)
        x = self.yoneda_proj(yoneda_repr)
        
        # 2. O(N) Topos-Mamba Katmanlarından Geçir (Dikkat mekanizması YOKTUR!)
        for block in self.blocks:
            x = block(x)
            
        x_norm = self.norm(x)
        
        # 3. Yeniden Kelimeye İzdüşüm (Reachability Projection)
        vocab_embeddings = torch.sigmoid(self.yoneda_proj.weight_raw)
        
        x_normalized = F.normalize(x_norm, p=2, dim=-1)
        vocab_normalized = F.normalize(vocab_embeddings, p=2, dim=0)
        
        cosine_sim = torch.matmul(x_normalized, vocab_normalized) 
        
        reachability_logits = (cosine_sim + 1.0) / 2.0
        reachability_logits = torch.clamp(reachability_logits, min=1e-6, max=1.0 - 1e-6)
        
        # Sadece O(N) logit döner. KV-Cache'e İHTİYAÇ YOKTUR!
        # Çünkü Mamba'da her şey 'State'in içindedir.
        return reachability_logits
