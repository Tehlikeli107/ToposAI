import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =====================================================================
# 1. MULTI-UNIVERSE TOPOS ATTENTION (MUTA)
# GPT'nin Multi-Head Attention'ı yerine geçer.
# İstatistiksel Dot-Product yerine Lukasiewicz Mantık Evrenleri kullanır.
# =====================================================================
class MultiUniverseToposAttention(nn.Module):
    def __init__(self, d_model, num_universes):
        super().__init__()
        self.num_universes = num_universes
        self.d_universe = d_model // num_universes
        assert d_model % num_universes == 0, "d_model, num_universes'a tam bölünmeli"

        # Mantıksal Önermeler (Q) ve Gerçekler (K)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, SeqLen, D = x.shape

        # Topos Doğruluk Değerleri [0, 1] (Sigmoid ile)
        Q = torch.sigmoid(self.q_proj(x))
        K = torch.sigmoid(self.k_proj(x))
        V = self.v_proj(x)

        # Evrenlere (Universes) Bölme İşlemi
        # Şekil: [Batch, Universes, SeqLen, D_universe]
        Q = Q.view(B, SeqLen, self.num_universes, self.d_universe).transpose(1, 2).contiguous()
        K = K.view(B, SeqLen, self.num_universes, self.d_universe).transpose(1, 2).contiguous()
        V = V.view(B, SeqLen, self.num_universes, self.d_universe).transpose(1, 2).contiguous()

        # Q => K Mantıksal Çıkarımı (Lukasiewicz: min(1, 1 - Q + K))
        # Q: [B, U, SeqLen, D_u] -> [B, U, SeqLen, 1, D_u]
        # K: [B, U, SeqLen, D_u] -> [B, U, 1, SeqLen, D_u]
        Q_exp = Q.unsqueeze(3) 
        K_exp = K.unsqueeze(2) 
        
        implication = torch.clamp(1.0 - Q_exp + K_exp, max=1.0)
        
        # Conjunction (D_u boyutu üzerinden ortalama al) -> Çıktı: [B, U, SeqLen, SeqLen]
        truth_matrix = implication.mean(dim=-1)

        # Causal Mask (Geleceği Görme Engeli)
        local_mask = torch.tril(torch.ones(SeqLen, SeqLen, device=x.device)).view(1, 1, SeqLen, SeqLen)
        truth_matrix = truth_matrix.masked_fill(local_mask == 0, -1e9)

        # Softmax ile Doğrulukları Normalize Et
        attn_weights = F.softmax(truth_matrix * 5.0, dim=-1)

        # Değerleri (V) topla
        # attn_weights: [B, U, SeqLen, SeqLen]
        # V: [B, U, SeqLen, D_u]
        out = torch.matmul(attn_weights, V) # [B, U, SeqLen, D_u]
        
        # Evrenleri tekrar tek bir boyutta birleştir (Concatenate)
        out = out.transpose(1, 2).contiguous().view(B, SeqLen, D)
        return self.out_proj(out)


# =====================================================================
# 2. TOPOS TRANSFORMER BLOĞU VE MODELİ
# =====================================================================
class ToposTransformerBlock(nn.Module):
    def __init__(self, d_model, num_universes):
        super().__init__()
        self.muta = MultiUniverseToposAttention(d_model, num_universes)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.muta(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class ToposTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_universes=4, num_layers=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(100, d_model) # Max 100 token
        
        self.blocks = nn.ModuleList([
            ToposTransformerBlock(d_model, num_universes) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, SeqLen = idx.shape
        pos = torch.arange(0, SeqLen, device=idx.device).unsqueeze(0)
        
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        # Geleceği görmeyi engelleyen Causal Mask (Alt üçgen matris)
        mask = torch.tril(torch.ones(SeqLen, SeqLen, device=idx.device)).view(1, 1, SeqLen, SeqLen)
        
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.norm(x)
        return self.fc_out(x)


# =====================================================================
# 3. EĞİTİM VE TEST DÖNGÜSÜ
# =====================================================================
def train_topos_transformer():
    # Mini Mantık ve Neden-Sonuç Veri Seti
    dataset = [
        "eğer yağmur yağarsa yer ıslak olur . yağmur yağdı . o halde yer ıslak .",
        "eğer güneş açarsa hava sıcak olur . güneş açtı . o halde hava sıcak .",
        "eğer ateş yanarsa su kaynar . ateş yandı . o halde su kaynar .",
        "eğer rüzgar eserse yaprak kımıldar . rüzgar esti . o halde yaprak kımıldar ."
    ]
    
    text = " ".join(dataset)
    words = text.split()
    vocab = list(set(words))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    
    # Veri hazırlığı (X: Context, Y: Next Token)
    seq_length = 6
    X_train, Y_train = [], []
    data = [word2idx[w] for w in words]
    for i in range(len(data) - seq_length):
        X_train.append(data[i : i+seq_length])
        Y_train.append(data[i+1 : i+seq_length+1])
        
    X_train = torch.tensor(X_train)
    Y_train = torch.tensor(Y_train)
    
    print(f"Topos-Transformer Eğitiliyor... | Vocab: {len(vocab)} | Evren Sayısı (Universes): 4 | Katman: 2")
    
    model = ToposTransformer(vocab_size=len(vocab), d_model=64, num_universes=4, num_layers=2)
    optimizer = optim.AdamW(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X_train)
        loss = criterion(logits.view(-1, len(vocab)), Y_train.view(-1))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    print("\nEğitim Tamamlandı! Modelin Mantıksal Üretim (Generation) Testi:")
    model.eval()
    
    def generate(prompt, max_new_tokens=4):
        tokens = [word2idx[w] for w in prompt.split()]
        idx = torch.tensor([tokens])
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = model(idx)
                next_token = torch.argmax(logits[0, -1, :]).item()
                idx = torch.cat([idx, torch.tensor([[next_token]])], dim=1)
                if idx2word[next_token] == ".":
                    break
        return " ".join([idx2word[t.item()] for t in idx[0]])

    print("\n[PROMPT 1] İstem: 'yağmur yağdı . o halde'")
    print("Topos Üretimi:", generate("yağmur yağdı . o halde", 3))
    
    print("\n[PROMPT 2] İstem: 'güneş açtı . o halde'")
    print("Topos Üretimi:", generate("güneş açtı . o halde", 3))

    print("\n[PROMPT 3] İstem: 'ateş yandı . o halde'")
    print("Topos Üretimi:", generate("ateş yandı . o halde", 3))

if __name__ == "__main__":
    train_topos_transformer()
