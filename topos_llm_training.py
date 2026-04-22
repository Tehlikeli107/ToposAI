import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DifferentiableToposAttention(nn.Module):
    """Lukasiewicz MV-Cebiri tabanlı Türevlenebilir Topos Attention"""
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x, apply_causal_mask=True):
        batch, seq_len, dim = x.shape
        
        # Topos Doğruluk Değerleri (0.0 ile 1.0 arası)
        Q = torch.sigmoid(self.q_proj(x)) 
        K = torch.sigmoid(self.k_proj(x)) 
        V = self.v_proj(x)
        
        Q_exp = Q.unsqueeze(2) # (batch, seq_q, 1, dim)
        K_exp = K.unsqueeze(1) # (batch, 1, seq_k, dim)
        
        # Lukasiewicz İmplikası (Q => K): min(1.0, 1.0 - Q + K)
        # Eğer K, Q'nun gereksinimlerini karşılıyorsa (K >= Q), doğruluk 1.0 olur.
        # Değilse kısmi doğruluk üretir. Geriye yayılım (backprop) için mükemmeldir.
        implication = torch.clamp(1.0 - Q_exp + K_exp, max=1.0)
        
        # Conjunction (Tüm özelliklerin mantıksal birleşimi)
        attention_truth = implication.mean(dim=-1) 
        
        if apply_causal_mask:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)
            # Mantıksal olarak imkansız olan durumlara 0.0 (False) veriyoruz
            attention_truth = attention_truth.masked_fill(mask == 0, 0.0)
            
        attn_weights = F.softmax(attention_truth * 5.0, dim=-1) 
        output = torch.matmul(attn_weights, V)
        
        return output

class ToposLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 50, dim) * 0.01) # Max seq_len 50
        
        self.topos_attn = DifferentiableToposAttention(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, vocab_size)

    def forward(self, idx):
        seq_len = idx.size(1)
        x = self.embedding(idx) + self.pos_encoding[:, :seq_len, :]
        
        # Topos Attention Bloğu
        attn_out = self.topos_attn(x)
        x = self.norm1(x + attn_out)
        
        # FFN Bloğu
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        logits = self.fc_out(x)
        return logits

def train_and_generate():
    # Mini bir corpus (dilbilgisi kuralları içerir)
    text = "ali topu at . ayşe ip atla . kedi süt iç . köpek kemik ye . ali ip atla . ayşe topu at . kedi kemik ye . köpek süt iç ."
    words = text.split()
    vocab = list(set(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for i, w in enumerate(vocab)}
    
    # Eğitim verisini hazırla (Cümleleri kaydırarak X ve Y oluştur)
    data = [word_to_idx[w] for w in words]
    seq_length = 4
    
    X_train, Y_train = [], []
    for i in range(len(data) - seq_length):
        X_train.append(data[i : i+seq_length])
        Y_train.append(data[i+1 : i+seq_length+1])
        
    X_train = torch.tensor(X_train)
    Y_train = torch.tensor(Y_train)
    
    print(f"Topos-LLM Eğitimi Başlıyor | Vocab Size: {len(vocab)} | Toplam Veri: {len(X_train)}")
    
    model = ToposLanguageModel(vocab_size=len(vocab), dim=32)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    # Eğitim Döngüsü
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X_train)
        loss = criterion(logits.view(-1, len(vocab)), Y_train.view(-1))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 40 == 0:
            print(f"Epoch {epoch:03d} | Loss (Hata): {loss.item():.4f}")
            
    print("\nEğitim Tamamlandı. Şimdi Topos-LLM'in Text Üretmesini Test Ediyoruz:")
    model.eval()
    
    # Jeneratör (Üretim) Fonksiyonu
    def generate(prompt_text, max_len=5):
        tokens = [word_to_idx[w] for w in prompt_text.split()]
        input_tensor = torch.tensor([tokens])
        
        for _ in range(max_len):
            with torch.no_grad():
                logits = model(input_tensor)
                # Sadece son token'ın tahmini alınır
                next_token_logits = logits[0, -1, :] 
                next_token = torch.argmax(next_token_logits).item()
                
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]])], dim=1)
                
                if idx_to_word[next_token] == ".": # Nokta gelirse dur
                    break
                    
        return " ".join([idx_to_word[t.item()] for t in input_tensor[0]])

    print("İstem: 'kedi süt' -> Çıktı:", generate("kedi süt"))
    print("İstem: 'köpek kemik' -> Çıktı:", generate("köpek kemik"))
    print("İstem: 'ayşe ip' -> Çıktı:", generate("ayşe ip"))
    print("İstem: 'ali topu' -> Çıktı:", generate("ali topu"))

if __name__ == "__main__":
    train_and_generate()
