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
        Q = torch.sigmoid(self.q_proj(x)) 
        K = torch.sigmoid(self.k_proj(x)) 
        V = self.v_proj(x)
        
        Q_exp = Q.unsqueeze(2) 
        K_exp = K.unsqueeze(1) 
        
        # Lukasiewicz: K, Q'nun koşullarını ne kadar karşılıyor?
        implication = torch.clamp(1.0 - Q_exp + K_exp, max=1.0)
        attention_truth = implication.mean(dim=-1) 
        
        if apply_causal_mask:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)
            attention_truth = attention_truth.masked_fill(mask == 0, 0.0)
            
        attn_weights = F.softmax(attention_truth * 10.0, dim=-1) 
        output = torch.matmul(attn_weights, V)
        
        return output

class ToposLogicModel(nn.Module):
    def __init__(self, vocab_size, dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 50, dim) * 0.02)
        
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
        
        attn_out = self.topos_attn(x)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        logits = self.fc_out(x)
        return logits

def train_negation_test():
    # Veri seti iki FARKLI MANTIKSAL EVREN (Topos) içeriyor.
    # [T] (True/Olumlu Evren) ve [F] (False/Olumsuz Evren)
    
    sentences = [
        "[T] kuş kanat çırpar .",
        "[T] balık suda yüzer .",
        "[T] aslan et yer .",
        "[F] kuş kanat çırpmaz .",
        "[F] balık suda yüzmez .",
        "[F] aslan et yemez ."
    ]
    
    words = " ".join(sentences).split()
    vocab = list(set(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for i, w in enumerate(vocab)}
    
    # Her cümle 5 kelimeden oluşuyor (Örn: [T] kuş kanat çırpar .)
    seq_length = 4 
    X_train, Y_train = [], []
    
    for sentence in sentences:
        tokens = sentence.split()
        data = [word_to_idx[w] for w in tokens]
        for i in range(len(data) - seq_length):
            X_train.append(data[i : i+seq_length])
            Y_train.append(data[i+1 : i+seq_length+1])
            
    X_train = torch.tensor(X_train)
    Y_train = torch.tensor(Y_train)
    
    print(f"Topos Negasyon Testi Başlıyor | Mantıksal Evrenler Eğitiliyor...")
    
    model = ToposLogicModel(vocab_size=len(vocab), dim=64)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X_train)
        loss = criterion(logits.view(-1, len(vocab)), Y_train.view(-1))
        
        loss.backward()
        optimizer.step()
        
    print(f"Eğitim Tamamlandı. Loss: {loss.item():.4f}")
    
    model.eval()
    def generate(prompt_text):
        tokens = [word_to_idx[w] for w in prompt_text.split()]
        input_tensor = torch.tensor([tokens])
        
        with torch.no_grad():
            logits = model(input_tensor)
            next_token = torch.argmax(logits[0, -1, :]).item()
            return idx_to_word[next_token]

    print("\n--- NEGASYON VE MANTIKSAL EVREN TEST SONUÇLARI ---")
    print("Normal LLM'ler 'kuş kanat' bağlamını gördüğünde kafası karışır. Bizim Topos ise bağlama (evrene) bakar.")
    
    test_cases = [
        "[T] kuş kanat",
        "[F] kuş kanat",
        "[T] balık suda",
        "[F] balık suda",
        "[T] aslan et",
        "[F] aslan et"
    ]
    
    for case in test_cases:
        prediction = generate(case)
        print(f"Evren/Bağlam: '{case}' -> Tahmin: {prediction}")

if __name__ == "__main__":
    train_negation_test()
