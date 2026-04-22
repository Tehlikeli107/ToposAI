import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DifferentiableToposAttention(nn.Module):
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
        
        # Lukasiewicz İmplikası (Mantıksal Kapsayıcılık)
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

def train_syllogism_test():
    # 1. Tam Eğitim (Kalıbı öğrenmesi için)
    train_full = [
        "ali insan . insan fani . ali fani .",
        "can kedi . kedi tatlı . can tatlı .",
        "alp köpek . köpek hızlı . alp hızlı ."
    ]
    
    # 2. Eksik Eğitim (Sadece öncüller var, sonuç YOK)
    # Model "cem vahşi" veya "efe uçucu" kelimelerini yan yana HİÇ görmeyecek.
    train_partial = [
        "cem aslan . aslan vahşi .",
        "efe şahin . şahin uçucu ."
    ]
    
    sentences = train_full + train_partial
    words = " ".join(sentences).split()
    vocab = list(set(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for i, w in enumerate(vocab)}
    
    X_train, Y_train = [], []
    
    # Dil Modeli Eğitimi (Next word prediction)
    for sentence in sentences:
        data = [word_to_idx[w] for w in sentence.split()]
        for i in range(len(data) - 3): # 3 kelimelik pencereler
            X_train.append(data[i : i+3])
            Y_train.append(data[i+1 : i+4])
            
    X_train = torch.tensor(X_train)
    Y_train = torch.tensor(Y_train)
    
    print(f"Topos Syllogism (Mantıksal Geçişlilik) Testi Başlıyor...")
    
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
    def predict_next(prompt_text):
        tokens = [word_to_idx[w] for w in prompt_text.split()]
        input_tensor = torch.tensor([tokens])
        with torch.no_grad():
            logits = model(input_tensor)
            next_token = torch.argmax(logits[0, -1, :]).item()
            return idx_to_word[next_token]

    print("\n--- ZERO-SHOT MANTIKSAL ÇIKARIM TESTİ ---")
    print("Model 'cem vahşi' ilişkisini eğitimde ASLA görmedi. Bakalım mantığı kurabilecek mi?")
    
    test_cases = [
        "cem aslan . aslan vahşi . cem",
        "efe şahin . şahin uçucu . efe"
    ]
    
    for case in test_cases:
        prediction = predict_next(case)
        print(f"Bağlam: '{case}' -> Tahmin Edilen Özellik: {prediction}")

if __name__ == "__main__":
    train_syllogism_test()
