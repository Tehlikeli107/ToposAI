import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =====================================================================
# YONEDA LEMMA TABANLI SIFIR-VEKTÖR (ZERO-EMBEDDING) DİL MODELİ
# Kategori Teorisinin kalbi: Bir nesne, diğerleriyle olan ilişkilerinin bütünüdür.
# X ≅ Hom(-, X)
# =====================================================================

class YonedaEmbedding(nn.Module):
    """
    Kelimelerin sabit bir vektörü (nn.Embedding) YOKTUR.
    Bunun yerine kelimeler arası mantıksal ilişkileri (Morphisms) tutan bir matris vardır.
    Bir kelimenin "Anlamı" (Representation), o kelimenin tüm evrendeki (Vocabulary) 
    diğer kelimelerle olan ilişki oklarının (Hom-set) bütünüdür.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Kelimeler arası ilişki matrisi: Hom(A, B)
        # Satır i'den Sütun j'ye giden okun (mantıksal bağın) gücü.
        self.morphisms_logits = nn.Parameter(torch.randn(vocab_size, vocab_size))

    def get_morphisms(self):
        # Oklar [0, 1] arası mantıksal doğruluk değerleridir
        return torch.sigmoid(self.morphisms_logits)

    def forward(self, idx):
        # idx: [Batch, SeqLen]
        # Normalde Embedding katmanı [Batch, SeqLen, Dim] döndürür.
        # Yoneda Lemma'ya göre bir kelimenin Embedding'i (Anlamı), 
        # o kelimenin Vocabulary'deki diğer TÜM kelimelere olan ilişkileridir!
        # Yani boyutumuz (Dim), Vocab Size'ın ta kendisidir.
        
        R = self.get_morphisms() # [Vocab, Vocab]
        
        # Yoneda Representation: Seçilen kelimelerin ilişki vektörlerini (oklarını) çek
        # Çıktı: [Batch, SeqLen, VocabSize]
        yoneda_repr = F.embedding(idx, R) 
        
        return yoneda_repr

class YonedaLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 1. Standart nn.Embedding YERİNE Yoneda Lemma kullanıyoruz!
        self.yoneda = YonedaEmbedding(vocab_size)
        
        # Boyut (Dim) = Vocab Size oldu, çünkü anlam = diğer kelimelere olan oklar.
        dim = vocab_size 
        
        # Sadece bağlamı (context) harmanlayan ufak bir doğrusal katman
        # Dikkat: Parametre sayımız devasa embedding matrisleri olmadığı için inanılmaz küçük!
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, vocab_size)
        )

    def forward(self, idx):
        # 1. Kelimenin "İçsel" bir vektörü yok. Sadece ilişkiler ağı (Yoneda) var.
        # x: [Batch, SeqLen, VocabSize]
        x = self.yoneda(idx)
        
        # 2. Cümledeki kelimelerin ilişkilerini zaman (Sequence) boyunca topla (Basit bir Attention/Bag of Words)
        # Sadece önceki kelimelerin ilişkilerini topluyoruz (Causal)
        B, SeqLen, D = x.shape
        causal_mask = torch.tril(torch.ones(SeqLen, SeqLen, device=x.device)).unsqueeze(0) # [1, SeqLen, SeqLen]
        
        # Her adımda, o adıma kadarki kelimelerin Yoneda anlamlarının (ilişkilerinin) ortalaması
        # matmul ile broadcasting: [1, SeqLen, SeqLen] x [B, SeqLen, D] -> [B, SeqLen, D]
        context = torch.matmul(causal_mask, x) / causal_mask.sum(dim=-1, keepdim=True) # [B, SeqLen, D]
        
        # 3. Sonraki kelimeyi tahmin et (Sıradaki kelime hangi ilişki ağına sahip olmalı?)
        logits = self.ffn(context)
        return logits

def train_yoneda_model():
    # Mini Veri Seti (Varlıklar ve İlişkileri)
    dataset = [
        "kedi fare yakalar .",
        "kedi süt içer .",
        "köpek kemik yer .",
        "köpek kedi kovalar .",
        "fare peynir yer ."
    ]
    
    text = " ".join(dataset)
    words = text.split()
    vocab = list(set(words))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    
    # N-gram pencereleri (Önceki 3 kelimeye bakarak sıradakini tahmin et)
    seq_length = 3
    X_train, Y_train = [], []
    data = [word2idx[w] for w in words]
    for i in range(len(data) - seq_length):
        X_train.append(data[i : i+seq_length])
        Y_train.append(data[i+1 : i+seq_length+1])
        
    X_train = torch.tensor(X_train)
    Y_train = torch.tensor(Y_train)
    
    print("--- YONEDA LEMMA (ZERO-EMBEDDING) DİL MODELİ EĞİTİMİ ---")
    print(f"Kelimelerin içsel vektörleri YOK! Sadece {len(vocab)}x{len(vocab)} boyutunda bir 'Oklar (Morphism) Matrisi' öğrenilecek.\n")
    
    model = YonedaLanguageModel(vocab_size=len(vocab))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, 401):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X_train)
        loss = criterion(logits.view(-1, len(vocab)), Y_train.view(-1))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    print("\nEğitim Tamamlandı! Şimdi modelin içini (Kategori Teorisini) açıp inceliyoruz:")
    model.eval()
    
    with torch.no_grad():
        R = model.yoneda.get_morphisms() # [Vocab, Vocab] İlişki (Oklar) Matrisi
        
        print("\n[YONEDA EMBEDDING ANALİZİ]")
        print("Bir kelimenin anlamı, diğer kelimelerle olan en güçlü bağlarıdır (Hom-Set):")
        
        test_words = ["kedi", "köpek", "fare"]
        for w in test_words:
            idx = word2idx[w]
            yoneda_vector = R[idx] # Kedi'nin diğer tüm kelimelerle olan ilişkisi
            
            # En güçlü 3 bağı (Oku) bul
            top_vals, top_indices = torch.topk(yoneda_vector, 3)
            
            relations = [f"{idx2word[i.item()]} ({val.item():.2f})" for val, i in zip(top_vals, top_indices)]
            print(f"  '{w.upper()}' kelimesinin Yoneda Anlamı (En Güçlü Okları) -> {', '.join(relations)}")

    print("\n[CÜMLE ÜRETİMİ (GENERATION)]")
    def generate(prompt, max_tokens=2):
        tokens = [word2idx[w] for w in prompt.split()]
        idx = torch.tensor([tokens])
        
        for _ in range(max_tokens):
            with torch.no_grad():
                logits = model(idx)
                next_token = torch.argmax(logits[0, -1, :]).item()
                idx = torch.cat([idx, torch.tensor([[next_token]])], dim=1)
                if idx2word[next_token] == ".":
                    break
        return " ".join([idx2word[t.item()] for t in idx[0]])

    print("İstem: 'kedi süt' -> Çıktı:", generate("kedi süt"))
    print("İstem: 'köpek kemik' -> Çıktı:", generate("köpek kemik"))
    print("İstem: 'köpek kedi' -> Çıktı:", generate("köpek kedi"))

if __name__ == "__main__":
    train_yoneda_model()
