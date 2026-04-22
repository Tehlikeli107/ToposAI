import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import time
import os
import argparse

# =====================================================================
# TOPOS-TRANSFORMER PRETRAINING ENGINE (UÇTAN UCA EĞİTİM BORU HATTI)
# Modelimizi oyuncak (sentetik) cümlelerle değil, gerçek metin veri
# kümeleri (örn: TinyStories, Shakespeare) ile otoregresif olarak eğitir.
# Loss değerlerini kaydeder ve periyodik "Checkpoint" (.pt) alır.
# =====================================================================

from topos_ai.models import ToposTransformer

class SimpleTextDataset(Dataset):
    """Metin verisini alıp PyTorch veri kümesine çeviren basit yapıcı."""
    def __init__(self, text, seq_len):
        # Basit char-level (karakter bazlı) veya kelime bazlı tokenizer simülasyonu
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        self.data = [self.stoi[c] for c in text]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train_topos_model(dataset_text, seq_len=64, batch_size=32, max_iters=2000, lr=1e-3, device='cuda'):
    print("=========================================================================")
    print(" 🚀 TOPOS-TRANSFORMER PRETRAINING ENGINE BAŞLATILIYOR")
    print("=========================================================================\n")
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("Uyarı: CUDA bulunamadı. Eğitim CPU üzerinde yapılacak (Yavaş olabilir).")
        device = 'cpu'
        
    # Veri Kümesi ve Loader
    dataset = SimpleTextDataset(dataset_text, seq_len)
    vocab_size = dataset.vocab_size
    print(f"[VERİ] Toplam Karakter: {len(dataset.data)}, Sözlük Boyutu (Vocab): {vocab_size}")
    
    # DataLoader (Sürekli random batch çekebilmek için iterator)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    def get_batch():
        while True:
            for X, Y in dataloader:
                yield X.to(device), Y.to(device)
    batch_iter = get_batch()

    # Modeli Kur (Hafif bir model: 4 Evren, 4 Katman)
    model = ToposTransformer(vocab_size=vocab_size, d_model=128, num_universes=4, num_layers=4)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] ToposTransformer Yüklendi. Toplam Parametre: {total_params / 1e6:.2f} Milyon")

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)
    print("\n[EĞİTİM] Döngü Başlıyor...\n")

    t0 = time.time()
    for iter_num in range(1, max_iters + 1):
        # Forward Pass
        X, Y = next(batch_iter)
        logits = model(X) # [B, T, Vocab]
        
        # Loss Hesapla
        loss = criterion(logits.view(-1, vocab_size), Y.view(-1))
        
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Loglama
        if iter_num % 100 == 0 or iter_num == 1:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            # Örnek metin üret (Model ne öğreniyor?)
            model.eval()
            with torch.no_grad():
                # Rastgele bir harften başla
                sample_idx = torch.tensor([[dataset.stoi['T']]], device=device)
                for _ in range(20): # 20 karakter üret
                    out = model(sample_idx)
                    next_token = torch.argmax(out[0, -1, :]).item()
                    sample_idx = torch.cat([sample_idx, torch.tensor([[next_token]], device=device)], dim=1)
                
                generated_text = "".join([dataset.itos[i.item()] for i in sample_idx[0]])
            model.train()
            
            print(f"Iter {iter_num:<5} | Loss: {loss.item():.4f} | Süre: {dt:.2f}s | Üretim: '{generated_text.replace(chr(10), ' ')}'")

        # Checkpoint Kaydet
        if iter_num % 500 == 0:
            checkpoint_path = f"checkpoints/topos_model_iter_{iter_num}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [>] Model Ağırlıkları Kaydedildi: {checkpoint_path}")

    print("\n🎉 Eğitim Başarıyla Tamamlandı! Nihai Model 'checkpoints/' klasöründedir.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Topos-Transformer Pretrainer CLI')
    parser.add_argument('--text', type=str, default="The quick brown fox jumps over the lazy dog. " * 100, help='Eğitim için kullanılacak ham metin.')
    parser.add_argument('--iters', type=int, default=500, help='Toplam iterasyon sayısı.')
    parser.add_argument('--batch', type=int, default=16, help='Batch Size.')
    parser.add_argument('--seq', type=int, default=32, help='Sequence Length.')
    
    args = parser.parse_args()
    
    # Eğer varsayılan argüman kullanıldıysa, biraz daha anlamlı bir metin indirelim (Shakespeare simülasyonu)
    train_text = args.text
    if len(train_text) < 1000:
        import urllib.request
        try:
            print("İnternetten örnek metin (Tiny Shakespeare) indiriliyor...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            train_text = urllib.request.urlopen(url).read().decode('utf-8')[:50000] # İlk 50K karakter
        except:
            print("İndirme başarısız. Gömülü metinle devam ediliyor.")
            
    train_topos_model(train_text, seq_len=args.seq, batch_size=args.batch, max_iters=args.iters)
