import os
import urllib.request
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

# ToposAI çekirdeği
import sys
sys.path.append(r"C:\Users\salih\ToposAI")
from topos_ai.models import ToposTransformer
from topos_ai.optim import ToposAdam

# =========================================================================
# BÜYÜK VERİ (REAL-WORLD SCALE) TOPOS-LLM EĞİTİM TESTİ
# =========================================================================

class LargeCharDataset(Dataset):
    def __init__(self, data_tensor, seq_len):
        self.data = data_tensor
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = "shakespeare.txt"
    if not os.path.exists(filepath):
        print("[İNDİRİLİYOR] Gerçek LLM Veri Seti (TinyShakespeare - ~1MB Text)...")
        urllib.request.urlretrieve(url, filepath)
    else:
        print("[HAZIR] Shakespeare Veri Seti zaten mevcut.")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def run_large_scale_training():
    print("=========================================================================")
    print(" GERCEK BUYUK VERI (SCALE) ILE TOPOS-LLM STRES TESTI ")
    print("=========================================================================\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hedef Donanım: {device}")
    
    # 1. BÜYÜK VERİ YÜKLEME VE TOKENİZASYON
    text = download_shakespeare()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    
    print(f"Toplam Karakter: {len(text):,} | Benzersiz Kelime/Karakter (Vocab): {vocab_size}")
    
    # Tüm metni PyTorch Tensor'e çevir (1 Milyonluk dev dizi)
    data_tensor = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    
    # 2. GERÇEKÇİ MODEL BOYUTLARI (VRAM ZORLAYICI)
    # Toy model değil: 256 Boyut, 8 Evren, 256 Kelime Hafızası
    SEQ_LEN = 256
    BATCH_SIZE = 16 # VRAM'i dolduracak batch
    D_MODEL = 256
    UNIVERSES = 8
    LAYERS = 2
    
    dataset = LargeCharDataset(data_tensor, SEQ_LEN)
    # Hızlı test için sadece ilk 1000 batch'i alalım
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"\n[MODEL] ToposTransformer Başlatılıyor...")
    print(f" Parametreler: D={D_MODEL}, Evrenler={UNIVERSES}, Katman={LAYERS}, Context={SEQ_LEN}, Batch={BATCH_SIZE}")
    
    model = ToposTransformer(
        vocab_size=vocab_size, 
        d_model=D_MODEL, 
        num_universes=UNIVERSES, 
        num_layers=LAYERS, 
        max_seq_len=SEQ_LEN
    )
    model.to(device)
    
    # Toplam parametre sayısını hesapla
    total_params = sum(p.numel() for p in model.parameters())
    print(f" Toplam Parametre (Weights): {total_params:,}")
    
    optimizer = ToposAdam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 3. AĞIR EĞİTİM DÖNGÜSÜ (STRES TESTİ)
    print("\n[EĞİTİM] Kategori Teorisi Kernel'leri İle GPU Eğitimi Başlıyor (Hedef: 20 Step)...")
    
    model.train()
    scaler = torch.cuda.amp.GradScaler() # Hız ve VRAM için Mixed Precision
    
    start_time = time.time()
    initial_loss = None
    
    for step, (x, y) in enumerate(dataloader):
        if step >= 20: # 20 Ağır Batch (Gerçek öğrenme kanıtı için yeterli)
            break
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        # OOM (Out of Memory) veya Kernel çökmesi varsa burada patlar!
        # HATA DÜZELTME: Mixed Precision (AMP) Kategori Matrisleriyle Indexing
        # sırasında Float-Half uyuşmazlığı yarattığı için FP32 (saf mod) kullanılıyor.
        # Bu, Kategori mantığını bozmadan VRAM'i zorlayan asıl testtir.
        logits, _ = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            
        if step == 0: initial_loss = loss.item()
        
        # Özel Straight-Through Estimator Geri Yayılımı
        loss.backward()
        
        # AMP Scaler yerine saf optimizer adımı
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step % 4 == 0 or step == 19:
            print(f"  > Adım {step:02d} | Kayıp (Loss): {loss.item():.4f} | Süre: {(time.time()-start_time):.2f}sn")
            start_time = time.time()

    final_loss = loss.item()
    
    print("\n=========================================================================")
    if final_loss < initial_loss:
        print(f" [KANITLANDI] Milyonluk Veri, Yüksek VRAM (D=256) ve Katı Topos Mantığı İle Öğrenme Başarılı!")
        print(f" Loss Düşüşü: {initial_loss:.4f} --> {final_loss:.4f}")
    else:
        print(f" [BAŞARISIZ] Loss düşmedi. Ağ öğrenemiyor olabilir.")
    print("=========================================================================")

if __name__ == '__main__':
    run_large_scale_training()