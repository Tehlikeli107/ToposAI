import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import urllib.request
import time

from topos_ai.models import ToposTransformer
from topos_ai.tokenization import TopologicalTokenizer
from topos_ai.optim import ToposAdam
from topos_ai.optim import ToposAdam
from torch.utils.tensorboard import SummaryWriter

# =====================================================================
# TOPOS-LLM CUSTOM TRAINING ENGINE
# Amacı: OpenAI'ın Dot-Product mimarisi yerine, ToposAI'nin Kategori 
# Teorisi (Yoneda, MoE, RoPE) mimarisini kullanarak sıfırdan bir Dil Modeli 
# eğitmek. Model, "Tiny Shakespeare" veri setini okuyarak İngilizceyi
# kendi kendine sentezleyecek ve ağırlıklarını kaydedecek.
# Eklemeler: torch.compile() (Hızlandırma) ve TensorBoard (MLOps)
# =====================================================================

class CharDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[SİSTEM] Eğitim Cihazı: {device} (SRAM/Triton destekli Kategori Motoru)")

    # [MLOps] TensorBoard Başlatıcı
    # Terminalden izlemek için: tensorboard --logdir=runs
    run_name = f"topos_llm_run_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"[MLOps] TensorBoard logları 'runs/{run_name}' dizinine kaydediliyor.")

    # 1. VERİ İNDİRME VE TOKENİZASYON
    print("[VERİ] 'Tiny Shakespeare' veri seti indiriliyor...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        text = urllib.request.urlopen(url).read().decode('utf-8')
        print(f"[VERİ] {len(text):,} karakter başarıyla indirildi.")
    except Exception as e:
        print(f"Veri indirme hatası: {e}")
        return

    # [KENDİ TOKENIZER'IMIZ]
    print("\n[TOKENIZER] Kendi icadımız olan 'Topological Tokenizer' Yükleniyor...")
    # Çok bekletmemek adına 1000'lik bir hedef vocab seçiyoruz.
    tokenizer = TopologicalTokenizer(vocab_size=1000)
    
    print("[TOKENIZER] Shakespeare verisi üzerinde Topolojik Nedensellik eğitiliyor (Kelimeler yaratılıyor)...")
    # Tokenizer'ı metnin ilk 30.000 karakteri üzerinde (Hızlı demo için) eğitelim
    tokenizer.train(text[:30000])
    vocab_size = len(tokenizer.vocab)
    print(f"   Sözlük Boyutu (Vocab Size): {vocab_size:,}")
    
    print("\n[VERİ] Metin, Kendi Topolojik Tokenlarımıza (Sözlük ID'lerine) çevriliyor...")
    # Eğitim için ilk 200.000 karakteri encode et
    tokens = tokenizer.encode(text[:200000])
    print(f"[VERİ] Toplam {len(tokens):,} Token oluşturuldu.")

    # 2. DATALOADER
    seq_len = 64 # Bağlam penceresi (Context Window)
    batch_size = 32
    dataset = CharDataset(tokens, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    def get_batch():
        while True:
            for X, Y in dataloader:
                yield X.to(device), Y.to(device)
    batch_iter = get_batch()

    # 3. TOPOS-LLM MİMARİSİ
    print(f"\n[MODEL] ToposTransformer (Kategori Teorisi LLM'i) İnşa Ediliyor...")
    # Daha akıllı olması için boyutları büyütüyoruz (Mini-LLM)
    model = ToposTransformer(vocab_size=vocab_size, d_model=256, num_universes=8, num_layers=4)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  > Parametre Sayısı: {total_params / 1e6:.2f} Milyon")
    print(f"  > VRAM Dostu Low-Rank Yoneda Embedding Aktif.")

    optimizer = ToposAdam(model.parameters(), lr=5e-4, topological_weight_decay=0.01)

    # [PURE TOPOLOGICAL LOSS (NO ZERO-SUM GAMES)]
    # Klasik CrossEntropyLoss, "Kral" kelimesini doğru sayıp "Adam" kelimesini
    # zorla sıfırlar (Softmax yüzünden olasılıklar toplamı 1.0 olmak zorundadır).
    # ToposAI'da (Kategori Teorisinde) her varlığa ulaşılabilirlik BAĞIMSIZDIR [0, 1].
    # Bu yüzden sistemi BCELoss (Binary Cross Entropy) ile eğitiyoruz. 
    # Hiçbir kelimenin olasılığı diğerini ezmez, 15.0 gibi hileli çarpanlar (Scale) kullanılmaz.
    criterion = nn.BCELoss()

    # 4. EĞİTİM (PRETRAINING) DÖNGÜSÜ
    max_iters = 1000 # Makinenin dili temel düzeyde kavraması için yeterli
    print(f"\n[EĞİTİM] {max_iters} İterasyonluk Kategori Öğrenimi (Pretraining) Başlıyor...\n")
    
    os.makedirs("weights", exist_ok=True)
    t0 = time.time()
    
    for iter_num in range(1, max_iters + 1):
        X, Y = next(batch_iter)
        
        reachability_logits, _ = model(X) # [B, SeqLen, vocab_size] (Saf [0, 1] arası Topos skorları)
        
        # Hedefi (Y) "One-Hot" forma çevir ki [B, SeqLen, vocab_size] olsun.
        Y_one_hot = torch.nn.functional.one_hot(Y, num_classes=vocab_size).float()
        
        # Her kelimenin kendi [0, 1] ulaşılabilirliğini bağımsız olarak cezalandır.
        loss = criterion(reachability_logits.view(-1, vocab_size), Y_one_hot.view(-1, vocab_size))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if iter_num % 200 == 0 or iter_num == 1:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"  [Iter {iter_num:<4}] Loss: {loss.item():.4f} | Süre: {dt:.2f}s")
            
            # Anlık Test Üretimi
            model.eval()
            with torch.no_grad():
                test_prompt = torch.tensor([tokenizer.encode("ROMEO:\n")], dtype=torch.long, device=device)
                for _ in range(15):
                    out_logits, _ = model(test_prompt)
                    next_tok = torch.argmax(out_logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
                    test_prompt = torch.cat([test_prompt, next_tok], dim=1)
                gen_text = tokenizer.decode(test_prompt[0].tolist()).replace('\n', ' ')
                print(f"      AI Diyor ki: '{gen_text}'")
            model.train()

    # 5. AĞIRLIKLARI KAYDET (CHECKPOINT)
    save_path = "weights/topos_custom_llm.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ [BAŞARILI] Topos-LLM Eğitimi Tamamlandı!")
    print(f"Beyin ağırlıkları '{save_path}' konumuna kaydedildi.")
    print("Artık 'chat_custom_llm.py' çalıştırarak kendi AI'ınızla konuşabilirsiniz!")

if __name__ == "__main__":
    train()
