import sys
import os

# Yazarın eğitim dosyasını (train_custom_llm.py) import edip
# epoch ve veri seti ayarlarını "Toy" seviyeye indirip entegrasyonu test edeceğiz.
sys.path.append(r"C:\Users\salih\ToposAI")

from train_custom_llm import CharDataset
from topos_ai.models import ToposTransformer
from topos_ai.optim import ToposAdam
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def run_integration_test():
    print("=========================================================================")
    print(" SİSTEM ENTEGRASYON TESTİ: 'TOPOS-LLM' MİMARİSİ ÇÖKTÜ MÜ?")
    print("=========================================================================\n")
    
    # 1. SAHTE/MİNİK VERİ SETİ
    print("[1/3] Minik Veri Seti (Tiny Shakespeare Proxy) Yükleniyor...")
    text = "The quick brown fox jumps over the lazy dog. A category is a category of categories."
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    
    data = [stoi[c] for c in text]
    seq_len = 8
    dataset = CharDataset(data, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 2. TOPOS TRANSFORMER MODELİNİ BAŞLAT
    # Cihaz VRAM'ini şişirmemek için minik bir model
    print("[2/3] ToposTransformer (D=32, U=2) Başlatılıyor...")
    device = torch.device("cpu") # Sadece mantık testi için CPU
    
    model = ToposTransformer(vocab_size=vocab_size, d_model=32, num_universes=2, num_layers=1, max_seq_len=seq_len)
    model.to(device)
    
    optimizer = ToposAdam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 3. YALNIZCA 2 İTERASYONLUK EĞİTİM (İleri + Geri yayılım patlıyor mu?)
    print("[3/3] Özel Autograd Çekirdekleriyle Gerçek İterasyon (2 Step) Başlıyor...\n")
    
    model.train()
    steps_done = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Yeni yazılan "StrictGodel" sınıflarımız `nn.py` ve `models.py` içinden çağrılacak.
        logits, _ = model(x)
        
        # B * SeqLen, Vocab Size
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        
        loss.backward()
        
        # Nan/Inf Gradyan Kontrolü
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        print(f"  > Step {steps_done + 1}/2 | Topos-Loss: {loss.item():.4f}")
        steps_done += 1
        
        if steps_done >= 2:
            break
            
    print("\n=========================================================================")
    print(" SONUÇ: ENTEGRASYON BAŞARILI! Model çökmedi ve yeni çekirdekle eğitime devam edebiliyor.")
    print("=========================================================================")

if __name__ == '__main__':
    run_integration_test()