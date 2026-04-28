import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import time
from topos_ai.mamba import ToposMambaLM
from topos_ai.tokenization import TopologicalTokenizer

# =====================================================================
# BIG DATA DEMO: STREAMING STATE ON WIKITEXT-2
# İddia: Transformer modelleri "Büyük Veri" (Örn: Wikipedia okumak)
# işlerken tüm geçmişi KV-Cache (RAM) içinde tutmak zorundadır. Bu
# yüzden uzun metinlerde çökeler (OOM). Topos-Mamba (State Space Models)
# ise geçmişi tek boyutlu bir 'State (Hafıza Vektörü)' içinde taşır.
# Bu deneyde, makineye on binlerce kelimelik gerçek HuggingFace
# WikiText-2 veri setini 'Streaming (Akış)' olarak chunk'lar halinde 
# yedireceğiz. VRAM kullanımı ve loss raporlanır; bu demo literal sonsuz
# bağlam veya sabit bellek garantisi vermez.
# =====================================================================

def run_infinite_context_benchmark():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 64: STREAMING STATE (WIKITEXT-2 BIG DATA) ")
    print(" İddia: Klasik YZ'ler (GPT-4) 100.000 kelime okuduğunda GPU'ları patlar")
    print(" (OOM), çünkü tüm geçmişi RAM'de (KV-Cache) tutarlar.")
    print(" ToposMamba ise O(N) karmaşıklıklı ve 'Stateful (Durumsal)' bir")
    print(" Kategori motorudur. Bu test, makineye gerçek HuggingFace Wikipedia")
    print(" metinlerini on binlerce token halinde (Chunk by Chunk) aralıksız")
    print(" akıtır. VRAM değişimi ve loss ölçülür; sonuçlar donanım ve model")
    print(" boyutuna bağlıdır. Bu bir streaming-state demosudur, unbounded")
    print(" memory garantisi değildir.")
    print("=========================================================================\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        from datasets import load_dataset
        import datasets
        # HuggingFace gereksiz uyarılarını kapat
        datasets.logging.set_verbosity_error()
        
        print("[VERİ]: HuggingFace 'WikiText-2' Veri Seti (Streaming Modu) Yükleniyor...")
        # Sadece Streaming kullanıyoruz ki RAM dolmasın!
        dataset = load_dataset('wikitext', 'wikitext-2-v1', split='train', streaming=True)
    except ImportError:
        print("🚨 HATA: 'datasets' kütüphanesi bulunamadı! 'pip install datasets' çalıştırın.")
        return
    except Exception as e:
        print(f"🚨 HATA: İnternet veya Veri seti bağlantısı kurulamadı: {e}")
        return

    # Kendi Tokenizer'ımızı ayağa kaldıralım
    tokenizer = TopologicalTokenizer(vocab_size=500)
    
    # Canlı akışta metni toplayıp tokenlara bölelim
    print("  > İlk makaleler çekilip Tokenize ediliyor...")
    text_buffer = ""
    max_tokens_to_read = 30000 # 30 Bin kelime/token okuyacağız!
    
    iterator = iter(dataset)
    
    while len(text_buffer) < max_tokens_to_read * 5: # Kaba bir karakter-token çarpanı
        try:
            row = next(iterator)
            if row['text'].strip():
                text_buffer += row['text'] + " "
        except StopIteration:
            break

    # İlk önce biraz train yapalım ki sözlük dolsun
    tokenizer.train(text_buffer[:50000])
    
    all_tokens = tokenizer.encode(text_buffer)
    all_tokens = all_tokens[:max_tokens_to_read] # Sınırı kes
    
    total_tokens = len(all_tokens)
    print(f"  > Hazırlanan Toplam Token: {total_tokens:,}\n")
    
    if total_tokens < 1000:
        print("Yeterli veri çekilemedi.")
        return

    # MODELİ KUR
    vocab_size = tokenizer.target_vocab_size
    d_model = 64
    chunk_size = 512 # Her seferinde 512 kelimelik parçalar halinde akıtacağız
    
    mamba = ToposMambaLM(vocab_size, d_model=d_model, num_layers=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("--- STREAMING-STATE TESTİ BAŞLIYOR ---")
    print(f"  > Okuma Hızı (Chunk Size): {chunk_size} Token/Adım")
    print(f"  > Beklenen Davranış: state taşınır; VRAM ve loss ölçülür.\n")
    
    print(f"{'Okunan Token':<20} | {'VRAM Tüketimi':<20} | {'Current Loss (Anlama Gücü)'}")
    print("-" * 75)
    
    # BÜYÜK VERİ AKIŞI (Streaming Loop)
    state = None # Streaming state carried across chunks.
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        initial_vram = torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        initial_vram = 0.0

    t0 = time.time()
    
    num_chunks = total_tokens // chunk_size
    
    for i in range(num_chunks):
        # Akıştan bir parça al
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        
        # Son parça hedef olacak (Next-word prediction)
        chunk_tokens = all_tokens[start_idx:end_idx]
        
        if len(chunk_tokens) < chunk_size:
            break
            
        x_input = torch.tensor([chunk_tokens[:-1]], dtype=torch.long, device=device)
        y_target = torch.tensor([chunk_tokens[1:]], dtype=torch.long, device=device)
        
        # MODELİ ÇALIŞTIR (STATE'İ İÇERİ VER, YENİ STATE'İ AL)
        with torch.no_grad(): # Sadece Inference/Reading yapıyoruz
            logits, new_state = mamba(x_input, states=state)
            
        # Loss hesapla
        loss = criterion(logits.view(-1, vocab_size), y_target.view(-1))
        
        # STATE GÜNCELLEMESİ! (Tüm geçmiş artık 'new_state' içinde)
        state = new_state
        
        # VRAM Kontrolü
        if device.type == 'cuda':
            current_vram = torch.cuda.memory_allocated() / (1024 * 1024)
            vram_usage = current_vram - initial_vram
        else:
            vram_usage = 0.0
            
        if (i + 1) % 5 == 0 or i == 0:
            print(f"{((i+1) * chunk_size):<20,} | {vram_usage:>10.1f} MB Ekstra | {loss.item():.4f}")

    t1 = time.time()

    print("\n[ÖLÇÜLEN SONUÇ: STREAMING-STATE BENCHMARK]")
    print(f"Toplam Okuma Süresi: {t1 - t0:.2f} saniye.")
    print("Klasik bir Transformer (GPT), 30.000 kelimeyi okuduğunda O(N^2) matrisler")
    print("ve Devasa KV-Cache matrisleri yüzünden GB'larca Ekstra VRAM yakardı.")
    print("ToposMamba tarafında ölçülen ek VRAM aşağıdaki tabloda raporlandı;")
    print("sonuçlar donanım, model boyutu ve PyTorch ayırıcı davranışına bağlıdır.")
    print("Çünkü geçmişin tamamı, RAM'e kaydedilmek yerine ToposAI'ın 64-Boyutlu")
    print("'Bulanık Durum (Fuzzy State)' vektörü içinde topolojik olarak sıkıştırılmıştır.")
    print("Bu, chunked streaming kullanımının pratik bir demosudur; sınırsız veri")
    print("veya sabit bellek garantisi değildir.")

if __name__ == "__main__":
    run_infinite_context_benchmark()
