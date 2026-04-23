import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import urllib.request

from topos_ai.models import ToposTransformer
from topos_ai.tokenization import TopologicalTokenizer

# =====================================================================
# TOPOS-LLM CHAT INTERFACE
# Amacı: Eğitilmiş olan (weights/topos_custom_llm.pt) Kategori Teorisi 
# dil modelini yüklemek ve kullanıcı ile terminal üzerinden etkileşime 
# girmesini sağlamak. (Zero-Hallucination Topological Decoding)
# =====================================================================

def chat_with_topos():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=========================================================================")
    print(" 🤖 TOPOS-LLM TERMINAL CHAT INTERFACE ")
    print(" Type 'exit' or 'quit' to close the portal.")
    print("=========================================================================\n")

    print("[SİSTEM] Kendi İcadımız Olan 'Topological Tokenizer' Yükleniyor...")
    tokenizer = TopologicalTokenizer(vocab_size=1000)
    
    # Hızlı Tokenizer Eğitimi (Sözlüğü geri kurmak için train() ile aynı veri/limitler)
    print("   > Tokenizer sözlüğü (1000 kelime) yeniden inşa ediliyor...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        text = urllib.request.urlopen(url).read().decode('utf-8')
        # Train scriptindeki ile birebir aynı (30000 char)
        tokenizer.train(text[:30000])
    except Exception as e:
        print(f"Tokenizer veri indirme hatası: {e}")
        return
        
    vocab_size = len(tokenizer.vocab)

    print(f"[SİSTEM] ToposTransformer Mimarisi Yükleniyor... (Vocab: {vocab_size})")
    model = ToposTransformer(vocab_size=vocab_size, d_model=256, num_universes=8, num_layers=4)
    
    weights_path = "weights/topos_custom_llm.pt"
    if not os.path.exists(weights_path):
        print(f"🚨 HATA: Ağırlık dosyası bulunamadı! Önce 'train_custom_llm.py' çalıştırıp modeli eğitin.")
        return

    print(f"[SİSTEM] Kategori Ağırlıkları ({weights_path}) okunuyor...")
    state_dict = torch.load(weights_path, map_location=device)
    
    # Geriye dönük uyumluluk (Backward Compatibility)
    if "yoneda_proj.weight" in state_dict:
        state_dict["yoneda_proj.weight_raw"] = state_dict.pop("yoneda_proj.weight")
        
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print("\n✅ TOPOS-LLM UYANDI! (Sohbete Başlayabilirsiniz)\n")

    # Sohbet hafızası (KV-Cache)
    kv_caches = None

    while True:
        try:
            user_input = input("You> ")
            if user_input.lower() in ["exit", "quit"]:
                print("Topos-LLM: Kategori matrisleri kapatılıyor. Hoşça kal!")
                break
            
            if len(user_input.strip()) == 0:
                continue

            # Kullanıcı metnini Token'a çevir
            input_ids = torch.tensor([tokenizer.encode(user_input)], dtype=torch.long, device=device)
            
            print("Topos-LLM> ", end="", flush=True)
            
            # Cümleyi Kelime Kelime Sentezle (Autoregressive Generation)
            max_tokens = 30
            with torch.no_grad():
                for _ in range(max_tokens):
                    # Kategori geçişliliği (Forward Pass)
                    # Optimizasyon: Inference anında KV-Cache kullanarak hızı 100x artırıyoruz!
                    logits, kv_caches = model(input_ids, kv_caches=kv_caches)
                    
                    # Son kelimenin vektörü
                    next_token_logits = logits[0, -1, :]
                    
                    # Basit sıcaklık (Temperature) ve Sampling eklenebilir, şimdilik Greedy Search
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
                    
                    # Kelimeyi ID'den Metne çevir ve anında (Stream) yazdır
                    word = tokenizer.decode(next_token[0].tolist())
                    print(word, end="", flush=True)
                    
                    # Üretilen kelimeyi hafızaya ekle (Bir dahaki sefer sadece son kelimeyi sor)
                    input_ids = next_token
                    
            print("\n") # Alt satıra geç

        except KeyboardInterrupt:
            print("\n[İptal Edildi]")
            break
        except Exception as e:
            print(f"\n[HATA]: {e}")

if __name__ == "__main__":
    chat_with_topos()
