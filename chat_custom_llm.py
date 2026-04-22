import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from topos_ai.models import ToposTransformer

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

    print("[SİSTEM] GPT-2 Tokenizer Yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    print("[SİSTEM] ToposTransformer Mimarisi Yükleniyor...")
    model = ToposTransformer(vocab_size=vocab_size, d_model=256, num_universes=8, num_layers=4)
    
    weights_path = "weights/topos_custom_llm.pt"
    if not os.path.exists(weights_path):
        print(f"🚨 HATA: Ağırlık dosyası bulunamadı! Önce 'train_custom_llm.py' çalıştırıp modeli eğitin.")
        return

    print(f"[SİSTEM] Kategori Ağırlıkları ({weights_path}) okunuyor...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    print("\n✅ TOPOS-LLM UYANDI! (Sohbete Başlayabilirsiniz)\n")

    while True:
        try:
            user_input = input("You> ")
            if user_input.lower() in ["exit", "quit"]:
                print("Topos-LLM: Kategori matrisleri kapatılıyor. Hoşça kal!")
                break
            
            if len(user_input.strip()) == 0:
                continue

            # Kullanıcı metnini Token'a çevir
            input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
            
            print("Topos-LLM> ", end="", flush=True)
            
            # Cümleyi Kelime Kelime Sentezle (Autoregressive Generation)
            max_tokens = 30
            with torch.no_grad():
                for _ in range(max_tokens):
                    # Kategori geçişliliği (Forward Pass)
                    logits = model(input_ids)
                    
                    # Son kelimenin vektörü
                    next_token_logits = logits[0, -1, :]
                    
                    # Basit sıcaklık (Temperature) ve Sampling eklenebilir, şimdilik Greedy Search
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
                    
                    # Kelimeyi ID'den Metne çevir ve anında (Stream) yazdır
                    word = tokenizer.decode(next_token[0].tolist())
                    print(word, end="", flush=True)
                    
                    # Üretilen kelimeyi hafızaya ekle
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
            print("\n") # Alt satıra geç

        except KeyboardInterrupt:
            print("\n[İptal Edildi]")
            break
        except Exception as e:
            print(f"\n[HATA]: {e}")

if __name__ == "__main__":
    chat_with_topos()
