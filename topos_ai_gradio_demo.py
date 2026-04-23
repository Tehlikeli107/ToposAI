import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import gradio as gr
from topos_ai.models import ToposTransformer
from topos_ai.tokenization import TopologicalTokenizer

# =====================================================================
# TOPOS-AI: HUGGINGFACE / GRADIO WEB DASHBOARD
# Amacı: Yatırımcılara (VC) ve araştırmacılara, arka planda çalışan
# o devasa Kategori Teorisi matrislerini görsel, tıklanabilir ve
# şık bir web arayüzü ile sunmak. Siyah terminal ekranını bırakıp,
# "Ürün" (Product) satma aşamasına geçiş.
# =====================================================================

# Global Değişkenler (Model bir kere yüklenecek)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = None
model = None

def load_system():
    global tokenizer, model
    
    print("[SİSTEM] Kendi İcadımız Olan 'Topological Tokenizer' Yükleniyor...")
    tokenizer = TopologicalTokenizer(vocab_size=1000)
    
    tokenizer_path = "weights/tokenizer.json"
    if os.path.exists(tokenizer_path):
        print(f"   > Tokenizer sözlüğü ({tokenizer_path}) diskten yükleniyor...")
        try:
            tokenizer.load(tokenizer_path)
        except Exception as e:
            print(f"Tokenizer yükleme hatası: {e}")
            return "Model Yükleme Hatası (Tokenizer okunamadı)"
    else:
        print("🚨 HATA: 'weights/tokenizer.json' bulunamadı!")
        return "Model Yükleme Hatası (Lütfen önce 'topos-train' çalıştırıp modeli eğitin)"
        
    vocab_size = len(tokenizer.vocab)
    model = ToposTransformer(vocab_size=vocab_size, d_model=256, num_universes=8, num_layers=4)
    
    weights_path = "weights/topos_custom_llm.pt"
    if not os.path.exists(weights_path):
        return "HATA: 'weights/topos_custom_llm.pt' bulunamadı. Lütfen önce modeli eğitin."

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
        return "✅ Sistem Başarıyla Yüklendi! Kategori Matrisleri Aktif."
    except Exception as e:
        return f"Ağırlık Yükleme Hatası: {str(e)}"

def generate_topos_text(prompt, max_length):
    global tokenizer, model
    
    if model is None or tokenizer is None:
        status = load_system()
        if "HATA" in status:
            return status

    if not prompt.strip():
        return "Lütfen bir başlangıç metni (Prompt) giriniz."

    # Kullanıcı metnini Token'a çevir
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    # Cümleyi Kelime Kelime Sentezle
    kv_caches = None
    output_text = prompt
    
    with torch.no_grad():
        for _ in range(int(max_length)):
            logits, kv_caches = model(input_ids, kv_caches=kv_caches)
            next_token_logits = logits[0, -1, :]
            
            # Greedy Search
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
            word = tokenizer.decode(next_token[0].tolist())
            output_text += word
            
            input_ids = next_token

    return output_text

# --- GRADIO ARAYÜZÜ (GUI) ---
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🌌 ToposAI: Neuro-Symbolic AGI Framework
        ### Beyond Deep Learning: Category Theory & Formal Logic
        Welcome to the ToposAI web dashboard. This model is **NOT** a standard Transformer (like ChatGPT or LLaMA). 
        It does not use Dot-Products, Softmax, LayerNorm, or CrossEntropy. It operates purely on **Topological Reachability [0,1]**, **Fuzzy Logic (Lukasiewicz T-Norm)**, and **Bijective Functors**.
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Interact with the Pure Topos-LLM")
            prompt_input = gr.Textbox(lines=3, placeholder="Enter a prompt (e.g., 'ROMEO:')", label="Input Prompt")
            length_slider = gr.Slider(minimum=5, maximum=100, value=30, step=1, label="Max New Tokens")
            generate_btn = gr.Button("Generate with Category Theory", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Output")
            output_display = gr.Textbox(lines=10, label="ToposAI Response")
            
    generate_btn.click(fn=generate_topos_text, inputs=[prompt_input, length_slider], outputs=[output_display])
    
    gr.Markdown(
        """
        ---
        ### 🧠 Why ToposAI? (The Pitch for Investors & Researchers)
        *   **Reduced Hallucination (Theoretical):** Hallucinations in LLMs are caused by continuous vector interpolations. ToposAI uses discrete categorical paths. If there is no morphism (logical arrow), the model limits hallucination vectors.
        *   **O(1) Backward Pass Optimization:** We implemented a custom `Triton C++ Kernel` that accumulates gradients directly in SRAM, managing VRAM footprint and allowing the model to scale efficiently.
        *   **Formal Verification:** Integration with the **Lean 4** Theorem Prover. Categorical logic paths can be formally verified.
        
        *Built by the Principal Investigator. 2026.*
        """
    )

if __name__ == "__main__":
    print("ToposAI Web Sunucusu Başlatılıyor...")
    demo.launch(server_name="0.0.0.0", share=False, theme=gr.themes.Default())
