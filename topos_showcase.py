import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("=" * 70)
    print(" 🌌 TOPOS AI: NEURO-SYMBOLIC AGI FRAMEWORK (v1.0.0) 🌌")
    print(" Beyond Deep Learning: Category Theory, Topos Logic & Formal Verification")
    print("=" * 70)
    print("\nLütfen çalıştırmak istediğiniz Bilimsel Kanıt'ı (Proof) seçin:\n")

def run_showcase():
    options = {
        "1": ("Tıbbi Otonom Teorem Keşfi (Kategori Mantığı)", "experiments/autonomous_theorem_discovery.py"),
        "2": ("Zaman Makinesi (Holografik TQFT & Zamanı Geri Alma)", "experiments/holographic_emergent_space_tqft.py"),
        "3": ("Algoritmik Tekillik (C++ Kodu Yazan Gödel Makinesi)", "experiments/topological_algorithmic_self_optimization.py"),
        "4": ("Milyar Dolarlık Çapraz-Alan Tavsiye Motoru (Amazon/Netflix)", "applications/real_world_topological_recommendation.py"),
        "5": ("Topolojik Arbitraj (Binance Sıfır-Gecikme Sinyal Üreticisi)", "applications/real_world_bitcoin_topos_alpha.py"),
        "6": ("İnsan Dilinin Matematiği (Generative Universal Grammar)", "applications/generative_universal_grammar_topos.py"),
        "7": ("Katastrofik Unutmanın Ölümü (Kan Extensions Continual Learning)", "applications/categorical_kan_extensions_transfer_learning.py"),
        "8": ("Yüksek Kategori YZ (Saniyede Şekil Değiştiren HyperNetworks)", "experiments/higher_category_theory_hypernetworks.py"),
        "9": ("Topos-Mamba vs Attention (VRAM Sınırlarını Aşan Hız Testi)", "benchmarks/mamba_vs_attention_benchmark.py"),
        "0": ("Çıkış (Exit)", None)
    }

    while True:
        clear_screen()
        print_header()
        
        for key, (desc, _) in options.items():
            print(f"  [{key}] {desc}")
            
        choice = input("\nSeçiminiz (0-9)> ")
        
        if choice == "0":
            print("\nToposAI Sistemleri Kapatılıyor. Gelecekte görüşmek üzere!")
            break
            
        if choice in options:
            script_path = options[choice][1]
            if script_path and os.path.exists(script_path):
                clear_screen()
                print(f">>> ÇALIŞTIRILIYOR: {script_path} <<<\n")
                # Run the script using os.system for direct console output
                os.system(f"{sys.executable} {script_path}")
                input("\n[Devam etmek için ENTER'a basın...]")
            else:
                print(f"\nHATA: {script_path} bulunamadı!")
                time.sleep(2)
        else:
            print("\nGeçersiz seçim!")
            time.sleep(1)

if __name__ == "__main__":
    run_showcase()
