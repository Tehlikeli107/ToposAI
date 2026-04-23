import sys
import os

# ToposAI yolunu ekle (WSL dosya sisteminden Windows dosyasına erişim)
sys.path.append("/mnt/c/Users/salih/ToposAI")

from kaggle_environments import make

# Baseline ajan ve Topos ajan dosya yolları
baseline_agent_path = "/mnt/c/Users/salih/ToposAI/orbit_wars_env/main.py"
topos_agent_path = "/mnt/c/Users/salih/ToposAI/applications/kaggle_orbit_wars_topos_agent.py"

# Orbit wars çevresini başlat (2 ajanlı simülasyon)
print("[WSL]: Kaggle Orbit Wars Ortamı (C++ Motoru) Başlatılıyor...")
env = make("orbit_wars", debug=True, configuration={"episodeSteps": 100})

# Ajanları arenaya sok: ToposAI vs Kaggle Baseline
print("[WSL]: ToposAI vs Kaggle Baseline (Greedy Sniper) Arenaya Yükleniyor...")
# ToposAI dosyasında 'agent' adında callable bir wrapper yazmadığımız için,
# WSL üzerinde denemek adına şimdilik iki baseline'ı veya baseline vs random yapıp 
# arenanın çalıştığını test edelim.
env.run([topos_agent_path, baseline_agent_path])

print("\n--- 🏁 MAÇ SONUCU ---")
rewards = env.steps[-1][0]['reward'], env.steps[-1][1]['reward']
print(f"  ToposAI Ödülü: {rewards[0]} | Baseline Ödülü: {rewards[1]}")

if rewards[0] > rewards[1]:
    print("\n🏆 KAZANAN: TOPOS AI (Category Theory Agent)!")
elif rewards[1] > rewards[0]:
    print("\n🏆 KAZANAN: KAGGLE BASELINE (Nearest Sniper)!")
else:
    print("\n🤝 SONUÇ: BERABERLİK (Draw)!")

print("\n[WSL BİLGİ]: Kaggle C++ (OpenSpiel) Motoru Başarıyla Test Edildi!")
