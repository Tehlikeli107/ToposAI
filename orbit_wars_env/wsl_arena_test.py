import sys
import os

# ToposAI yolunu ekle (WSL dosya sisteminden Windows dosyasına erişim)
sys.path.append("/mnt/c/Users/salih/ToposAI")

from kaggle_environments import make

# Baseline ajan ve Topos ajan dosya yolları
baseline_agent_path = "/mnt/c/Users/salih/ToposAI/orbit_wars_env/main.py"
warlord_agent_path = "/mnt/c/Users/salih/ToposAI/applications/kaggle_orbit_wars_2500_elo.py"

# Orbit wars çevresini başlat (4 ajanlı simülasyon)
print("[WSL]: Kaggle Orbit Wars Ortamı (C++ Motoru) Başlatılıyor...")
env = make("orbit_wars", debug=True, configuration={"episodeSteps": 500})

# Ajanları arenaya sok: ToposAI Warlord vs 3x Kaggle Baseline
print("[WSL]: ToposAI Warlord vs 3x Kaggle Baseline (4-Player FFA) Arenaya Yükleniyor...")
env.run([warlord_agent_path, baseline_agent_path, baseline_agent_path, baseline_agent_path])

print("\n--- 🏁 MAÇ SONUCU (4 OYUNCULU KAOS) ---")
rewards = [env.steps[-1][i]['reward'] for i in range(4)]
print(f"  ToposAI Warlord: {rewards[0]}")
print(f"  Baseline 1:      {rewards[1]}")
print(f"  Baseline 2:      {rewards[2]}")
print(f"  Baseline 3:      {rewards[3]}")

if rewards[0] == max(rewards):
    print("\n🏆 KAZANAN: TOPOS AI (2500+ ELO Warlord)!")
else:
    print("\n🚨 KAZANAN: Düşman Botlarından Biri!")

print("\n[WSL BİLGİ]: Kaggle C++ (OpenSpiel) Motoru Başarıyla Test Edildi!")
