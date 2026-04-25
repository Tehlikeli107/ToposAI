import torch
import random
import time

# =====================================================================
# TOPOLOGICAL REINFORCEMENT LEARNING (ACTIVE INFERENCE / OTONOM AJAN)
# Klasik Q-Learning "Aksiyon-Değer (Ezber)" mantığıyla çalışır.
# Topos Ajanı (TRL) ise yürürken dünyanın "Kategori Teorisi Haritasını" 
# (Morfizmalar Ağı) çıkarır. Haritayı (Graph) birleştirdiği anda,
# hedefine ZERO-SHOT (Hiç deneme-yanılma yapmadan) en kısa yoldan ulaşır.
# =====================================================================

class TopologicalExplorer:
    def __init__(self, num_rooms):
        self.num_rooms = num_rooms
        # Ajanın kafasındaki Dünya Haritası (Knowledge Graph)
        # R[u, v] = 1.0 demek, "u odasından v odasına bir kapı var" demektir.
        self.R = torch.zeros(num_rooms, num_rooms)
        
    def explore_step(self, current_room, next_room):
        """Ajan dünyayı gezerken kapıları (Morfizmaları) hafızasına alır."""
        self.R[current_room, next_room] = 1.0
        self.R[next_room, current_room] = 1.0 # Kapılar genelde iki yönlüdür
        
    def find_path_to_goal(self, start_room, target_room):
        """
        [PLANNING VIA TRANSITIVE CLOSURE]
        Ajan, 'Ben Peynire nasıl giderim?' diye sorunca, rastgele (Q-Learning) 
        yürümez. Kafasındaki Matrisi (R) kullanarak Peynire giden "En Kısa Oku"
        (Morfizma zincirini) matematiksel bir Devre (Circuit) olarak çeker.
        """
        if start_room == target_room:
            return [start_room]
            
        # BFS (Breadth-First Search) ile Topos Matrisi üzerinden hedef arama
        queue = [[start_room]]
        visited = set()
        visited.add(start_room)
        
        while queue:
            path = queue.pop(0)
            node = path[-1]
            
            if node == target_room:
                return path # Peynire giden yolu buldu!
                
            # Kafasındaki haritaya bak (R matrisi)
            connected_rooms = torch.where(self.R[node] > 0.0)[0].tolist()
            
            for next_r in connected_rooms:
                if next_r not in visited:
                    visited.add(next_r)
                    new_path = list(path)
                    new_path.append(next_r)
                    queue.append(new_path)
                    
        return None # Hedefi (Peyniri) henüz haritasında bilmiyor!

def run_topological_rl_experiment():
    print("--- TOPOLOGICAL REINFORCEMENT LEARNING (OTONOM ROBOT AJAN) ---")
    print("Ajan, Q-Learning (Ezber) yerine, dünyanın 'Topolojik Haritasını' kurarak\nlabirenti Zero-Shot (Deneme-Yanılma Olmadan) çözecek.\n")

    # LABİRENT (GERÇEK DÜNYA)
    # Odalar: 0'dan 5'e kadar (6 Oda)
    # Harita Şekli:
    # 0 -- 1 -- 2
    #      |    |
    #      3 -- 4 -- 5 (PEYNİR)
    
    world_edges = [
        (0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)
    ]
    
    agent = TopologicalExplorer(num_rooms=6)
    
    start_pos = 0
    goal_pos = 5
    
    print(f"[GÖREV]: Ajan (Robot), Oda {start_pos}'dan başlayıp Peyniri (Oda {goal_pos}) bulmalı.\n")
    print("==== 1. AŞAMA: KEŞİF (EXPLORATION) ====")
    print("Ajan karanlıkta yürüyor ve kapıları (Topos Oklarını) haritalandırıyor...")
    
    # Ajan labirenti (Gerçek dünyayı) "Rastgele (Random Walk)" geziyor.
    # (Daha verimli Frontier Exploration da olabilirdi ama rastgele gezelim)
    
    current = start_pos
    for step in range(1, 21):
        # Gerçek dünyada current odasına bağlı kapıları bul
        valid_doors = [v for u, v in world_edges if u == current] + [u for u, v in world_edges if v == current]
        
        # Kapılardan birini seç
        next_room = random.choice(valid_doors)
        
        # Ajan gördüğü kapıyı Zihnine (Topos Matrisine) KAZIR!
        agent.explore_step(current, next_room)
        
        print(f"  [Adım {step:02d}]: Ajan Oda {current} -> Oda {next_room} geçti. (Hafıza Güncellendi).")
        
        if next_room == goal_pos:
            print(f"\n  [!] AJAN PEYNİRİ (HEDEFİ) BULDU! (Oda {goal_pos}). Keşif bitti.")
            break
            
        current = next_room
        time.sleep(0.1)

    print("\n==== 2. AŞAMA: HEDEF DEĞİŞİYOR VE ZEKANIN UYANIŞI ====")
    print("Peyniri (Hedefi) aldık, Oda 4'e (Geriye) koyduk!")
    print("Ajanı da Oda 0'a (Başlangıca) geri ışınladık.")
    
    new_start = 0
    new_goal = 4
    
    print("\n[KLASİK AI (Q-LEARNING) NE YAPARDI?]:")
    print(f"Klasik AI, yeni hedefi (Oda 4) bulmak için, tekrar Oda {new_start}'dan başlayıp ")
    print("milyonlarca kez rastgele yürüyüp duvara çarparak yeni 'Q-Tablosunu' ezberlemeye çalışırdı.")
    
    print("\n[TOPOS AI (ACTIVE INFERENCE) NE YAPIYOR?]:")
    print(f"Topos Ajanı, kafasında kurduğu Kategori Teorisindeki (Morfizma Matrisindeki)")
    print(f"zincirleme ağı okuyarak, Oda {new_start}'dan Oda {new_goal}'e giden YOLU ")
    print("HİÇ HATA YAPMADAN KENDİ KENDİNE BULACAKTIR (Zero-Shot Planning).\n")
    
    # Ajan zihnindeki haritadan "İdealize Rotayı" çıkarır (Planlama)
    path = agent.find_path_to_goal(new_start, new_goal)
    
    if path:
        path_str = " ➔ ".join(map(str, path))
        print(">>> [MÜKEMMEL ROTA HESAPLANDI] <<<")
        print(f"   PLANLANAN YOL: {path_str}")
        print("\nSONUÇ: Ajan, çevreyi 'Topolojik' (Ağ Matrisi) olarak algıladığı için,")
        print("Hedef veya Başlangıç noktası nereye değişirse değişsin,")
        print("sistemi 'Baştan Eğitmek (Retraining)' ZORUNDA KALMADAN, anlık (Zero-Shot)")
        print("olarak çözümü SENTEZLER!")
    else:
        print("[-] Ajan haritayı yeterince gezmediği için yolu henüz bulamadı.")

if __name__ == "__main__":
    run_topological_rl_experiment()
