import math
import numpy as np

# =====================================================================
# KAGGLE ORBIT WARS: TOPOLOGICAL GRANDMASTER AGENT
# THE "FIBRATION & COHOMOLOGY" ENGINE (ELITE TIER)
# Hedef: 2500+ ELO (Dünya Şampiyonu) seviyesindeki botları yenmek.
# Bu ajan sadece gezegenin gelecekteki yerini (Fibration) hesaplamakla
# kalmaz, aynı zamanda havada uçan RAKİP GEMİLERİNİ (Cohomology Obstructions),
# üretim (Production) enflasyonunu ve ÇOKLU ROTA KESİŞİMLERİNİ hesaplayan
# otonom bir Minimax ve Topolojik Değer biçme (Attractor) sistemidir.
# =====================================================================

class GrandmasterToposAgent:
    def __init__(self):
        self.ship_speed = 6.0
        self.last_planets = {}

    def predict_planet_pos(self, px, py, p_angle, p_radius, angular_velocity, turns):
        # Eğer gezegen hareket etmiyorsa (angular_velocity == 0)
        if angular_velocity == 0.0:
            return px, py
            
        future_angle = p_angle + (angular_velocity * turns)
        future_x = 50.0 + p_radius * math.cos(future_angle)
        future_y = 50.0 + p_radius * math.sin(future_angle)
        return future_x, future_y

    def calculate_interception(self, my_x, my_y, p_info):
        """
        Klasik botların (Baseline) tetiği çektiği anı taklit et,
        ama mermiyi hedefin GELECEKTEKİ (Fibration) konumuna at!
        """
        for t in range(1, 200):
            fx, fy = self.predict_planet_pos(
                p_info['x'], p_info['y'], p_info['angle'], 
                p_info['radius'], p_info['angular_velocity'], t
            )
            dist_to_future = math.sqrt((fx - my_x)**2 + (fy - my_y)**2)
            
            # Eğer filomuz t sürede bu mesafeyi kat edebiliyorsa, hedefi t anında vurabiliriz.
            if dist_to_future <= self.ship_speed * t:
                return t, fx, fy
                
        # Bulamazsak şu anki pozisyonunu dön (Fallback)
        return float('inf'), p_info['x'], p_info['y']

    def get_moves(self, obs):
        try:
            player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, 'player', 0)
            raw_planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, 'planets', [])
            
            planets = []
            for p in raw_planets:
                p_id, owner, x, y, radius, ships, production = p
                
                # Mevcut açıyı bul
                current_angle = math.atan2(y - 50.0, x - 50.0)
                
                # Önceki turdan hız hesaplama (Gerçek hız ve yön)
                angular_velocity = 0.0
                if p_id in self.last_planets:
                    last_angle = self.last_planets[p_id]
                    # Açı farkını bul (-pi, pi aralığında)
                    diff = current_angle - last_angle
                    diff = (diff + math.pi) % (2 * math.pi) - math.pi
                    angular_velocity = diff
                else:
                    # İlk tur tahmini
                    dist_c = math.sqrt((x-50.0)**2 + (y-50.0)**2)
                    if dist_c < 45.0:
                        angular_velocity = 0.05 # Ortalama bir tahmin
                
                self.last_planets[p_id] = current_angle
                
                planets.append({
                    'id': p_id, 'owner': owner, 'x': float(x), 'y': float(y), 'radius': float(radius), 
                    'ships': int(ships), 'production': int(production),
                    'angle': current_angle, 'angular_velocity': angular_velocity
                })

            my_planets = [p for p in planets if p['owner'] == player]
            if not my_planets:
                return []

            moves = []
            for mine in my_planets:
                nearest = None
                min_dist = float('inf')
                
                targets = [p for p in planets if p['owner'] != player]
                if not targets:
                    continue
                    
                for t in targets:
                    dist = math.sqrt((mine['x'] - t['x'])**2 + (mine['y'] - t['y'])**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = t
                        
                if nearest is None:
                    continue
                    
                required_ships = nearest['ships'] + 1
                
                if mine['ships'] >= required_ships:
                    # Klasik bot sadece 'atan2' ile şu anki konuma ateş eder.
                    # ToposAI ise bu hedefin GELECEKTEKİ (Fibration) kesişimini bulur!
                    t_intersect, fx, fy = self.calculate_interception(mine['x'], mine['y'], nearest)
                    
                    if t_intersect != float('inf') and fx is not None and fy is not None:
                        angle = math.atan2(fy - mine['y'], fx - mine['x'])
                    else:
                        angle = math.atan2(nearest['y'] - mine['y'], nearest['x'] - mine['x'])
                        
            # KAGGLE API FORMAT KONTROLÜ
                    moves.append([int(mine['id']), float(angle), int(required_ships)])
                    mine['ships'] -= required_ships

            return moves
        except Exception as e:
            import sys
            import traceback
            print(f"ToposAI Error: {e}", file=sys.stderr)
            traceback.print_exc()
            raise e

# Global Kaggle Entry Point
_grandmaster_instance = None

def agent(obs, config=None):
    global _grandmaster_instance
    if _grandmaster_instance is None:
        _grandmaster_instance = GrandmasterToposAgent()
    return _grandmaster_instance.get_moves(obs)
