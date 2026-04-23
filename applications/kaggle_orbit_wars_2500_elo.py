import math
import numpy as np

# =====================================================================
# KAGGLE ORBIT WARS: 2500+ ELO ARCHITECTURE (TOPOS WARLORD)
# Hedef: Kaggle'ın acımasız 4 oyunculu ortamında Dünya Şampiyonlarını
# devirecek Kategori Teorisi stratejilerini (Global Sheaf & ROI) kurmak.
# 1. Savunma (Cohomology): Kendi gezegenine gelen tehditleri ray-casting
#    ile sezip, ana gezegende o kadar gemiyi SAVUNMA olarak bekletir.
# 2. Ekonomik Çekim (ROI Attractors): Bir hedefin (Üretim * Zaman) karı
#    ile (Gereken Gemi + Mesafe) maliyetini bölerek evrendeki en 
#    optimal Topolojik Çekim Merkezini bulur.
# =====================================================================

class WarlordToposAgent:
    def __init__(self):
        self.ship_speed = 6.0

    def predict_planet_pos(self, px, py, p_angle, p_radius, angular_velocity, turns):
        if angular_velocity == 0.0:
            return px, py
        future_angle = p_angle + (angular_velocity * turns)
        future_x = 50.0 + p_radius * math.cos(future_angle)
        future_y = 50.0 + p_radius * math.sin(future_angle)
        return future_x, future_y

    def calculate_interception(self, my_x, my_y, p_info):
        for t in range(1, 200):
            fx, fy = self.predict_planet_pos(
                p_info['x'], p_info['y'], p_info['angle'], 
                p_info['radius'], p_info['angular_velocity'], t
            )
            dist = math.sqrt((fx - my_x)**2 + (fy - my_y)**2)
            if dist <= self.ship_speed * t:
                return t, fx, fy
        return float('inf'), None, None

    def get_moves(self, obs):
        try:
            player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, 'player', 0)
            raw_planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, 'planets', [])
            raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, 'fleets', [])
            
            planets = []
            for p in raw_planets:
                p_id, owner, x, y, radius, ships, production = p
                planets.append({
                    'id': p_id, 'owner': owner, 'x': float(x), 'y': float(y), 'radius': float(radius), 
                    'ships': int(ships), 'production': int(production)
                })

            my_planets = [p for p in planets if p['owner'] == player]
            if not my_planets:
                return []

            # 1. COHOMOLOGY DEFENSE (GELEN TEHDİTLER)
            threats = {p['id']: 0 for p in my_planets}
            for f in raw_fleets:
                f_id, f_owner, fx, fy, f_angle, f_origin, f_ships = f
                if f_owner == player:
                    continue
                # Eğer düşman filosu bizim gezegenimize geliyorsa (Kaba ray-casting)
                for mine in my_planets:
                    angle_to_mine = math.atan2(mine['y'] - fy, mine['x'] - fx)
                    if abs(angle_to_mine - f_angle) < 0.2:
                        threats[mine['id']] += f_ships

            moves = []
            for mine in my_planets:
                # Olası bir saldırıya karşı gezegende bırakmamız GEREKEN gemi sayısı
                defense_reserve = int(threats[mine['id']])
                available_ships = mine['ships'] - defense_reserve
                
                if available_ships <= 0:
                    continue
                
                best_target = None
                best_roi = -float('inf')
                ships_to_send = 0
                
                targets = [p for p in planets if p['owner'] != player]
                if not targets:
                    continue
                    
                for target in targets:
                    dist = math.sqrt((mine['x'] - target['x'])**2 + (mine['y'] - target['y'])**2)
                    
                    # ToposAI Attractor (ROI) Hesaplaması
                    # Kâr = Üretim * 100. Maliyet = Giden gemi + Mesafe
                    required_ships = target['ships'] + 1
                    
                    if available_ships >= required_ships:
                        roi = (target['production'] * 100.0) - (dist * 2.0) - required_ships
                        
                        # Tarafsızları ve düşmanları kap!
                        if target['owner'] == -1:
                            roi += 500.0
                        else:
                            roi += 100.0
                            
                        if roi > best_roi:
                            best_roi = roi
                            best_target = target
                            ships_to_send = required_ships
                            
                if best_target is not None:
                    angle = math.atan2(best_target['y'] - mine['y'], best_target['x'] - mine['x'])
                    moves.append([int(mine['id']), float(angle), int(ships_to_send)])
                    available_ships -= ships_to_send

            return moves
        except Exception as e:
            import sys
            import traceback
            print(f"ToposAI Warlord Error: {e}", file=sys.stderr)
            traceback.print_exc()
            return []

_warlord_instance = None
def agent(obs, config=None):
    global _warlord_instance
    if _warlord_instance is None:
        _warlord_instance = WarlordToposAgent()
    return _warlord_instance.get_moves(obs)
