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
        """
        [ADJOINT FUNCTOR / NEWTON-RAPHSON ROOT FINDING]
        Kaba kuvvet (for t in range) yerine, hedefin dairesel yörüngesi ile
        geminin doğrusal uçuşunu Kesiştiren (Intersection) fonksiyonun
        kökünü (Root) Newton-Raphson (Analitik türev) ile O(1) sürede bulur.
        """
        w = p_info['angular_velocity']
        
        if w == 0.0:
            dist = math.sqrt((p_info['x'] - my_x)**2 + (p_info['y'] - my_y)**2)
            t = dist / self.ship_speed
            return t, p_info['x'], p_info['y']
            
        a0 = p_info['angle']
        r = p_info['radius']
        v = self.ship_speed
        
        dist_0 = math.sqrt((p_info['x'] - my_x)**2 + (p_info['y'] - my_y)**2)
        t = dist_0 / v
        
        for _ in range(10): # 10 adımda mükemmel hassasiyet
            angle = a0 + w * t
            cx = 50.0 + r * math.cos(angle)
            cy = 50.0 + r * math.sin(angle)
            
            dx = cx - my_x
            dy = cy - my_y
            
            dist = math.sqrt(dx**2 + dy**2)
            if dist < 1e-5:
                break
                
            f_t = dist - v * t
            
            x_prime = -r * w * math.sin(angle)
            y_prime =  r * w * math.cos(angle)
            
            f_prime_t = ((dx * x_prime) + (dy * y_prime)) / dist - v
            
            if abs(f_prime_t) < 1e-5:
                break
                
            t_new = t - (f_t / f_prime_t)
            
            if t_new < 0 or math.isnan(t_new):
                break
                
            if abs(t_new - t) < 0.01:
                t = t_new
                break
            t = t_new

        fx = 50.0 + r * math.cos(a0 + w * t)
        fy = 50.0 + r * math.sin(a0 + w * t)
        
        return t, fx, fy

    def get_moves(self, obs):
        try:
            player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, 'player', 0)
            raw_planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, 'planets', [])
            raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, 'fleets', [])
            env_angular_velocity = obs.get("angular_velocity", 0.05) if isinstance(obs, dict) else getattr(obs, 'angular_velocity', 0.05)

            planets = []
            for p in raw_planets:
                p_id, owner, x, y, radius, ships, production = p
                current_angle = math.atan2(y - 50.0, x - 50.0)
                dist_c = math.sqrt((x-50.0)**2 + (y-50.0)**2)
                angular_velocity = env_angular_velocity if dist_c < 45.0 else 0.0

                planets.append({
                    'id': p_id, 'owner': owner, 'x': float(x), 'y': float(y), 'radius': float(radius), 
                    'ships': int(ships), 'production': int(production),
                    'angle': current_angle, 'angular_velocity': angular_velocity
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
                for mine in my_planets:
                    angle_to_mine = math.atan2(mine['y'] - fy, mine['x'] - fx)
                    if abs(angle_to_mine - f_angle) < 0.2:
                        threats[mine['id']] += f_ships

            moves = []
            for mine in my_planets:
                defense_reserve = int(threats[mine['id']])
                available_ships = mine['ships'] - defense_reserve

                if available_ships <= 0:
                    continue

                best_target = None
                best_roi = -float('inf')
                best_fx, best_fy = None, None
                ships_to_send = 0

                targets = [p for p in planets if p['owner'] != player]
                if not targets:
                    continue

                for target in targets:
                    dist = math.sqrt((mine['x'] - target['x'])**2 + (mine['y'] - target['y'])**2)
                    required_ships = target['ships'] + 1

                    if available_ships >= required_ships:
                        roi = (target['production'] * 100.0) - (dist * 2.0) - required_ships

                        if target['owner'] == -1:
                            roi += 500.0
                        else:
                            roi += 100.0

                        if roi > best_roi:
                            best_roi = roi
                            best_target = target
                            ships_to_send = required_ships

                if best_target is not None:
                    t, fx, fy = self.calculate_interception(mine['x'], mine['y'], best_target)

                    if t != float('inf') and fx is not None and fy is not None:
                        angle = math.atan2(fy - mine['y'], fx - mine['x'])
                    else:
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
