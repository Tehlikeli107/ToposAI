import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ToposMultiUniverse(nn.Module):
    """
    Birden fazla Local Truth (Bağlamsal Evren) destekleyen Topos Mimarisi.
    Her evren kendi mantıksal bağıntı matrisine (Relation Matrix) sahiptir.
    """
    def __init__(self, num_entities, num_universes):
        super().__init__()
        self.num_entities = num_entities
        self.num_universes = num_universes
        
        # Her evren için ayrı bir mantıksal matris (N_universes, N_entities, N_entities)
        self.relation_logits = nn.Parameter(torch.randn(num_universes, num_entities, num_entities))

    def get_relations(self, universe_idx=None):
        relations = torch.sigmoid(self.relation_logits)
        if universe_idx is not None:
            return relations[universe_idx]
        return relations

def lukasiewicz_composition(R1, R2):
    """A -> B ve B -> C ise A -> C (T-Norm / S-Norm Bileşkesi)"""
    R1_exp = R1.unsqueeze(2) 
    R2_exp = R2.unsqueeze(0) 
    t_norm = torch.clamp(R1_exp + R2_exp - 1.0, min=0.0) 
    composition, _ = torch.max(t_norm, dim=1) 
    return composition

def test_contextual_contradiction():
    entities = ["ateş", "su", "buhar", "buz", "küller", "dev_ateş"]
    entity_to_idx = {e: i for i, e in enumerate(entities)}
    
    universes = {"DÜNYA": 0, "BÜYÜ_EVRENİ": 1}
    
    # Model 6 varlık ve 2 farklı evren için eğitiliyor
    model = ToposMultiUniverse(num_entities=len(entities), num_universes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # EĞİTİM VERİSİ (BİLİNÇLİ ÇELİŞKİ)
    # 1. Dünya Evreni (Klasik Fizik)
    world_true = [("ateş", "su", "buhar"), ("su", "ateş", "küller")] 
    # Not: Basitleştirmek için A + B -> C mantığını A -> C olarak kodluyoruz (Ateş buhar yapar, Su küle çevirir/söndürür)
    world_relations = [("ateş", "buhar"), ("su", "küller")]
    
    # 2. Büyü Evreni (Ters Fizik - Çelişki)
    magic_true = [("ateş", "su", "buz"), ("su", "ateş", "dev_ateş")]
    magic_relations = [("ateş", "buz"), ("su", "dev_ateş")]
    
    print("Topos Düşünce Motoru Eğitiliyor... (Çelişkili Evrenler İzole Ediliyor)")
    for epoch in range(150):
        optimizer.zero_grad()
        R_all = model.get_relations()
        loss = 0
        
        # Dünya (Universe 0) Eğitimi
        R_world = R_all[0]
        for e1, e2 in world_relations:
            i, j = entity_to_idx[e1], entity_to_idx[e2]
            loss += (1.0 - R_world[i, j])**2 # Olmalı
        # Dünya'da büyü kuralları geçersiz (0.0)
        for e1, e2 in magic_relations:
            i, j = entity_to_idx[e1], entity_to_idx[e2]
            loss += (0.0 - R_world[i, j])**2
            
        # Büyü Evreni (Universe 1) Eğitimi
        R_magic = R_all[1]
        for e1, e2 in magic_relations:
            i, j = entity_to_idx[e1], entity_to_idx[e2]
            loss += (1.0 - R_magic[i, j])**2 # Olmalı
        # Büyü Evreninde fizik kuralları geçersiz (0.0)
        for e1, e2 in world_relations:
            i, j = entity_to_idx[e1], entity_to_idx[e2]
            loss += (0.0 - R_magic[i, j])**2
            
        loss.backward()
        optimizer.step()
        
    print("Eğitim Tamamlandı.\n")
    
    print("--- TOPOS MİMARİSİ: BAĞLAMSAL ÇELİŞKİ (LOCAL TRUTHS) TESTİ ---")
    print("Normal AI (Global Truth) bu iki zıt verinin ortalamasını alır ve kafası karışır.")
    print("Topos AI ise HANGİ EVRENDE olduğuna bakarak diğer evreni 'matematiksel hiçliğe' (0.0) indirger.\n")
    
    R_world_final = model.get_relations(universes["DÜNYA"])
    R_magic_final = model.get_relations(universes["BÜYÜ_EVRENİ"])
    
    test_queries = [
        ("ateş", "buhar"),
        ("ateş", "buz"),
        ("su", "küller"),
        ("su", "dev_ateş")
    ]
    
    for e1, e2 in test_queries:
        i, j = entity_to_idx[e1], entity_to_idx[e2]
        
        val_world = R_world_final[i, j].item()
        val_magic = R_magic_final[i, j].item()
        
        print(f"SORGU: '{e1.upper()}' -> '{e2.upper()}' Yapar mı?")
        print(f"  [DÜNYA EVRENİ] Doğruluk: {val_world:.3f} " + ("(DOĞRU)" if val_world > 0.8 else "(YALAN/İMKANSIZ)"))
        print(f"  [BÜYÜ EVRENİ]  Doğruluk: {val_magic:.3f} " + ("(DOĞRU)" if val_magic > 0.8 else "(YALAN/İMKANSIZ)"))
        print("-" * 60)

if __name__ == "__main__":
    test_contextual_contradiction()
