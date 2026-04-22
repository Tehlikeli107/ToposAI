import torch
import torch.nn as nn

class ToposRelationMatrix(nn.Module):
    """
    Kategori Teorisinde nesneler arası mantıksal bağıntıları (Relations) tutan matris.
    Değerler [0, 1] arasındadır (Doğruluk derecesi).
    """
    def __init__(self, num_entities):
        super().__init__()
        self.num_entities = num_entities
        # İlişki matrisini rastgele başlatıyoruz ama logit olarak.
        # Sigmoid ile 0-1 arasına çekeceğiz.
        self.relation_logits = nn.Parameter(torch.randn(num_entities, num_entities))

    def get_relations(self):
        # A -> B doğruluk matrisi (A satır, B sütun)
        return torch.sigmoid(self.relation_logits)

def lukasiewicz_composition(R1, R2):
    """
    İki mantıksal matrisin Lukasiewicz T-Norm ve S-Norm ile bileşkesi.
    A -> B (R1) ve B -> C (R2) ise A -> C hesaplar.
    
    R1: (N, M)
    R2: (M, K)
    Output: (N, K)
    
    Kategori Teorisindeki Morphism Composition işlemidir.
    """
    N, M = R1.shape
    _, K = R2.shape
    
    # Boyutları eşleştir (Broadcasting)
    R1_exp = R1.unsqueeze(2) # (N, M, 1)
    R2_exp = R2.unsqueeze(0) # (1, M, K)
    
    # 1. T-Norm (Mantıksal VE / Kesişim): max(0, R1 + R2 - 1)
    # Eğer A -> B doğruysa (1) ve B -> C doğruysa (1), o zaman 1 + 1 - 1 = 1 (Doğru)
    # Eğer biri yanlışsa (0), o zaman 0 + 1 - 1 = 0 (Yanlış)
    t_norm = torch.clamp(R1_exp + R2_exp - 1.0, min=0.0) # (N, M, K)
    
    # 2. S-Norm / Supremum (Mantıksal VEYA / En iyi yolu seç): max üzerinden
    # A'dan C'ye gitmek için en mantıklı aracı B'yi (M boyutu) bul.
    composition, _ = torch.max(t_norm, dim=1) # (N, K)
    
    return composition

def neuro_symbolic_syllogism():
    entities = ["cem", "aslan", "vahşi", "ali", "insan", "fani", "taş"]
    entity_to_idx = {e: i for i, e in enumerate(entities)}
    
    # 1. Model: Varlıklar arası ilişki matrisi
    model = ToposRelationMatrix(num_entities=len(entities))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # 2. Eğitim Verisi (Sadece "A -> B" doğrudan ilişkileri)
    # 1.0: Kesin Doğru, 0.0: Kesin Yanlış
    true_relations = [
        ("cem", "aslan"),
        ("aslan", "vahşi"),
        ("ali", "insan"),
        ("insan", "fani")
    ]
    
    # Taş hiçbir şeye bağlı değil (Negatif örnek)
    false_relations = [
        ("cem", "insan"),
        ("aslan", "fani"),
        ("ali", "aslan"),
        ("taş", "fani"),
        ("taş", "vahşi")
    ]
    
    print("Mantıksal Bağlantılar (1 Hop) Eğitiliyor...")
    for epoch in range(100):
        optimizer.zero_grad()
        R = model.get_relations()
        loss = 0
        
        # Doğru ilişkiler (R = 1 olmalı)
        for e1, e2 in true_relations:
            i, j = entity_to_idx[e1], entity_to_idx[e2]
            loss += (1.0 - R[i, j])**2 # MSE
            
        # Yanlış ilişkiler (R = 0 olmalı)
        for e1, e2 in false_relations:
            i, j = entity_to_idx[e1], entity_to_idx[e2]
            loss += (0.0 - R[i, j])**2
            
        loss.backward()
        optimizer.step()
        
    print("Eğitim Tamamlandı. Sadece 1. Derece (Doğrudan) bağları öğrendi.\n")
    
    # 3. Kategori Teorisi Testi: COMPOSITION (Bileşke / Syllogism)
    # Modeli "Cem -> Vahşi" diye EĞİTMEDİK! Sadece "Cem -> Aslan" ve "Aslan -> Vahşi" verdik.
    
    R = model.get_relations()
    
    # Transitive Closure (1 adımlık zıplamaların birleşimi)
    # R_composed = R * R (Lukasiewicz çarpımı)
    R_composed = lukasiewicz_composition(R, R)
    
    print("--- NEURO-SYMBOLIC MANTIKSAL ÇIKARIM (2. DERECE / SYLLOGISM) ---")
    
    test_queries = [
        ("cem", "vahşi"),  # Beklenen: ~1.0 (Çünkü Cem->Aslan->Vahşi)
        ("ali", "fani"),   # Beklenen: ~1.0 (Çünkü Ali->İnsan->Fani)
        ("cem", "fani"),   # Beklenen: ~0.0 (Mantıksız)
        ("taş", "vahşi")   # Beklenen: ~0.0 (Mantıksız)
    ]
    
    for e1, e2 in test_queries:
        i, j = entity_to_idx[e1], entity_to_idx[e2]
        
        # Öğrenilen (Doğrudan) ilişki
        direct_val = R[i, j].item()
        
        # Kategori Teorisinden gelen Çıkarımsal (Composed) ilişki
        composed_val = R_composed[i, j].item()
        
        print(f"Sorgu: {e1.upper()} -> {e2.upper()}")
        print(f"  Doğrudan Bağlantı (Modelin bildiği):  {direct_val:.3f}")
        print(f"  Bileşke Çıkarımı (Topos Composition): {composed_val:.3f}")
        
        if composed_val > 0.8:
            print(f"  SONUÇ: ZİNCİRLEME MANTIK KURULDU! ({e1} -> Aracı -> {e2})")
        else:
            print(f"  SONUÇ: Mantıksal Bağlantı Bulunamadı.")
        print("-" * 50)

if __name__ == "__main__":
    neuro_symbolic_syllogism()
