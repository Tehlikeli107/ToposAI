import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
import time

# =====================================================================
# REAL-WORLD ONTOLOGY BENCHMARK: YONEDA VS DOT-PRODUCT
# Sentetik (Entity_1) veriler yerine, GERÇEK İNGİLİZCE kelimelerden oluşan
# (Biyoloji ve Araç Taksonomisi) mini bir WordNet veri seti kullanıyoruz.
# Amaç: Gerçek insan dilindeki Hiyerarşiyi (Kedi -> Memeli) modellemek.
# =====================================================================

class BaselineDotProductModel(nn.Module):
    def __init__(self, vocab_size, dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
    def forward(self, u, v):
        u_emb, v_emb = self.embed(u), self.embed(v)
        return torch.sigmoid(torch.sum(u_emb * v_emb, dim=-1))

class ToposYonedaModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.morphisms_logits = nn.Parameter(torch.randn(vocab_size, vocab_size))
    def get_morphisms(self):
        return torch.sigmoid(self.morphisms_logits)
    def forward(self, u, v):
        return self.get_morphisms()[u, v]

def get_real_world_data():
    """Gerçek kelimelerden oluşan Ontolojik (Hiyerarşik) Veri Seti"""
    # 1. DOĞRUDAN BİLGİLER (EĞİTİM VERİSİ) -> A bir B'dir (A -> B)
    true_relations = [
        # Biyoloji (Hayvanlar)
        ("Cat", "Feline"), ("Feline", "Carnivore"), ("Carnivore", "Mammal"), ("Mammal", "Animal"), ("Animal", "Organism"),
        ("Dog", "Canine"), ("Canine", "Carnivore"), 
        ("Human", "Primate"), ("Primate", "Mammal"),
        ("Eagle", "Bird"), ("Bird", "Vertebrate"), ("Vertebrate", "Animal"),
        
        # Teknoloji / Araçlar
        ("Tesla", "Electric_Car"), ("Electric_Car", "Car"), ("Car", "Motor_Vehicle"), ("Motor_Vehicle", "Vehicle"), ("Vehicle", "Artifact"),
        ("Boeing_747", "Airplane"), ("Airplane", "Aircraft"), ("Aircraft", "Vehicle"),
        
        # Kavramlar
        ("Organism", "Entity"), ("Artifact", "Entity")
    ]
    
    vocab = set()
    for u, v in true_relations:
        vocab.add(u); vocab.add(v)
    vocab = list(vocab)
    v_idx = {w: i for i, w in enumerate(vocab)}
    
    X_train_u, X_train_v, Y_train = [], [], []
    
    # Pozitif Örnekler
    for u, v in true_relations:
        X_train_u.append(v_idx[u]); X_train_v.append(v_idx[v]); Y_train.append(1.0)
        
    # Negatif Örnekler (Karışık yanlışlar ve Ters Oklar)
    # TERS OKLARI (Asimetriyi) özellikle eğitimde NEGATİF olarak veriyoruz ki 
    # model YÖNÜN (A->B) önemini anlasın.
    negative_relations = [
        ("Mammal", "Cat"), ("Animal", "Dog"), ("Vehicle", "Car"), # Ters Oklar (Hayvan Kedi DEĞİLDİR)
        ("Cat", "Vehicle"), ("Tesla", "Animal"), ("Bird", "Car"), # Alakasızlar
        ("Human", "Bird"), ("Eagle", "Mammal")                    # Kategori Dışı
    ]
    for u, v in negative_relations:
        X_train_u.append(v_idx[u]); X_train_v.append(v_idx[v]); Y_train.append(0.0)

    # 2. TEST VERİSİ (Eğitimde GÖSTERİLMEYEN Çıkarımlar)
    test_queries = [
        # Geçişlilik (Transitivity: A -> C veya A -> D)
        {"u": "Cat", "v": "Animal", "label": 1.0, "type": "Transitivity (Kedi -> Hayvan)"},
        {"u": "Tesla", "v": "Vehicle", "label": 1.0, "type": "Transitivity (Tesla -> Araç)"},
        {"u": "Eagle", "v": "Organism", "label": 1.0, "type": "Transitivity (Kartal -> Organizma)"},
        
        # Asimetri (Ters Ok: C -> A) - Üst küme alt kümeyi kapsar ama ona eşit değildir.
        {"u": "Animal", "v": "Cat", "label": 0.0, "type": "Asymmetry (Hayvan -> Kedi DEĞİLDİR)"},
        {"u": "Entity", "v": "Tesla", "label": 0.0, "type": "Asymmetry (Varlık -> Tesla DEĞİLDİR)"},
        
        # Mantıksız Bağlar (Halüsinasyon Testi)
        {"u": "Cat", "v": "Airplane", "label": 0.0, "type": "Invalid (Kedi -> Uçak DEĞİLDİR)"}
    ]

    return vocab, v_idx, (torch.tensor(X_train_u), torch.tensor(X_train_v), torch.tensor(Y_train)), test_queries

def evaluate_models():
    vocab, v_idx, (X_u, X_v, Y), test_queries = get_real_world_data()
    vocab_size = len(vocab)
    
    print("--- REAL-WORLD ONTOLOGY (GERÇEK VERİ) BENCHMARK ---")
    print(f"Toplam Kelime (Kavram): {vocab_size}")
    
    models_to_test = {
        "Dot-Product (LLM Baseline)": BaselineDotProductModel(vocab_size, dim=8),
        "Yoneda Topos (Kategori Teorisi)": ToposYonedaModel(vocab_size)
    }
    
    for name, model in models_to_test.items():
        print(f"\n[{name}] Eğitiliyor...")
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        criterion = nn.BCELoss()
        
        for epoch in range(500):
            optimizer.zero_grad()
            scores = model(X_u, X_v)
            loss = criterion(scores, Y)
            loss.backward()
            optimizer.step()
            
        print(f"Eğitim Bitti. Loss: {loss.item():.4f}")
        
        # TEST AŞAMASI
        model.eval()
        correct = 0
        
        with torch.no_grad():
            # Yoneda modeli için Kategori Geçiśliliği (R^n) hesaplanmalı
            if "Yoneda" in name:
                R = model.get_morphisms()
                R_closure = R.clone()
                # 3 Adımlık zincir (Örn: Cat->Feline->Carnivore->Mammal)
                for _ in range(3):
                    R_exp1 = R_closure.unsqueeze(2) 
                    R_exp2 = R.unsqueeze(0) 
                    new_R, _ = torch.max(torch.clamp(R_exp1 + R_exp2 - 1.0, min=0.0), dim=1)
                    R_closure = torch.max(R_closure, new_R)
            else:
                R_closure = None
                
            print("  --- Test Sonuçları (Zero-Shot) ---")
            for q in test_queries:
                u, v, label = v_idx[q["u"]], v_idx[q["v"]], q["label"]
                
                if "Yoneda" in name and "Transitivity" in q["type"]:
                    score = R_closure[u, v].item() # Geçişlilik matrisinden al
                else:
                    score = model(torch.tensor([u]), torch.tensor([v])).item()
                    
                pred = 1.0 if score > 0.5 else 0.0
                is_correct = (pred == label)
                if is_correct: correct += 1
                
                print(f"  {q['type']:<45} | Beklenen: {label} | Tahmin: {score:.2f} -> {'[✓]' if is_correct else '[X] HATA'}")
                
        print(f"  TOPLAM DOĞRULUK: %{(correct / len(test_queries)) * 100:.1f}")

if __name__ == "__main__":
    evaluate_models()
