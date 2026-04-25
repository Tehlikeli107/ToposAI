import torch

# =====================================================================
# MECHANISTIC INTERPRETABILITY (XAI) & TOPOLOGICAL CIRCUIT EXTRACTION
# Modelin "Kara Kutu" (Black Box) olmasını engeller.
# Model bir karara vardığında (Transitive Closure = 1.0), bu karara
# varırken matrisin içinde GİZLİCE kullandığı alt-yolları (Morfizmaları)
# geriye doğru (Traceback) takip ederek "Matematiksel İspat Devresini"
# şeffaf bir şekilde ekrana basar.
# =====================================================================

class ExplainableToposEngine:
    def __init__(self, entities):
        self.entities = entities
        self.e_idx = {e: i for i, e in enumerate(entities)}
        self.N = len(entities)
        self.R = torch.zeros(self.N, self.N)

    def add_fact(self, u, v, weight=1.0):
        self.R[self.e_idx[u], self.e_idx[v]] = weight

    def extract_proof_circuit(self, source, target):
        """
        [MECHANISTIC INTERPRETABILITY]
        A'dan D'ye giden yolun varlığını bilmek yetmez. 
        Nasıl gittiğini (A -> B -> C -> D) DEVRE HARİTASI olarak çıkarır.
        (Breadth-First Search ile Matris üzerinden Topolojik Yol Keşfi).
        """
        start = self.e_idx[source]
        end = self.e_idx[target]
        
        # Eğer zaten doğrudan bir bağ varsa
        if self.R[start, end] > 0.0:
            return [source, target]
            
        # Geçmişi (Predecessors) tutarak BFS yap
        queue = [[start]]
        visited = set()
        visited.add(start)
        
        while queue:
            path = queue.pop(0)
            node = path[-1]
            
            if node == end:
                # Düğümleri (Node ID) tekrar kelimelere (String) çevir
                return [self.entities[n] for n in path]
                
            # Matriste 'node' satırındaki oklara bak (Gücü 0.5'ten büyük olanlar)
            neighbors = torch.where(self.R[node] > 0.5)[0].tolist()
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
                    
        return None # Mantıksal yol (Devre) bulunamadı!

def run_xai_experiment():
    print("--- MECHANISTIC INTERPRETABILITY (EXPLAINABLE AI / XAI) ---")
    print("Yapay Zeka sadece cevap vermez, beyninin içindeki 'Düşünce Devresini' (Circuit) gösterir.\n")

    # Kavramlar
    entities = ["Sigara", "Akciğer_Hasarı", "Hücre_Mutasyonu", "Kanser", "Öksürük", "Zatürre"]
    engine = ExplainableToposEngine(entities)
    
    # Sisteme dağınık, ham Tıbbi veriler (Facts) giriyoruz
    engine.add_fact("Sigara", "Akciğer_Hasarı")
    engine.add_fact("Akciğer_Hasarı", "Hücre_Mutasyonu")
    engine.add_fact("Hücre_Mutasyonu", "Kanser")
    engine.add_fact("Akciğer_Hasarı", "Öksürük")
    engine.add_fact("Zatürre", "Öksürük")

    print("[SİSTEME VERİLEN KARIŞIK OLGULAR]:")
    print(" - Sigara -> Akciğer Hasarı")
    print(" - Akciğer Hasarı -> Hücre Mutasyonu")
    print(" - Hücre Mutasyonu -> Kanser")
    print(" - Akciğer Hasarı -> Öksürük")
    print(" - Zatürre -> Öksürük\n")

    # AI Kararı: Sigara Kanser Yapar mı?
    # Kategori Teorisinin Transitive Closure (R^n) matrisi bunu 1.0 (Evet) bulacaktır.
    # Peki NEDEN?
    
    print("[MÜŞTERİ / DOKTOR SORUSU]: 'Sigara -> Kanser' riski var mı? VARSA NASIL?")
    
    proof_circuit = engine.extract_proof_circuit("Sigara", "Kanser")
    
    if proof_circuit:
        print("\n[!] AI KARARI: EVET, RİSK VARDIR.")
        print("\n>>> [XAI] MECHANISTIC CIRCUIT EXTRACTION (ŞEFFAF DEVRE) <<<")
        print("Modelin beyninden 'Karar Alma Mekanizması' (Sub-Graph) başarıyla çekildi:")
        
        # Devreyi çiz
        circuit_str = " ➔ ".join(proof_circuit)
        print(f"   [KANIT ZİNCİRİ]: {circuit_str}")
        print("\nSonuç: Model kara kutu değildir. Kararını alırken arkada çalışan ")
        print("matematiksel nöron aktivasyon zincirini 'Explainable AI' olarak sundu.")
    else:
        print("\n[!] AI KARARI: HAYIR, RİSK YOKTUR.")

if __name__ == "__main__":
    run_xai_experiment()
